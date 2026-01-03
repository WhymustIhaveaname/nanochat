# Nanochat 可能的升级

本文档记录了 `/mnt/data/ning/code/nanochat` 仓库中实现的功能升级，包括代码实现和 MFU 结果。

---

## 目录

1. [FSDP2 多节点分布式训练](#1-fsdp2-多节点分布式训练)
2. [火山引擎多节点部署](#2-火山引擎多节点部署)
3. [参数化模型配置](#3-参数化模型配置)
4. [权重初始化改进](#4-权重初始化改进)
5. [MFU 优化建议](#5-mfu-优化建议)

---

## 1. FSDP2 多节点分布式训练

### 功能描述

添加 FSDP2 (Fully Sharded Data Parallel) 支持，可以训练更大的模型（d40+），支持跨节点分布式训练。

### 代码实现

#### 1.1 FSDP2 模型包装 (scripts/base_train.py)

```python
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
from nanochat.gpt import GPT, GPTConfig, Block

# 新增配置选项
use_fsdp = False  # use FSDP for memory-efficient training of large models (d40+)
use_muon = True   # use Muon optimizer for matrix params; False = use AdamW (useful for FSDP debugging)

# Wrap model with FSDP2 if enabled (for large models like d40+)
if use_fsdp and ddp:
    print0("Using FSDP2 for memory-efficient distributed training")
    # FSDP2 requires uniform dtype - convert entire model to bfloat16
    model = model.to(dtype=torch.bfloat16)
    # Initialize device mesh for FSDP2
    mesh = init_device_mesh("cuda", (ddp_world_size,))
    fsdp_mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )
    # FSDP2: shard each Block first, then the whole model
    for block in model.transformer.h:
        fully_shard(block, mesh=mesh, mp_policy=fsdp_mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=fsdp_mp_policy)
    orig_model = model  # for FSDP2, orig_model is the sharded model
```

#### 1.2 优化器 FSDP 兼容 (nanochat/gpt.py)

```python
def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                     weight_decay=0.0, use_fsdp=False, use_muon=True):
    # ...

    # When use_fsdp=True, use standard optimizers (FSDP handles distributed sync)
    # When use_fsdp=False and ddp=True, use custom distributed optimizers (original behavior)
    if use_fsdp:
        AdamWFactory = partial(torch.optim.AdamW, fused=True)
    else:
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)

    # Create the optimizer for the linear layers (Muon or AdamW)
    if use_muon:
        if rank == 0:
            print("Using Muon optimizer for matrix parameters")
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        if use_fsdp:
            MuonFactory = Muon
        else:
            MuonFactory = DistMuon if ddp else Muon
        matrix_optimizer = MuonFactory(matrix_params, **muon_kwargs)
    else:
        if rank == 0:
            print("Using AdamW optimizer for matrix parameters (Muon disabled)")
        # Use AdamW for matrix params too, with scaled LR
        adam_matrix_groups = [dict(params=matrix_params, lr=matrix_lr * dmodel_lr_scale)]
        matrix_optimizer = AdamWFactory(adam_matrix_groups, **adamw_kwargs)
```

#### 1.3 FSDP Checkpoint 保存

```python
# For FSDP2, parameters are DTensors - get full tensor for saving
if use_fsdp and ddp:
    from torch.distributed.tensor import DTensor
    sharded_state_dict = orig_model.state_dict()
    model_state_dict = {}
    for key, value in sharded_state_dict.items():
        if isinstance(value, DTensor):
            model_state_dict[key] = value.full_tensor()
        else:
            model_state_dict[key] = value
else:
    model_state_dict = orig_model.state_dict()

# Save checkpoint
torch.save({
    'model': model_state_dict,
    # ...
}, checkpoint_path)
```

### MFU 结果

| 配置 | 并行策略 | 优化器 | device_batch_size | MFU |
|------|---------|--------|-------------------|-----|
| n4 d20 | DDP | Muon | 16 | **~43%** |
| n4 d20 | FSDP | AdamW | 16 | **~37%** |
| n16 d20 | FSDP | AdamW | 16 | 14-32% (不稳定) |
| n16 d48 | FSDP | AdamW | 4 | ~13-15% |

**结论**：FSDP 比 DDP 低约 6% MFU（n4 d20），主要来自通信开销。但 FSDP 可以训练更大模型。

### 已知问题

1. **Muon + FSDP 不兼容**：Muon 的 Newton-Schulz 迭代在 DTensor 上触发大量跨节点通信，导致 MFU 只有 3%
2. **n16 规模 MFU 不稳定**：所有并行策略在 n16 下都有 MFU 高方差问题
3. **Sample Generation 不支持 FSDP**：FSDP 权重是 DTensor，与普通 Tensor 不兼容

---

## 2. 火山引擎多节点部署

### 功能描述

支持在火山引擎 (Volcengine) 平台上进行多节点分布式训练。

### 代码实现

#### 2.1 分布式训练配置 (pretrain.sh)

```bash
# Number of processes/GPUs to use
NPROC_PER_NODE=${MLP_WORKER_GPU:-$(nvidia-smi --list-gpus | wc -l)}

# Distributed training configuration
if [ -n "$MLP_WORKER_NUM" ] && [ "$MLP_WORKER_NUM" -gt 1 ]; then
    TORCHRUN_CMD="torchrun --nnodes=$MLP_WORKER_NUM --node_rank=$MLP_ROLE_INDEX \
                  --nproc_per_node=$NPROC_PER_NODE \
                  --master_addr=$MLP_WORKER_0_HOST --master_port=$MLP_WORKER_0_PORT"
    echo "Volcengine multi-node: NNODES=$MLP_WORKER_NUM, NODE_RANK=$MLP_ROLE_INDEX"
else
    TORCHRUN_CMD="torchrun --standalone --nproc_per_node=$NPROC_PER_NODE"
    echo "Single-node training: NPROC_PER_NODE=$NPROC_PER_NODE"
fi
```

#### 2.2 火山引擎提交脚本 (ning/volc/volc_submit.sh)

```bash
#!/bin/bash
# 火山引擎任务提交脚本
# 用法: NNODES=4 bash ning/volc/volc_submit.sh bash pretrain.sh --depth=32
```

#### 2.3 Dataloader 多节点修复 (nanochat/dataloader.py)

```python
# 原版问题：节点数超过 6-7 时，next(train_loader) 无限挂起
# 原因：每个 parquet 文件独立分配 row groups，当 rank >= num_row_groups 时该 rank 永远读不到数据

# 修复：把所有文件的 row groups 视为全局序列
effective_rank = ddp_rank % total_row_groups  # wrap around 处理
```

---

## 3. 参数化模型配置

### 功能描述

自动根据模型深度配置训练参数，支持 d20-d64 不同规模模型。

### 代码实现 (pretrain.sh)

```bash
# Model configuration: auto-configure based on depth
MODEL_DEPTH=${MODEL_DEPTH:-20}

case $MODEL_DEPTH in
    20)
        DATASET_SHARDS=240
        DEFAULT_DEVICE_BATCH_SIZE=16
        ;;
    22)
        DATASET_SHARDS=300
        DEFAULT_DEVICE_BATCH_SIZE=16
        ;;
    26)
        DATASET_SHARDS=500
        DEFAULT_DEVICE_BATCH_SIZE=8
        ;;
    32)
        DATASET_SHARDS=800
        DEFAULT_DEVICE_BATCH_SIZE=4
        ;;
    40)
        DATASET_SHARDS=1500
        DEFAULT_DEVICE_BATCH_SIZE=1
        ;;
    44)
        DATASET_SHARDS=2000
        DEFAULT_DEVICE_BATCH_SIZE=1
        ;;
    48)
        DATASET_SHARDS=2500
        DEFAULT_DEVICE_BATCH_SIZE=1
        ;;
    56)
        DATASET_SHARDS=3500
        DEFAULT_DEVICE_BATCH_SIZE=1
        ;;
    64)
        DATASET_SHARDS=5000
        DEFAULT_DEVICE_BATCH_SIZE=1
        ;;
esac

# 模式切换支持
# bash pretrain.sh                     # pretrain only (default)
# bash pretrain.sh --posttrain         # posttrain only (skip pretrain)
# bash pretrain.sh --fullrun           # pretrain + posttrain
```

### 模型规模参考

| 深度 | 参数量 | n_embd | DATASET_SHARDS | 训练数据量 |
|------|--------|--------|----------------|-----------|
| d12 | 0.14B | 768 | - | - |
| d20 | 0.5B | 1280 | 240 | ~24GB |
| d22 | 0.6B | 1408 | 300 | ~30GB |
| d26 | 1.0B | 1664 | 500 | ~50GB |
| d32 | 1.7B | 2048 | 800 | ~80GB |
| d40 | 3.3B | 2560 | 1500 | ~150GB |
| d48 | 5.6B | 3072 | 2500 | ~250GB |
| d56 | 8.9B | 3584 | 3500 | ~350GB |
| d64 | 13.2B | 4096 | 5000 | ~500GB |

---

## 4. 权重初始化改进

### 功能描述

使用论文 [arXiv:2310.17813](https://arxiv.org/abs/2310.17813) 的初始化方案替换原版的 Uniform 分布初始化。

### 代码实现 (nanochat/gpt.py)

#### 原版初始化

```python
def init_weights(self):
    # Embedding and unembedding
    torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
    torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

    # Transformer blocks: uniform init with bound = sqrt(3) * std
    n_embd = self.config.n_embd
    s = 3**0.5 * n_embd**-0.5  # sqrt(3) multiplier for Uniform to match Normal std
    for block in self.transformer.h:
        torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
        torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
        torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
        torch.nn.init.zeros_(block.attn.c_proj.weight)
        torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
        torch.nn.init.zeros_(block.mlp.c_proj.weight)
```

#### ning 改进版初始化

```python
def init_weights(self):
    self.apply(self._init_weights)
    # zero out classifier weights
    torch.nn.init.zeros_(self.lm_head.weight)
    # zero out c_proj weights in all blocks
    for block in self.transformer.h:
        torch.nn.init.zeros_(block.mlp.c_proj.weight)
        torch.nn.init.zeros_(block.attn.c_proj.weight)
    # init the rotary embeddings
    self.init_rotary_embeddings()
    # Cast the embeddings from fp32 to bf16
    self.transformer.wte.weight.data = self.transformer.wte.weight.data.to(torch.bfloat16)

def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        # https://arxiv.org/pdf/2310.17813
        fan_out = module.weight.size(0)
        fan_in = module.weight.size(1)
        std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)
```

### 差异分析

| 特性 | 原版 | ning 改进版 |
|------|------|------------|
| 分布类型 | Uniform | Normal |
| std 计算 | `1/sqrt(n_embd)` | `1/sqrt(fan_in) * min(1, sqrt(fan_out/fan_in))` |
| lm_head 初始化 | `std=0.001` | zeros |
| 遍历方式 | 手动遍历每层 | `self.apply()` 递归 |

---

## 5. MFU 优化建议

### 当前 MFU 状态

nanochat 在 H100 上的 MFU 约为 **45%**，处于行业正常水平：

| 项目 | 年份 | MFU |
|------|------|-----|
| GPT-3 (OpenAI) | 2020 | 19.6% |
| Megatron-LM | 2021 | ~30% |
| PaLM (Google) | 2022 | 46.2% |
| **nanochat** | 2024 | **~45%** |
| MegaScale (ByteDance) | 2024 | ~55% |

### MFU 损失分解

```
理论峰值 (H100 BF16)              100% (989 TFLOPS)
├── 非 MatMul 操作                 -20%  (softmax, RMSNorm, ReLU²)
├── 通信开销                       -15%  (跨节点 all-reduce)
├── 内存带宽限制                   -10%  (embedding, activation)
├── Kernel 开销 & 同步              -5%  (CUDA kernel launch)
└── 其他                            -5%  (编译、调度)
= 实际 MFU                        ~45%
```

### 优化建议

| 优化方向 | 工作量 | MFU 提升 | 风险 | 推荐度 |
|---------|--------|----------|------|--------|
| FlashAttention-3 | 1-2天 | +10-15% | 低 | ⭐⭐⭐⭐⭐ |
| FSDP 迁移 | 1-2周 | +5-8% | 中 | ⭐⭐⭐⭐ |
| Tensor Parallelism | 2-4周 | +5-10% | 中 | ⭐⭐⭐ |
| FP8 训练 | 1-2月 | +50-100%* | 高 | ⭐⭐ |

*FP8 的提升是吞吐量，不是 MFU（MFU 计算基准会变成 1979 TFLOPS）

### FlashAttention-3 升级示例

```python
# 安装
# pip install flash-attn --no-build-isolation

# nanochat/gpt.py 修改
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

class CausalSelfAttention(nn.Module):
    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        if FLASH_ATTN_AVAILABLE and kv_cache is None:
            # 使用 FlashAttention-3（训练时）
            y = flash_attn_func(q, k, v, causal=True)
        else:
            # 回退到 PyTorch SDPA（推理时或无 flash-attn）
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2)

        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y)
```

**预期效果**：MFU 从 ~45% 提升到 **55-60%**。

---

## 训练结果示例 (d22 模型)

来自 ning 的 `report.md`：

### 训练配置

- **模型**: d22 (707M 参数)
- **硬件**: 8x NVIDIA H100 80GB HBM3
- **训练时间**: 15h5m
- **训练 tokens**: 14.16B
- **MFU**: 15.34%

### 评估结果

| 阶段 | CORE | ARC-Easy | ARC-Challenge | GSM8K | MMLU | ChatCORE |
|------|------|----------|---------------|-------|------|----------|
| BASE | 0.2236 | - | - | - | - | - |
| MID | - | 0.4503 | 0.3285 | 0.0371 | 0.3293 | 0.2666 |
| SFT | - | 0.4777 | 0.3183 | 0.0607 | 0.3352 | 0.2709 |
| RL | - | - | - | 0.0872 | - | - |

---

## 参考来源

- ning 仓库: `/mnt/data/ning/code/nanochat`
- `ning/docs/fsdp_dev_notes.md`: FSDP 开发笔记
- `ning/docs/mfu_analysis.md`: MFU 分析与优化指南
- `report.md`: 训练报告

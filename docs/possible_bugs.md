# Nanochat 可能的 Bug 及修复

本文档记录了在 `/mnt/data/ning/code/nanochat` 仓库中发现并修复的原版 nanochat 可能存在的 bug。

---

## 1. CORE Metric 评估导致内存泄漏 (OOM)

### 问题描述

训练过程中，每次 CORE metric 评估后 GPU 内存会突然增加，导致后续训练步骤 OOM（Out of Memory）。

### 原因分析

CORE 评估使用 `orig_model`（未编译的原始模型），而不是 `torch.compile()` 编译后的模型。`orig_model` 在 forward 时会创建独立的：
- cuDNN workspace
- CUDA buffers
- 中间激活值缓存

这些资源与编译后模型的资源是分开管理的，PyTorch 不会自动释放它们，导致内存持续增长。

### Bug 代码 (scripts/base_train.py)

```python
# 原版代码 - 评估后未清理 GPU 缓存
if master_process and (last_step or (step > 0 and step % core_metric_every == 0)):
    results = evaluate_model(orig_model, tokenizer, device, max_per_task=core_metric_max_per_task)
    # 问题：orig_model 创建的 CUDA buffers 没有被释放
    # 后续训练时内存不足导致 OOM
```

### 修复代码

```python
# 修复后 - 评估后清理 GPU 缓存
if master_process and (last_step or (step > 0 and step % core_metric_every == 0)):
    # Clear CUDA cache before evaluation to free memory from training
    torch.cuda.empty_cache()

    results = evaluate_model(orig_model, tokenizer, device, max_per_task=core_metric_max_per_task)

    # Clear CUDA cache created by orig_model to prevent OOM during subsequent training
    # orig_model creates its own cuDNN workspace and buffers separate from compiled model
    torch.cuda.empty_cache()
```

### 同样的问题也出现在 Sample Generation

```python
# 原版代码
if master_process and (last_step or (step > 0 and step % sample_every == 0)):
    samples = generate_samples(orig_model, ...)
    # 同样的问题：未清理缓存

# 修复后
if master_process and (last_step or (step > 0 and step % sample_every == 0)):
    samples = generate_samples(orig_model, ...)
    # Clear CUDA cache created by orig_model (same reason as CORE eval above)
    torch.cuda.empty_cache()
```

### 影响范围

- 长时间训练（多次触发 CORE eval）
- 大模型训练（显存接近上限）
- 多 GPU 训练（每个 GPU 都会积累未释放的缓存）

---

## 2. Rotary Embedding 输出 dtype 不一致

### 问题描述

`apply_rotary_emb()` 函数可能返回与输入不同的 dtype，导致后续计算出现精度问题或类型不匹配错误。

### 原因分析

在 `apply_rotary_emb` 中，`cos` 和 `sin` 张量可能是 `float32` 类型（因为频率计算需要高精度），而输入 `x` 是 `bfloat16`。计算过程中会发生隐式类型提升，但返回时没有显式转换回原始类型。

### Bug 代码 (nanochat/gpt.py)

```python
def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings to x."""
    d = x.size(-1) // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last dim into two halves
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)  # 问题：返回的 dtype 可能与 x 不同
```

### 修复代码

```python
def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings to x."""
    d = x.size(-1) // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last time into two halves
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)  # re-assemble
    out = out.to(x.dtype)  # ensure input/output dtypes match
    return out
```

### 影响范围

- 混合精度训练（autocast 环境下）
- 不同 GPU 架构（dtype 处理可能有差异）
- 梯度计算（dtype 不一致可能导致梯度异常）

---

## 3. CUDA 内存分配环境变量名称错误

### 问题描述

环境变量名称拼写错误，导致 `expandable_segments` 配置未生效。

### 原因分析

PyTorch 的 CUDA 内存分配器配置环境变量名称是 `PYTORCH_CUDA_ALLOC_CONF`，不是 `PYTORCH_ALLOC_CONF`。少了 `CUDA_` 前缀导致配置被忽略。

### Bug 代码 (scripts/base_train.py)

```python
# 错误的环境变量名
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
```

### 修复代码

```python
# 正确的环境变量名
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

### 影响范围

- 内存碎片问题：`expandable_segments` 未启用会导致更多内存碎片
- 大模型训练可能更早遇到 OOM
- 内存使用效率降低

### 参考

- [PyTorch CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)

---

## 总结

| Bug | 严重程度 | 触发条件 | 修复难度 |
|-----|---------|---------|---------|
| CORE Metric 内存泄漏 | 🔴 高 | 长时间训练 + 大模型 | 低 |
| Rotary Embedding dtype | 🟡 中 | 混合精度训练 | 低 |
| 环境变量名称错误 | 🟡 中 | 所有训练 | 低 |

---

## 参考来源

- ning 仓库: `/mnt/data/ning/code/nanochat`
- `ning/docs/work_log.md`: CORE Metric Memory Leak Bug
- `ning/docs/fsdp_dev_notes.md`: 开发调试记录
- Git diff 对比分析

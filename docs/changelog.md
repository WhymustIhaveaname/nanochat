# 更新日志

## 2025-01-04

### 修复 Chinchilla 比例计算

**问题**：原代码使用总参数量（包括 embedding 和 lm_head）来计算 Chinchilla 训练 token 比例，但 embedding 层的参数量与词表大小直接相关，而非模型的学习能力。使用不同词表大小的 tokenizer 会导致计算出不同的训练 token 数。

**修复**：现在使用**非 embedding 参数量**（排除 `wte` 和 `lm_head`）来计算 Chinchilla 比例。

**修改文件**：`scripts/base_train.py`

**修改内容**：
```python
# 之前
num_params = sum(p.numel() for p in model.parameters())
target_tokens = target_param_data_ratio * num_params

# 之后
num_params = sum(p.numel() for p in model.parameters())
# 通过遍历 named_parameters 跳过 wte 和 lm_head，避免 tied weights 时重复计算
num_params_non_embedding = sum(
    p.numel() for name, p in orig_model.named_parameters()
    if "wte" not in name and "lm_head" not in name
)
# 验证：非 embedding 参数量应接近理论值 12 * n_layer * n_embd^2
theoretical_params = 12 * num_layers * model_dim ** 2
assert abs(num_params_non_embedding / theoretical_params - 1.0) < 0.02
target_tokens = target_param_data_ratio * num_params_non_embedding
```

**日志输出变化**：
```
# 之前
Number of parameters: 28,901,376

# 之后
Number of parameters: 28,901,376 (non-embedding: 18,234,880)
Tokens : Params (non-embedding) ratio: 20.00
```

---

### Training Logger (CSV 日志)

新增 `nanochat/training_logger.py` 模块，用于保存训练日志以便 scaling law 分析。

**设计**：
- 接口参考 wandb，传入 dict 即可记录
- 自动管理 `_timestamp` 和 `_walltime` 字段
- 日志保存到 `$NANOCHAT_BASE_DIR/training_logs/` 目录
- 文件名格式：`{run_name}_{timestamp}.csv`

**使用**：
```python
from nanochat.training_logger import TrainingLogger

logger = TrainingLogger(log_dir="path/to/logs", run_name="d20_exp1")
logger.log({"step": 100, "train_loss": 2.345, "mfu": 45.2})
logger.close()
```

**修改文件**：`scripts/base_train.py`
- 添加 logger 初始化和 `log()` 调用

---

## 2025-01-05

### 优化器和学习率调度支持

新增可配置参数，支持纯 AdamW 优化器和 cosine decay 学习率调度。

**新增配置项**（`scripts/base_train.py`）：
```python
use_muon = True      # 使用 Muon 优化器 (False = 全部用 AdamW)
lr_schedule = "linear"  # 学习率衰减: "linear" 或 "cosine"
```

**使用示例**：
```bash
# 纯 AdamW + cosine decay + warmup 到 10%
python -m scripts.base_train --use_muon=False --lr_schedule=cosine --warmup_ratio=0.1 --warmdown_ratio=0.9 --final_lr_frac=0.1
```

**修改文件**：
- `nanochat/gpt.py`：`setup_optimizers()` 新增 `use_muon` 参数
  - `use_muon=False` 时，dim < 2 的参数不应用 weight_decay
  - 打印各参数组的 lr 和 weight_decay 便于验证
- `scripts/base_train.py`：新增 `use_muon` 和 `lr_schedule` 配置项

---

### Training Logger 增强

- 支持 `subdir` 参数，日志保存到 `training_logs/{run}/{run_name}_{mmdd}.csv`
- 支持 `metadata` 参数，在 CSV 开头写入 `# key=value` 格式的注释
- 新增 `print_every` 参数控制打印频率（默认每 10 步）
- 文件名 timestamp 格式简化为 `mmdd`

---

### wandb 日志改进

- `run_name` 统一为 `f"d{depth}_{run}"`，与 CSV logger 一致
- wandb 训练日志频率改为每步记录，与 CSV logger 同步
- 所有 `wandb_run.log()` 添加 `step=step` 参数和 `global_step` 字段

---

## TODO

- [ ] MFU 计算硬编码了 H100 的理论峰值 (989 TFLOPs/s)，应该根据 GPU 型号自动查询对应的 FLOPs

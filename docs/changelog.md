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

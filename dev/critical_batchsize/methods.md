# Critical Batch Size 测量方法

## 前提

**Critical batch size $B_c$ 与 loss 相关**，所以我们测量的是某个特定 loss 水平附近的 $B_c$。

---

## 方法一：定义法（基于 $\mathcal{B}_{\text{noise}}$）

### 理论基础

论文公式 2.1-2.8 定义了 noise scale $\mathcal{B}_{\text{noise}}$：

$$\mathcal{B}_{\text{noise}} = \frac{\text{tr}(H\Sigma)}{G^T H G}$$

最优学习率与 batch size 的关系：

$$\epsilon_{\text{opt}}(B) = \frac{\epsilon_{\max}}{1 + \mathcal{B}_{\text{noise}}/B}$$

单步 loss 下降的理论公式：

$$\Delta L(B, \epsilon) = \epsilon |G|^2 - \frac{1}{2}\epsilon^2 \cdot \text{tr}(H\Sigma) \left(\frac{1}{\mathcal{B}_{\text{noise}}} + \frac{1}{B}\right)$$

### 测量步骤

#### Step 1: 固定优化时刻

选择一个确定的训练时刻（checkpoint），此时 eval loss 可以测得很准。

#### Step 2: 对每个 B，扫描 $\epsilon$ 找最优

对于给定的 batch size $B$：
1. 用不同的学习率 $\epsilon$ 做**一步**随机梯度下降
2. 测量 eval loss 的下降 $\Delta L(B, \epsilon)$
3. 拟合二次函数找到最优 $\epsilon_{\text{opt}}(B)$

> **注意**：必须用 **eval loss**，不能用 training loss！

由于 $\Delta L$ 是 $\epsilon$ 的二次函数，用二次回归找最低点很安全。

可选：打印回归系数 $|G|^2$ 和 $\text{tr}(H\Sigma)(1/\mathcal{B}_{\text{noise}} + 1/B)$，检查趋势是否符合预期。

#### Step 3: 线性回归得到 $\mathcal{B}_{\text{noise}}$

将公式变换为线性形式：

$$\frac{1}{\epsilon_{\text{opt}}} = \frac{1}{\epsilon_{\max}} + \frac{\mathcal{B}_{\text{noise}}}{\epsilon_{\max}} \cdot \frac{1}{B}$$

$$\mathcal{B}_{\text{noise}} = \frac{\text{斜率}}{\text{截距}}$$

这是标准的**线性最小二乘回归**。

### 计算复杂度分析

假设：
- 5 个 batch size
- 每个 B 试 5 个 learning rate（或用二次回归只需 3 个点）
- 每个实验需要 eval loss

**总计**：$5 \times 5 = 25$ 次 eval

主要开销在 **eval loss 计算**，梯度下降本身的复杂度可以忽略。

可选增强：每个 B 用不同的数据 sample 取平均，会增加实验次数。

---

## 方法二：Simple Noise Scale（基于 $\mathcal{B}_{\text{simple}}$）

### 理论基础

假设 Hessian $H \propto I$（well-conditioned），则 $\mathcal{B}_{\text{noise}}$ 简化为：

$$\mathcal{B}_{\text{simple}} = \frac{\text{tr}(\Sigma)}{|G|^2}$$

梯度范数的期望（公式 A.1）：

$$\mathbb{E}[|G_{\text{est}}|^2] = |G|^2 + \frac{\text{tr}(\Sigma)}{B}$$

### 测量步骤

#### Step 1: 固定优化时刻

同方法一。

#### Step 2: 不同 B 计算梯度范数

对于不同的 batch size $B$，计算 $|G_{\text{est}}|^2$（仅需前向+反向，无需 eval）。

#### Step 3: 线性回归得到 $\mathcal{B}_{\text{simple}}$

对 $(1/B, |G_{\text{est}}|^2)$ 做线性回归：

$$\mathcal{B}_{\text{simple}} = \frac{\text{斜率}}{\text{截距}} = \frac{\text{tr}(\Sigma)}{|G|^2}$$

### 计算复杂度分析

假设 5 个 batch size：**仅 5 次梯度计算**（前向+反向）。

对比方法一（25 次 eval）：小 $5 \times \frac{\text{eval}}{\text{梯度计算}}$ 倍。

### 可选增强

- 每个 B 用不同数据 sample 取平均，提高精度
- 可分别统计**全局** $|G_{\text{est}}|^2$ 和**每层** $|G_{\text{est}}|^2$，观察是否一致

---

## 方法三：积分法（Full Training Runs）

### 与前两种方法的区别

- 方法一、二是**微分法**：在训练的某一瞬间测量
- 方法三是**积分法**：对整段训练过程积分

### 理论基础

用不同 batch size $B$ 训练到目标 loss，记录：
- $S$：达到目标 loss 所需的 step 数
- $E = B \times S$：消耗的数据量

公式 2.11：

$$\frac{S}{S_{\min}} - 1 = \left(\frac{E}{E_{\min}} - 1\right)^{-1}$$

定义 critical batch size（公式 2.12）：

$$\mathcal{B}_{\text{crit}} = \frac{E_{\min}}{S_{\min}}$$

### 测量步骤

1. 选择目标 loss $L_{\text{target}}$
2. 用不同 $B$ 训练，记录 $(S, E)$
3. 拟合公式 2.11 得到 $S_{\min}$、$E_{\min}$
4. 计算 $\mathcal{B}_{\text{crit}} = E_{\min} / S_{\min}$

### 线性拟合推导

公式 2.11 可以变形为线性形式，便于使用线性最小二乘拟合：

$$\left(\frac{S}{S_{\min}} - 1\right)\left(\frac{E}{E_{\min}} - 1\right) = 1$$

$$\frac{SE}{S_{\min}E_{\min}} = \frac{S}{S_{\min}} + \frac{E}{E_{\min}}$$

$$SE = S \cdot E_{\min} + E \cdot S_{\min}$$

$$1 = \frac{E_{\min}}{E} + \frac{S_{\min}}{S}$$

即 $1/S$ 与 $1/E$ 呈线性关系：

$$\frac{1}{S} = \frac{1}{S_{\min}} - \frac{E_{\min}}{S_{\min}} \cdot \frac{1}{E}$$

拟合 $1/S = a + b \cdot (1/E)$，则：
- $S_{\min} = 1/a$
- $E_{\min} = -b/a$
- $\mathcal{B}_{\text{crit}} = E_{\min}/S_{\min} = -b$

### 计算复杂度

需要完整训练多个模型，**远大于微分法**。

### 开放问题

理论上 $\mathcal{B}_{\text{crit}} \approx \mathcal{B}_{\text{noise}}$，但积分法测的是一段 loss 区间 $[L_1, L_2]$ 的平均效应。

**好奇**：从不同的 $L_1$ 积分到不同的 $L_2$，会如何影响 $\mathcal{B}_{\text{crit}}$？

---

## 开放问题

1. **不同 optimizer 的影响**：SGD / AdamW / Muon 的 $\mathcal{B}_{\text{noise}}$ 是否不同？
2. **不同任务的影响**：图片分类 vs Transformer 语言模型，$\mathcal{B}_{\text{noise}}$ 差多少？
3. **$\mathcal{B}_{\text{simple}}$ 与层数的关系**：全局 $|G_{\text{est}}|^2$ vs 每层 $|G_{\text{est}}|^2$，noise scale 是否一致？
4. **$\mathcal{B}_{\text{crit}}$ 与积分区间的关系**：从 $L_1$ 积分到 $L_2$，不同的 $(L_1, L_2)$ 如何影响 $\mathcal{B}_{\text{crit}}$？

---

## 实验设计：最小化实验量

对于给定的 optimizer 和任务，如何用最少的实验覆盖三种方法？

### Phase 1: $\mathcal{B}_{\text{crit}}$ 实验（同时为后续实验铺垫）

1. 用不同 $B$ 完整训练，记录：
   - Training loss（密）
   - Eval loss（稀）
2. 每次 eval 时保存 checkpoint
3. 这些 checkpoint 即为 $\mathcal{B}_{\text{noise}}$ 和 $\mathcal{B}_{\text{simple}}$ 的"固定优化时刻"

> **为什么 $\mathcal{B}_{\text{crit}}$ 可以用 training loss？**
> 积分法拟合整条曲线，噪声被平均掉；而定义法测单步 $\Delta L$，必须用 eval loss。

### Phase 2: $\mathcal{B}_{\text{noise}}$ + $\mathcal{B}_{\text{simple}}$ 实验

从 Phase 1 的 checkpoint 出发：

1. 做 $\mathcal{B}_{\text{noise}}$：不同 $B$、不同 $\epsilon$，测 eval loss 的 $\Delta L$
2. **同时**统计 $|G_{\text{est}}|^2$，顺便得到 $\mathcal{B}_{\text{simple}}$

#### $\mathcal{B}_{\text{simple}}$ 的额外数据点

对于每个 $B$，有两个层次的梯度：
- **Micro batch**（聚合前）：每个 device 的梯度，多次平均
- **Full $B$**（聚合后）：all-reduce 后的梯度

所以如果 $\mathcal{B}_{\text{noise}}$ 测 5 个 $B$，$\mathcal{B}_{\text{simple}}$ 可以得到 $5 \times 2 = 10$ 个数据点。

---

## 估算实验范围

Kaplan scaling law 给出 $\mathcal{B}_{\text{crit}}$ 与 loss 的关系：

$$\mathcal{B}_{\text{crit}} = \frac{B^*}{L^{1/\alpha_B}}$$

其中 $B^* \approx 2 \times 10^8$ tokens，$\alpha_B \approx 0.21$。

| Loss | $\mathcal{B}_{\text{crit}}$ |
|------|----------------------------|
| 6 | ~40K |
| 5 | ~94K |
| 4 | ~267K |
| 3 | ~1M |

---

## nanochat 参考配置

| 参数 | 值 |
|------|-----|
| depth | 20 |
| model_dim | 1280 |
| batch_size | 524K (2^19) |

LR（针对上述配置调优）：

| 参数组 | LR | 优化器 | 备注 |
|--------|-----|--------|------|
| embedding | 0.3 | Adam | input muP |
| unembedding | 0.004 | Adam | output muP |
| matrix | 0.02 | Muon | |
| scalar | 0.5 | Adam | 我们不训这个 |

muP 表格：

| | Input weights & all biases | Hidden weights | Output weights |
|---|---|---|---|
| Init. Var. | 1/fan_in | 1/fan_in | 1/fan_in² |
| SGD LR | fan_out | 1 | 1/fan_in |
| Adam LR | 1 | 1/fan_in | 1/fan_in |

**depth=4 时的 LR（按 muP 缩放，基于 batch_size=524K）**：

| 参数组 | SGD | Adam |
|--------|-----|------|
| embd | 0.06 | 0.3 |
| matrix | 0.02 | 0.1 |
| lm_head | 0.02 | 0.02 |

> 备注：对于 AdamW，$\text{lr} \propto \sqrt{\text{batch\_size}}$，上表基于 batch_size=524K

---

## 训练命令

### depth=4 (Adam LR 基准: embd=0.3, matrix=0.1, lm_head=0.02)

```bash
# bs=16384, scale=0.177
torchrun --standalone --nproc_per_node=2 -m dev.critical_batchsize.transformer.train \
  depth=4 batch_size=16384 lr_embd=0.053 lr_matrix=0.018 lr_lm_head=0.0035

# bs=32768, scale=0.25
torchrun --standalone --nproc_per_node=2 -m dev.critical_batchsize.transformer.train \
  depth=4 batch_size=32768 lr_embd=0.075 lr_matrix=0.025 lr_lm_head=0.005
```

### depth=8 (Adam LR 基准: embd=0.3, matrix=0.05, lm_head=0.01)

```bash
# bs=16384, scale=0.177
torchrun --standalone --nproc_per_node=2 -m dev.critical_batchsize.transformer.train \
  depth=8 batch_size=16384 lr_embd=0.053 lr_matrix=0.0088 lr_lm_head=0.0018

# bs=32768, scale=0.25
torchrun --standalone --nproc_per_node=2 -m dev.critical_batchsize.transformer.train \
  depth=8 batch_size=32768 lr_embd=0.075 lr_matrix=0.013 lr_lm_head=0.0025
```

---

## `measure.py` 顶层设计

实现方法 1（$\mathcal{B}_{\text{noise}}$）和方法 2（$\mathcal{B}_{\text{simple}}$）。

### 设计原则

1. **从 Phase 1 的 checkpoint 出发**：复用 `train.py` 保存的模型状态
2. **方法 2 是方法 1 的副产品**：计算梯度时顺便记录 $|G_{\text{est}}|^2$
3. **使用相同的 optimizer**：$\mathcal{B}_{\text{noise}}$ 的理论公式是 optimizer-specific 的，必须用 `train.py` 中相同的 optimizer 类型

### 数据加载

数据集按 parquet 文件划分，需要区分三种用途：

| 用途 | 数据来源 | split 参数 |
|------|----------|------------|
| Phase 1 训练 | 前 N-2 个 parquet | `"train"` |
| 计算梯度 | **倒数第二个 parquet** | `"heldout"` |
| 计算 eval loss | 最后一个 parquet | `"val"` |

**为什么不能用 training data 算梯度**：模型已经在这些数据上优化过，梯度有 bias。

**为什么不能用 eval data 算梯度**：相当于在 eval 时 cheating。

需要扩展 dataloader 支持 `split="heldout"`：

```python
# nanochat/dataloader.py 中的修改
parquet_paths = list_parquet_files()
if split == "train":
    parquet_paths = parquet_paths[:-2]  # 前 N-2 个
elif split == "heldout":
    parquet_paths = parquet_paths[-2:-1]  # 倒数第二个
elif split == "val":
    parquet_paths = parquet_paths[-1:]  # 最后一个
```

> **注意**：这意味着 Phase 1 的 `train.py` 也需要改用 `[:-2]` 而不是 `[:-1]`，否则 training data 会包含 heldout parquet。

### 输入参数

`measure.py` 有自己的 `config.yaml`（主配置）：

```yaml
# 必需：run 目录（包含 config.yaml, loss_eval.csv, checkpoints/）
run_dir: dev/critical_batchsize/transformer/outputs/02-02_d4_adamw_8192

# Optimizer（可以和训练时不同）
optimizer: sgd  # sgd / adamw

# Learning rate 扫描范围（绝对值）
lr_min: 0.001
lr_max: 1.0
lr_steps: 7  # 对数均匀分布

# Batch sizes（tokens）
batch_sizes: [16384, 32768, 65536, 131072, 262144]
```

**只优化 transformer layers**（embedding 和 lm_head 冻结），所有参数用同一个 lr。

从 `run_dir/config.yaml` 继承：
- `depth`（决定 model_dim, num_heads）
- `seq_len`（决定 micro batch 大小）
- `eval_seq_len`, `max_eval_tokens`

代码会：
1. 从 `run_dir/loss_eval.csv` 读取每个 checkpoint 的 baseline eval loss
2. 遍历 `run_dir/checkpoints/*.pt` 中的所有 checkpoint

### 核心流程

**Phase 1: 初始化**
1. 加载 `run_dir/config.yaml`，创建模型（冻结 embedding 和 lm_head）
2. 加载 `run_dir/loss_eval.csv`，建立 step → baseline eval loss 的映射
3. 列出 `run_dir/checkpoints/*.pt` 中的所有 checkpoint

**Phase 2: 对每个 checkpoint**
1. 加载 checkpoint 权重，记录其 step 和 baseline eval loss $L_0$
2. 对每个 batch_size $B$：
   - 从 **held-out data（倒数第二个 parquet）** 采样 $B$ tokens，计算梯度 $g$
   - 记录梯度范数 $|g|^2$（用于 $\mathcal{B}_{\text{simple}}$）
   - 对每个 learning_rate $\epsilon$：
     - 恢复模型到 checkpoint 状态
     - 用 optimizer 执行一步更新
     - 在 **validation data（最后一个 parquet）** 上计算新 eval loss $L(B, \epsilon)$
   - 二次回归 $L(\epsilon)$ 找 $\epsilon_{\text{opt}}(B)$

**Phase 3: 拟合**
1. 线性回归 $1/\epsilon_{\text{opt}}$ vs $1/B$ → $\mathcal{B}_{\text{noise}}$
2. 线性回归 $|g|^2$ vs $1/B$ → $\mathcal{B}_{\text{simple}}$

### 关键实现细节

#### 梯度计算与复用

```python
def compute_gradient(model, loader, batch_size, seq_len):
    """计算一个 batch 的梯度，返回梯度和梯度范数。"""
    model.zero_grad()
    num_micro = batch_size // seq_len
    for _ in range(num_micro):
        x, y = next(loader)
        loss = model(x, y)
        (loss / num_micro).backward()

    # 记录梯度范数（用于 B_simple）
    grad_norm_sq = sum((p.grad ** 2).sum() for p in model.parameters())

    # 深拷贝梯度，用于在不同 lr 之间复用
    # （恢复 checkpoint 时 p.grad 会丢失）
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return grads, grad_norm_sq.item()
```

#### Optimizer 状态初始化与单步更新

两个 tricky 的问题：
1. **Optimizer 有状态**：SGD 有 momentum，AdamW 有一阶矩和二阶矩。每次实验必须创建**全新的 optimizer**。
2. **AdamW 二阶矩 warmup**：如果从 0 开始，第一步行为和稳态不同。需要用当前梯度初始化状态。

```python
def init_optimizer_state(optimizer, optimizer_type):
    """根据 optimizer 类型初始化状态。

    - SGD (momentum=0)：无状态，不需要初始化
    - AdamW：用当前梯度初始化 exp_avg 和 exp_avg_sq

    注意：SGD with momentum 没有简单的初始化能让状态一步后不变，
    所以 measure.py 只支持 SGD (momentum=0) 或 AdamW。
    """
    if optimizer_type == "sgd":
        for group in optimizer.param_groups:
            assert group.get('momentum', 0) == 0, "SGD must have momentum=0"
        return

    elif optimizer_type == "adamw":
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = optimizer.state[p]
                g = p.grad
                # 初始化后一步更新状态应不变（可用于验证）
                state['exp_avg'] = g.clone()
                state['exp_avg_sq'] = (g ** 2).clone()
                state['step'] = torch.tensor(0)
```

#### 二次回归找最优 LR

```python
def find_optimal_lr(lrs, losses):
    """
    L(ε) = L₀ - a·ε + b·ε²  (理论上 a > 0, b > 0)
    最优点：ε_opt = a / (2b)
    """
    # 拟合 L = c + a·ε + b·ε²（b 应为正，a 应为负）
    eps = np.array(lrs)
    L = np.array(losses)
    A = np.vstack([np.ones_like(eps), eps, eps**2]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, L, rcond=None)
    c, a, b = coeffs
    if b <= 0:
        warn("二次项系数非正，拟合可能有问题")
    eps_opt = -a / (2 * b)
    return eps_opt, c, a, b
```

### 输出

输出到 `run_dir/measure/`：

```
measure/
├── raw_data.csv             # 原始数据 (step, batch_size, lr, loss, grad_norm_sq)
├── results.csv              # 拟合结果 (step, eval_loss, B_noise, B_noise_r2, B_simple, B_simple_r2)
├── step_00100_contour.png   # Loss 等高线图（横轴 B，纵轴 lr）
├── step_00100_fit.png       # 左：1/ε_opt vs 1/B（B_noise）  右：|g|² vs 1/B（B_simple）
└── ...
```

### 与 `train.py` 的配置对齐

`measure.py` 从 `run_dir/config.yaml` 继承：
- `depth`（决定 model_dim, num_heads）
- `seq_len`（决定 micro batch 大小）
- `eval_seq_len`, `max_eval_tokens`

以下参数由 `measure.py` 自己的 config 指定：
- `optimizer`（可以和训练时不同）
- `lr_min`, `lr_max`, `lr_steps`
- `batch_sizes`

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

$$\Delta L(B, \epsilon) = \epsilon |G|^2 + \frac{1}{2}\epsilon^2 \cdot \text{tr}(H\Sigma) \left(\frac{1}{\mathcal{B}_{\text{noise}}} + \frac{1}{B}\right)$$

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

## 文件结构

```
dev/critical_batchsize/
├── methods.md                 # 本文档
│
├── transformer/               # Transformer 语言模型实验
│   ├── train.py               # Phase 1: 训练 + 记录 loss + 保存 checkpoint
│   ├── measure.py             # Phase 2: 测 B_noise + B_simple
│   ├── outputs_sgd/
│   ├── outputs_adamw/
│   └── outputs_muon/
│
├── cnn/                       # CNN 图片分类实验（结构同 transformer/）
│
└── analysis/                  # 公共数据分析
    ├── fit_bcrit.py
    ├── fit_bnoise.py
    ├── fit_bsimple.py
    └── *.png / *.csv          # 分析结果直接堆这里
```

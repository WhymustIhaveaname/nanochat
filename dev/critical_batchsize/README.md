# Critical Batch Size 实验

测量和对比 $\mathcal{B}_{\text{crit}}$、$\mathcal{B}_{\text{noise}}$、$\mathcal{B}_{\text{simple}}$ 三种方法。

详见 [methods.md](methods.md)

---

## 实验路线

- [ ] **Step 1**: SGD + Transformer
- [ ] **Step 2**: AdamW + Transformer
- [ ] **Step 3**: CNN（ConvNeXt-Tiny + CIFAR-100 / ImageNet-100）
- [ ] **Step 4**: Muon

---

## 文件结构

```
dev/critical_batchsize/
├── methods.md                 # 方法文档
│
├── transformer/               # Transformer 语言模型实验
│   ├── train.py               # Phase 1: 训练 + 记录 loss + 保存 checkpoint
│   ├── measure.py             # Phase 2: 测 B_noise + B_simple
│   └── outputs/               # 输出目录 (日期_depth_optimizer_batchsize)
│
├── cnn/                       # CNN 图片分类实验（结构同 transformer/）
│
└── analysis/                  # 公共数据分析
    ├── fit_bcrit.py
    ├── fit_bnoise.py
    ├── fit_bsimple.py
    └── *.png / *.csv          # 分析结果直接堆这里
```

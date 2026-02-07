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
│   ├── measure.log            # measure.py 的参考输出（用于回归对比）
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

---

## measure.py 参考输出

运行配置：`run_dir=02-02_d4_adamw_8192`, `step=3072`, `optimizer=sgd`, `batch_sizes=[8192,16384,32768,65536]`, `lr=[0,2,4,6,8,10]`, `max_eval_tokens=null`(16M)

```
B=  8192  |g|²=9.959494e-03  ε_opt=2.0485
B= 16384  |g|²=2.829672e-03  ε_opt=4.0057
B= 32768  |g|²=7.497375e-03  ε_opt=2.1593
B= 65536  |g|²=4.936834e-03  ε_opt=2.0444

B_noise  = -99   (R² = 0.0003)
B_simple = 8921  (R² = 0.3232)
```

完整日志见 `transformer/measure.log`。

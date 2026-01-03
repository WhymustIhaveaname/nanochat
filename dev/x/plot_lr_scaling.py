"""
Plot lr/bs and lr/sqrt(bs) vs #params for GPT-3 and LLaMA models.
Data extracted from:
- Table 2.1 (GPT-3): Sizes, architectures, and learning hyper-parameters
- Table 1 (LLaMA): LLaMA 2 family of models
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# Data from Table 2.1 (GPT-3)
# Format: (params_B, batch_size_M, learning_rate)
# =============================================================================
gpt3_data = [
    (0.125, 0.5, 6.0e-4),   # GPT-3 Small (125M)
    (0.350, 0.5, 3.0e-4),   # GPT-3 Medium (350M)
    (0.760, 0.5, 2.5e-4),   # GPT-3 Large (760M)
    (1.3,   1.0, 2.0e-4),   # GPT-3 XL (1.3B)
    (2.7,   1.0, 1.6e-4),   # GPT-3 2.7B
    (6.7,   2.0, 1.2e-4),   # GPT-3 6.7B
    (13.0,  2.0, 1.0e-4),   # GPT-3 13B
    (175.0, 3.2, 0.6e-4),   # GPT-3 175B
]

# =============================================================================
# Data from Table 1 (LLaMA 1 + LLaMA 2)
# Format: (params_B, batch_size_M, learning_rate)
# =============================================================================
llama_data = [
    # LLaMA 2
    (7,  4.0, 3.0e-4),
    (13, 4.0, 3.0e-4),
    (34, 4.0, 1.5e-4),
    (70, 4.0, 1.5e-4),
]

# Convert to numpy arrays
gpt3_params = np.array([d[0] for d in gpt3_data])
gpt3_bs = np.array([d[1] for d in gpt3_data])
gpt3_lr = np.array([d[2] for d in gpt3_data])

llama_params = np.array([d[0] for d in llama_data])
llama_bs = np.array([d[1] for d in llama_data])
llama_lr = np.array([d[2] for d in llama_data])

# Compute lr/bs and lr/sqrt(bs)
gpt3_lr_over_bs = gpt3_lr / gpt3_bs
gpt3_lr_over_sqrt_bs = gpt3_lr / np.sqrt(gpt3_bs)

llama_lr_over_bs = llama_lr / llama_bs
llama_lr_over_sqrt_bs = llama_lr / np.sqrt(llama_bs)

# =============================================================================
# Linear regression in log-log space (using GPT-3 data)
# log(y) = slope * log(x) + intercept  =>  y = 10^intercept * x^slope
# =============================================================================
def fit_power_law(x, y):
    """Fit y = a * x^b in log-log space, return (a, b, r_value)"""
    log_x = np.log10(x)
    log_y = np.log10(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
    a = 10 ** intercept
    b = slope
    return a, b, r_value

# Fit for lr/bs
a1, b1, r1 = fit_power_law(gpt3_params, gpt3_lr_over_bs)
print(f"lr/bs = {a1:.2e} * P^{b1:.3f}, R = {r1:.4f}")

# Fit for lr/sqrt(bs)
a2, b2, r2 = fit_power_law(gpt3_params, gpt3_lr_over_sqrt_bs)
print(f"lr/sqrt(bs) = {a2:.2e} * P^{b2:.3f}, R = {r2:.4f}")

# Generate fit lines
x_fit = np.logspace(np.log10(0.1), np.log10(200), 100)
y_fit1 = a1 * x_fit ** b1
y_fit2 = a2 * x_fit ** b2

# =============================================================================
# Plotting
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left plot: lr/bs vs #params
ax1 = axes[0]
ax1.plot(x_fit, y_fit1, 'tab:blue', linestyle='--', linewidth=2,
         label=f'fit: {a1:.2e}·P^{b1:.2f}, R={r1:.3f}')
ax1.scatter(gpt3_params, gpt3_lr_over_bs, c='tab:blue', marker='o', s=60, label='GPT-3')
ax1.scatter(llama_params, llama_lr_over_bs, c='tab:orange', marker='x', s=60, label='LLaMA')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('#params (B)', fontsize=12)
ax1.set_ylabel(r'$\mathrm{lr} / \mathrm{bs}$', fontsize=12)
ax1.set_title(r'$\mathrm{lr}/\mathrm{bs}$ vs #params', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right plot: lr/sqrt(bs) vs #params
ax2 = axes[1]
ax2.plot(x_fit, y_fit2, 'tab:blue', linestyle='--', linewidth=2,
         label=f'fit: {a2:.2e}·P^{b2:.2f}, R={r2:.3f}')
ax2.scatter(gpt3_params, gpt3_lr_over_sqrt_bs, c='tab:blue', marker='o', s=60, label='GPT-3')
ax2.scatter(llama_params, llama_lr_over_sqrt_bs, c='tab:orange', marker='x', s=60, label='LLaMA')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('#params (B)', fontsize=12)
ax2.set_ylabel(r'$\mathrm{lr} / \sqrt{\mathrm{bs}}$', fontsize=12)
ax2.set_title(r'$\mathrm{lr}/\sqrt{\mathrm{bs}}$ vs #params', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('result_lr_scaling.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved to result_lr_scaling.png")

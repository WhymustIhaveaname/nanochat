"""
Phase 2: Measure B_noise and B_simple from Phase 1 checkpoints.

Usage:
    python -m dev.critical_batchsize.transformer.measure \
        run_dir=dev/critical_batchsize/transformer/outputs/02-02_d4_adamw_8192
"""

import csv
import os
from contextlib import nullcontext

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from nanochat.common import autodetect_device_type, print0
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer


def build_model(run_cfg_dict, vocab_size, device):
    run_cfg = OmegaConf.create(run_cfg_dict) if isinstance(run_cfg_dict, dict) else run_cfg_dict
    model_config = GPTConfig(
        sequence_len=run_cfg.seq_len,
        vocab_size=vocab_size,
        n_layer=run_cfg.depth,
        n_head=run_cfg.depth,
        n_kv_head=run_cfg.depth,
        n_embd=run_cfg.depth * 64,
    )
    model = GPT(model_config).to(device)
    model.init_weights()
    model.value_embeds = nn.ModuleDict({})
    model.resid_lambdas.requires_grad_(False)
    model.x0_lambdas.requires_grad_(False)
    model.resid_lambdas.fill_(1.0)
    model.x0_lambdas.fill_(0.0)

    for n, p in model.named_parameters():
        p.requires_grad_("wte" not in n and "lm_head" not in n)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return model, trainable_params


def _build_optimizer(optimizer_type, params, lr):
    if optimizer_type == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0)
    if optimizer_type == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.9), fused=False)
    raise ValueError(f"Unknown optimizer: {optimizer_type}")


def _init_optimizer_state(optimizer, optimizer_type):
    if optimizer_type == "sgd":
        for group in optimizer.param_groups:
            assert group["momentum"] == 0, "SGD must have momentum=0"
        return

    if optimizer_type == "adamw":
        # AdamW 默认 state (exp_avg=0, exp_avg_sq=0, step=0) 第一步等价于稳态，无需初始化
        # for group in optimizer.param_groups:
        #     for p in group["params"]:
        #         if p.grad is None:
        #             continue
        #         state = optimizer.state[p]
        #         g = p.grad.to(dtype=p.dtype)
        #         state["exp_avg"] = g.clone()
        #         state["exp_avg_sq"] = g**2
        #         state["step"] = torch.tensor(10000)
        p = next(p for group in optimizer.param_groups for p in group["params"] if p.grad is not None)
        assert optimizer.state[p] == {}, f"Expected empty state, got {optimizer.state[p]}"
        return

    raise ValueError(f"Unsupported optimizer: {optimizer_type}")


def _check_optimizer_state(optimizer, optimizer_type, n=4):
    if optimizer_type == "adamw":
        group = optimizer.param_groups[-1]
        beta1, beta2 = group["betas"]
        p = next(p for p in reversed(group["params"]) if p.grad is not None)
        s = optimizer.state[p]
        g = p.grad.flatten()[:n]
        # print(
        #     f"  [check] grad={g.tolist()}"
        #     f"  grad²={(g**2).tolist()}"
        #     f"  exp_avg={s['exp_avg'].flatten()[:n].tolist()}"
        #     f"  exp_avg_sq={s['exp_avg_sq'].flatten()[:n].tolist()}"
        #     f"  step={s['step']}"
        # )
        assert s["step"] == 1
        assert torch.allclose(s["exp_avg"].flatten()[:n], g * (1 - beta1), atol=1e-7)
        assert torch.allclose(s["exp_avg_sq"].flatten()[:n], g**2 * (1 - beta2), atol=1e-7)
    elif optimizer_type == "sgd":
        pass
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")


def _eval_worker_loop(gpu_id, run_cfg_dict, vocab_size, eval_batches_cpu, optimizer_type, task_queue, result_queue):
    """Persistent worker: build model once on assigned GPU, process tasks from queue."""
    device = torch.device(f"cuda:{gpu_id}")
    model, trainable_params = build_model(run_cfg_dict, vocab_size, device)
    eval_batches = [(x.to(device), y.to(device), n) for x, y, n in eval_batches_cpu]
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    while True:
        task = task_queue.get()
        if task is None:
            break
        task_id, ckpt_state_cpu, grads_cpu, lr = task

        ckpt_state = {k: v.to(device) for k, v in ckpt_state_cpu.items()}
        model.load_state_dict(ckpt_state)
        model.train()

        for n, p in model.named_parameters():
            if n in grads_cpu:
                p.grad = grads_cpu[n].to(device=device, dtype=p.dtype)

        optimizer = _build_optimizer(optimizer_type, trainable_params, lr)
        _init_optimizer_state(optimizer, optimizer_type)
        optimizer.step()
        _check_optimizer_state(optimizer, optimizer_type)

        model.eval()
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad(), autocast_ctx:
            for x, y, num_tokens in eval_batches:
                total_loss += model(x, y).item() * num_tokens
                total_tokens += num_tokens
        result_queue.put((task_id, total_loss / total_tokens))


def build_data(tokenizer, run_cfg, device):
    heldout_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, 1, run_cfg.seq_len, split="heldout", device=device
    )
    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, 1, run_cfg.eval_seq_len, split="val", device=device
    )
    return heldout_loader, val_loader


def compute_gradient(model, loader, batch_size, seq_len, autocast_ctx):
    model.zero_grad()
    for _ in range(batch_size // seq_len):
        x, y = next(loader)
        with autocast_ctx:
            loss = model(x, y)
        (loss * seq_len / batch_size).backward()

    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def cache_eval_batches(loader, max_tokens):
    """Cache eval batches on CPU for consistent evaluation across all LR sweeps."""
    batches = []
    total_tokens = 0
    for x, y in loader:
        num_tokens = (y >= 0).sum().item()
        batches.append((x.cpu(), y.cpu(), num_tokens))
        total_tokens += num_tokens
        if total_tokens >= max_tokens:
            break
    return batches


def fit_quadratic(lrs, losses):
    A = np.vstack([np.ones_like(lrs), lrs, lrs**2]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, np.array(losses), rcond=None)
    c, a, b = coeffs
    y_pred = c + a * lrs + b * lrs**2
    ss_res = np.sum((np.array(losses) - y_pred) ** 2)
    ss_tot = np.sum((np.array(losses) - np.mean(losses)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return -a / (2 * b), r2, (c, a, b)


def fit_linear(x, y):
    A = np.vstack([np.ones_like(x), x]).T
    (a, b), _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = a + b * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return b / a, 1 - ss_res / ss_tot


@hydra.main(config_path=".", config_name="measure_config", version_base=None)
def main(cfg: DictConfig):
    device_type = autodetect_device_type()
    device = torch.device(device_type)
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    )

    run_cfg = OmegaConf.load(os.path.join(cfg.run_dir, "config.yaml"))
    print0(f"Run dir: {cfg.run_dir}, depth: {run_cfg.depth}, seq_len: {run_cfg.seq_len}")

    step = cfg.step
    baseline_loss = None
    with open(os.path.join(cfg.run_dir, "loss_eval.csv")) as f:
        for row in csv.DictReader(f):
            if int(row["step"]) == step:
                baseline_loss = float(row["loss"])
                break
    assert baseline_loss is not None, f"Step {step} not found in loss_eval.csv"

    ckpt_path = os.path.join(cfg.run_dir, "checkpoints", f"step_{step:05d}.pt")
    print0(f"Step {step} | Baseline loss: {baseline_loss:.4f} | Checkpoint: {ckpt_path}")

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    model, trainable_params = build_model(run_cfg, vocab_size, device)
    print0(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

    heldout_loader, val_loader = build_data(tokenizer, run_cfg, device)

    max_eval_tokens = cfg.max_eval_tokens or run_cfg.max_eval_tokens
    eval_batches = cache_eval_batches(val_loader, max_eval_tokens)
    print0(f"Cached {len(eval_batches)} eval batches ({max_eval_tokens} tokens)")

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    run_cfg_dict = OmegaConf.to_container(run_cfg)
    num_workers = torch.cuda.device_count()
    workers = []
    for gpu_id in range(num_workers):
        p = ctx.Process(
            target=_eval_worker_loop,
            args=(gpu_id, run_cfg_dict, vocab_size, eval_batches, cfg.optimizer, task_queue, result_queue),
        )
        p.start()
        workers.append(p)
    print0(f"Created {num_workers} eval workers")

    lrs = np.linspace(cfg.lr_min, cfg.lr_max, cfg.lr_steps)
    batch_sizes = list(cfg.batch_sizes)
    print0(f"Batch sizes: {batch_sizes}")
    print0(f"LRs: [{', '.join(f'{lr:.4f}' for lr in lrs)}]")

    ckpt_state = torch.load(ckpt_path, map_location=device, weights_only=True)
    ckpt_state_cpu = {k: v.cpu() for k, v in ckpt_state.items()}

    eps_opts = []
    all_coeffs = []
    grad_norm_sqs = []
    all_losses = np.zeros((len(batch_sizes), len(lrs)))

    for b_idx, B in enumerate(batch_sizes):
        print0(f"\n  [B={B}] Computing gradient...")
        model.load_state_dict(ckpt_state)
        model.train()
        grads = compute_gradient(model, heldout_loader, B, run_cfg.seq_len, autocast_ctx)
        grad_norm_sq = sum((p.grad**2).sum().item() for p in model.parameters() if p.grad is not None)
        grad_norm_sqs.append(grad_norm_sq)
        print0(f"  [B={B}] |g|² = {grad_norm_sq:.6e}")

        grads_cpu = {k: v.cpu() for k, v in grads.items()}
        for lr_idx, lr in enumerate(lrs):
            task_queue.put((lr_idx, ckpt_state_cpu, grads_cpu, float(lr)))

        results = {}
        for _ in range(len(lrs)):
            task_id, loss = result_queue.get()
            results[task_id] = loss
        losses_for_B = [results[i] for i in range(len(lrs))]
        all_losses[b_idx] = losses_for_B

        print0(f"    {'lr':>8}  {'loss':>8}  {'ΔL':>10}")
        for lr_idx, lr in enumerate(lrs):
            print0(f"    {lr:>8.4f}  {losses_for_B[lr_idx]:>8.4f}  {losses_for_B[lr_idx] - baseline_loss:>+10.4f}")

        eps_opt, r2, coeffs = fit_quadratic(lrs, losses_for_B)
        eps_opts.append(eps_opt)
        all_coeffs.append(coeffs)
        c, a, b = coeffs
        print0(f"  [B={B}] Fit: L(ε) = {c:.4f} + {a:.4f}*ε + {b:.4f}*ε²")
        print0(f"  [B={B}] ε_opt = {eps_opt:.4f}, R² = {r2:.4f}")

    plot_lr_fits(batch_sizes, lrs, all_losses, all_coeffs, eps_opts, step, cfg.optimizer, cfg.run_dir)

    inv_B = 1.0 / np.array(batch_sizes)
    inv_eps = 1.0 / np.array(eps_opts)
    B_noise, B_noise_r2 = fit_linear(inv_B, inv_eps)
    B_simple, B_simple_r2 = fit_linear(inv_B, np.array(grad_norm_sqs))

    print0("\n  === Final Fits ===")
    print0("  B_noise fitting (1/ε_opt vs 1/B):")
    for B, eps_opt in zip(batch_sizes, eps_opts):
        print0(f"    B={B:>6}  1/B={1 / B:.2e}  ε_opt={eps_opt:.4f}  1/ε_opt={1 / eps_opt:.2f}")
    print0(f"  => B_noise = {B_noise:.0f} (R² = {B_noise_r2:.4f})")

    print0("\n  B_simple fitting (|g|² vs 1/B):")
    for B, g_sq in zip(batch_sizes, grad_norm_sqs):
        print0(f"    B={B:>6}  1/B={1 / B:.2e}  |g|²={g_sq:.6e}")
    print0(f"  => B_simple = {B_simple:.0f} (R² = {B_simple_r2:.4f})")

    plot_contour(batch_sizes, lrs, all_losses, step, cfg.optimizer, cfg.run_dir)
    plot_fits(batch_sizes, eps_opts, grad_norm_sqs, B_noise, B_simple, step, cfg.optimizer, cfg.run_dir)

    for _ in workers:
        task_queue.put(None)
    for p in workers:
        p.join()
    print0("\nDone.")


########## Plotting functions ##########


def plot_lr_fits(batch_sizes, lrs, all_losses, all_coeffs, eps_opts, step, optimizer, out_dir):
    n_batch = len(batch_sizes)
    cols = min(3, n_batch)
    rows = (n_batch + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    lr_fine = np.linspace(lrs.min(), lrs.max(), 100)
    for i, B in enumerate(batch_sizes):
        ax = axes[i // cols, i % cols]
        c, a, b = all_coeffs[i]
        y_fit = c + a * lr_fine + b * lr_fine**2

        ax.scatter(lrs, all_losses[i], color="tab:blue", s=40, zorder=3)
        ax.plot(lr_fine, y_fit, "r-", linewidth=2)
        ax.axvline(eps_opts[i], color="green", linestyle="--", label=f"ε_opt={eps_opts[i]:.4f}")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Loss")
        ax.set_title(f"B={B}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for i in range(n_batch, rows * cols):
        axes[i // cols, i % cols].axis("off")

    fig.suptitle(f"LR Sweep Fits (Step {step}, {optimizer})", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"step_{step:05d}_{optimizer}_lr_fits.png"), dpi=300)
    plt.close(fig)


def plot_contour(batch_sizes, lrs, losses, step, optimizer, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    B_grid, lr_grid = np.meshgrid(batch_sizes, lrs)
    contour = ax.contourf(B_grid, lr_grid, losses.T, levels=20, cmap="viridis")
    fig.colorbar(contour, ax=ax, label="Loss")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"Loss Contour (Step {step}, {optimizer})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"step_{step:05d}_{optimizer}_contour.png"), dpi=300)
    plt.close(fig)


def plot_fits(batch_sizes, eps_opts, grad_norm_sqs, B_noise, B_simple, step, optimizer, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    B_arr = np.array(batch_sizes)
    inv_B = 1.0 / B_arr
    inv_B_line = np.linspace(inv_B.min() * 0.8, inv_B.max() * 1.2, 100)

    # ε_opt vs B (raw data, no fit)
    axes[0].plot(B_arr, eps_opts, "o-", color="tab:blue", markersize=6)
    axes[0].set_xlabel("B")
    axes[0].set_ylabel("$\\epsilon_{opt}$")
    axes[0].set_title("$\\epsilon_{opt}$ vs B")
    axes[0].grid(True, alpha=0.3)

    # B_noise fit: 1/ε = a + b/B → (1/a)/ε = 1 + (b/a)/B
    inv_eps = 1.0 / np.array(eps_opts)
    axes[1].scatter(inv_B, inv_eps, color="tab:blue", s=50, zorder=3)
    a, b = np.linalg.lstsq(np.vstack([np.ones_like(inv_B), inv_B]).T, inv_eps, rcond=None)[0]
    axes[1].plot(inv_B_line, a + b * inv_B_line, "r-", linewidth=2)
    axes[1].set_xlabel("1/B")
    axes[1].set_ylabel("1/ε_opt")
    axes[1].text(
        0.05,
        0.95,
        f"${1 / a:.2g}/\\epsilon = 1 + {b / a:.2g}/B$",
        transform=axes[1].transAxes,
        fontsize=10,
        verticalalignment="top",
    )
    axes[1].set_title(f"$B_{{noise}}$ = {B_noise:.0f}")
    axes[1].grid(True, alpha=0.3)

    # B_simple fit: |g|² = a2 + b2/B → |g|²/a2 = 1 + (b2/a2)/B
    g_sq = np.array(grad_norm_sqs)
    axes[2].scatter(inv_B, g_sq, color="tab:green", s=50, zorder=3)
    a2, b2 = np.linalg.lstsq(np.vstack([np.ones_like(inv_B), inv_B]).T, g_sq, rcond=None)[0]
    axes[2].plot(inv_B_line, a2 + b2 * inv_B_line, "r-", linewidth=2)
    axes[2].set_xlabel("1/B")
    axes[2].set_ylabel("$|g|^2$")
    axes[2].text(
        0.05,
        0.95,
        f"$|g|^2/{a2:.2g} = 1 + {b2 / a2:.2g}/B$",
        transform=axes[2].transAxes,
        fontsize=10,
        verticalalignment="top",
    )
    axes[2].set_title(f"$B_{{simple}}$ = {B_simple:.0f}")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Step {step} ({optimizer})", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"step_{step:05d}_{optimizer}_fit.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

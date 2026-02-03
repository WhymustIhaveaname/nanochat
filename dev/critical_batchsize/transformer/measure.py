"""
Phase 2: Measure B_noise and B_simple from Phase 1 checkpoints.

Usage:
    python -m dev.critical_batchsize.transformer.measure \
        run_dir=dev/critical_batchsize/transformer/outputs/02-02_d4_adamw_8192
"""

import csv
import glob
import os
from contextlib import nullcontext
from dataclasses import dataclass

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from nanochat.common import autodetect_device_type, print0
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer


# =============================================================================
# Data classes and builders
# =============================================================================


@dataclass
class MeasureContext:
    model: any
    trainable_params: list
    heldout_loader: any
    eval_batches: list  # cached eval batches for consistent evaluation
    autocast_ctx: any
    run_cfg: any
    cfg: any
    lrs: np.ndarray
    batch_sizes: list


def build_model(run_cfg, device):
    tokenizer = get_tokenizer()
    model_config = GPTConfig(
        sequence_len=run_cfg.seq_len,
        vocab_size=tokenizer.get_vocab_size(),
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
    return model, trainable_params, tokenizer


def build_data(tokenizer, run_cfg, device):
    heldout_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, 1, run_cfg.seq_len, split="heldout", device=device
    )
    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, 1, run_cfg.eval_seq_len, split="val", device=device
    )
    return heldout_loader, val_loader


def build_optimizer(optimizer_type, params, lr):
    if optimizer_type == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0)
    if optimizer_type == "adamw":
        # fused=False because we pre-initialize state with potentially different dtype
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.9), fused=False)
    raise ValueError(f"Unknown optimizer: {optimizer_type}")


# =============================================================================
# Core computation
# =============================================================================


def compute_gradient(model, loader, batch_size, seq_len, autocast_ctx):
    model.zero_grad()
    for _ in range(batch_size // seq_len):
        x, y = next(loader)
        with autocast_ctx:
            loss = model(x, y)
        (loss * seq_len / batch_size).backward()

    grad_norm_sq = sum((p.grad**2).sum().item() for p in model.parameters() if p.grad is not None)
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return grads, grad_norm_sq


def restore_gradients(model, grads):
    for n, p in model.named_parameters():
        if n in grads:
            p.grad = grads[n].clone()


def init_optimizer_state(optimizer, optimizer_type):
    if optimizer_type == "sgd":
        for group in optimizer.param_groups:
            assert group["momentum"] == 0, "SGD must have momentum=0"
        return

    if optimizer_type == "adamw":
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = optimizer.state[p]
                g = p.grad.to(dtype=p.dtype)  # match param dtype (bfloat16)
                state["exp_avg"] = g.clone()
                state["exp_avg_sq"] = g**2
                state["step"] = torch.tensor(0)


def verify_optimizer_state_unchanged(optimizer, trainable_params, state_before):
    """Verify that exp_avg and exp_avg_sq didn't change after step (for AdamW with betas=(0.9, 0.9))."""
    for p in trainable_params:
        if p in optimizer.state:
            for k in ["exp_avg", "exp_avg_sq"]:
                if k in state_before[id(p)]:
                    before = state_before[id(p)][k]
                    after = optimizer.state[p][k]
                    if not torch.allclose(before, after, rtol=1e-5, atol=1e-8):
                        diff = (after - before).abs()
                        print0(f"ERROR: {k} changed significantly! max_diff={diff.max().item():.6e}")
                        print0(f"  before: norm={before.norm().item():.6e}, after: norm={after.norm().item():.6e}")


def cache_eval_batches(loader, max_tokens):
    """Cache eval batches for consistent evaluation across all LR sweeps."""
    batches = []
    total_tokens = 0
    for x, y in loader:
        batches.append((x.clone(), y.clone()))
        total_tokens += (y >= 0).sum().item()
        if total_tokens >= max_tokens:
            break
    return batches


def do_eval(model, eval_batches, autocast_ctx):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad(), autocast_ctx:
        for x, y in eval_batches:
            num_tokens = (y >= 0).sum().item()
            total_loss += model(x, y).item() * num_tokens
            total_tokens += num_tokens
    model.train()
    return total_loss / total_tokens


def find_optimal_lr(lrs, losses):
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


def sweep_lrs(ctx, ckpt_state, grads, step, B, grad_norm_sq, baseline_loss, raw_writer):
    losses = []
    print0(f"    {'lr':>8}  {'loss':>8}  {'ΔL':>10}")
    for lr in ctx.lrs:
        ctx.model.load_state_dict(ckpt_state)
        ctx.model.train()
        restore_gradients(ctx.model, grads)

        optimizer = build_optimizer(ctx.cfg.optimizer, ctx.trainable_params, lr)
        init_optimizer_state(optimizer, ctx.cfg.optimizer)
        state_before = {
            id(p): {k: v.clone() if torch.is_tensor(v) else v for k, v in optimizer.state[p].items()}
            for p in ctx.trainable_params
            if p in optimizer.state
        }
        optimizer.step()
        verify_optimizer_state_unchanged(optimizer, ctx.trainable_params, state_before)

        eval_loss = do_eval(ctx.model, ctx.eval_batches, ctx.autocast_ctx)
        delta_loss = eval_loss - baseline_loss  # negative = loss decreased (good)
        losses.append(eval_loss)
        raw_writer.writerow([step, B, lr, eval_loss, grad_norm_sq])
        print0(f"    {lr:>8.4f}  {eval_loss:>8.4f}  {delta_loss:>+10.4f}")
    return losses


def process_checkpoint(ctx, ckpt_state, step, baseline_loss, raw_writer, out_dir):
    eps_opts, grad_norm_sqs = [], []
    all_losses = np.zeros((len(ctx.batch_sizes), len(ctx.lrs)))
    all_coeffs = []

    for b_idx, B in enumerate(ctx.batch_sizes):
        print0(f"\n  [B={B}] Computing gradient...")
        ctx.model.load_state_dict(ckpt_state)
        ctx.model.train()
        grads, grad_norm_sq = compute_gradient(ctx.model, ctx.heldout_loader, B, ctx.run_cfg.seq_len, ctx.autocast_ctx)
        grad_norm_sqs.append(grad_norm_sq)
        print0(f"  [B={B}] |g|² = {grad_norm_sq:.6e}")

        losses_for_B = sweep_lrs(ctx, ckpt_state, grads, step, B, grad_norm_sq, baseline_loss, raw_writer)
        all_losses[b_idx] = losses_for_B

        eps_opt, r2, coeffs = find_optimal_lr(ctx.lrs, losses_for_B)
        eps_opts.append(eps_opt)
        all_coeffs.append(coeffs)
        c, a, b = coeffs
        print0(f"  [B={B}] Fit: L(ε) = {c:.4f} + {a:.4f}*ε + {b:.4f}*ε²")
        print0(f"  [B={B}] ε_opt = {eps_opt:.4f}, R² = {r2:.4f}")

    plot_lr_fits(ctx.batch_sizes, ctx.lrs, all_losses, all_coeffs, eps_opts, step, out_dir)
    return eps_opts, grad_norm_sqs, all_losses


def measure(cfg: DictConfig):
    device_type = autodetect_device_type()
    device = torch.device(device_type)
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    )

    run_cfg = OmegaConf.load(os.path.join(cfg.run_dir, "config.yaml"))
    print0(f"Run dir: {cfg.run_dir}, depth: {run_cfg.depth}, seq_len: {run_cfg.seq_len}")

    step_to_loss = {}
    with open(os.path.join(cfg.run_dir, "loss_eval.csv")) as f:
        for row in csv.DictReader(f):
            step_to_loss[int(row["step"])] = float(row["loss"])

    ckpt_paths = sorted(glob.glob(os.path.join(cfg.run_dir, "checkpoints", "step_*.pt")))
    if cfg.single_step is not None:
        ckpt_paths = [p for p in ckpt_paths if f"step_{cfg.single_step:05d}.pt" in p]
    print0(f"Found {len(ckpt_paths)} checkpoints")

    model, trainable_params, tokenizer = build_model(run_cfg, device)
    print0(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

    heldout_loader, val_loader = build_data(tokenizer, run_cfg, device)

    # Cache eval batches for consistent evaluation across all LR sweeps
    max_eval_tokens = cfg.max_eval_tokens or run_cfg.max_eval_tokens
    eval_batches = cache_eval_batches(val_loader, max_eval_tokens)
    print0(f"Cached {len(eval_batches)} eval batches ({max_eval_tokens} tokens)")

    lrs = np.linspace(cfg.lr_min, cfg.lr_max, cfg.lr_steps)
    batch_sizes = list(cfg.batch_sizes)
    lr_str = ", ".join(f"{lr:.4f}" for lr in lrs)
    print0(f"Batch sizes: {batch_sizes}")
    print0(f"LRs: [{lr_str}]")

    out_dir = os.path.join(cfg.run_dir, "measure")
    os.makedirs(out_dir, exist_ok=True)

    raw_file = open(os.path.join(out_dir, "raw_data.csv"), "w", newline="")
    raw_writer = csv.writer(raw_file)
    raw_writer.writerow(["step", "batch_size", "lr", "loss", "grad_norm_sq"])

    results_file = open(os.path.join(out_dir, "results.csv"), "w", newline="")
    results_writer = csv.writer(results_file)
    results_writer.writerow(["step", "eval_loss", "B_noise", "B_noise_r2", "B_simple", "B_simple_r2"])

    ctx = MeasureContext(
        model=model,
        trainable_params=trainable_params,
        heldout_loader=heldout_loader,
        eval_batches=eval_batches,
        autocast_ctx=autocast_ctx,
        run_cfg=run_cfg,
        cfg=cfg,
        lrs=lrs,
        batch_sizes=batch_sizes,
    )

    for ckpt_idx, ckpt_path in enumerate(ckpt_paths):
        step = int(os.path.basename(ckpt_path).replace("step_", "").replace(".pt", ""))
        baseline_loss = step_to_loss[step]
        print0(f"\n{'=' * 60}")
        print0(f"[{ckpt_idx + 1}/{len(ckpt_paths)}] Step {step} | Baseline loss: {baseline_loss:.4f}")
        print0(f"{'=' * 60}")

        ckpt_state = torch.load(ckpt_path, map_location=device, weights_only=True)
        eps_opts, grad_norm_sqs, all_losses = process_checkpoint(
            ctx, ckpt_state, step, baseline_loss, raw_writer, out_dir
        )
        raw_file.flush()

        # Final fitting
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

        results_writer.writerow([step, step_to_loss[step], B_noise, B_noise_r2, B_simple, B_simple_r2])
        results_file.flush()

        plot_contour(batch_sizes, lrs, all_losses, step, out_dir)
        plot_fits(batch_sizes, eps_opts, grad_norm_sqs, B_noise, B_simple, step, out_dir)

    raw_file.close()
    results_file.close()
    print0("\nDone.")


def plot_lr_fits(batch_sizes, lrs, all_losses, all_coeffs, eps_opts, step, out_dir):
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

    fig.suptitle(f"LR Sweep Fits (Step {step})", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"step_{step:05d}_lr_fits.png"), dpi=150)
    plt.close(fig)


def plot_contour(batch_sizes, lrs, losses, step, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    B_grid, lr_grid = np.meshgrid(batch_sizes, lrs)
    contour = ax.contourf(B_grid, lr_grid, losses.T, levels=20, cmap="viridis")
    fig.colorbar(contour, ax=ax, label="Loss")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"Loss Contour (Step {step})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"step_{step:05d}_contour.png"), dpi=150)
    plt.close(fig)


def plot_fits(batch_sizes, eps_opts, grad_norm_sqs, B_noise, B_simple, step, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    inv_B = 1.0 / np.array(batch_sizes)
    inv_B_line = np.linspace(inv_B.min() * 0.8, inv_B.max() * 1.2, 100)

    # B_noise fit: 1/ε = a + b/B, B_noise = b/a
    inv_eps = 1.0 / np.array(eps_opts)
    axes[0].scatter(inv_B, inv_eps, color="tab:blue", s=50, zorder=3)
    a, b = np.linalg.lstsq(np.vstack([np.ones_like(inv_B), inv_B]).T, inv_eps, rcond=None)[0]
    axes[0].plot(inv_B_line, a + b * inv_B_line, "r-", linewidth=2)
    axes[0].set_xlabel("1/B")
    axes[0].set_ylabel("1/ε_opt")
    axes[0].text(
        0.05,
        0.95,
        f"$1/\\epsilon = {a:.2f} + {b:.0f}/B$",
        transform=axes[0].transAxes,
        fontsize=10,
        verticalalignment="top",
    )
    axes[0].set_title(f"$B_{{noise}}$ = {B_noise:.0f}")
    axes[0].grid(True, alpha=0.3)

    # B_simple fit: |g|² = a + b/B, B_simple = b/a
    g_sq = np.array(grad_norm_sqs)
    axes[1].scatter(inv_B, g_sq, color="tab:green", s=50, zorder=3)
    a2, b2 = np.linalg.lstsq(np.vstack([np.ones_like(inv_B), inv_B]).T, g_sq, rcond=None)[0]
    axes[1].plot(inv_B_line, a2 + b2 * inv_B_line, "r-", linewidth=2)
    axes[1].set_xlabel("1/B")
    axes[1].set_ylabel("$|g|^2$")
    axes[1].text(
        0.05,
        0.95,
        f"$|g|^2 = {a2:.2e} + {b2:.2e}/B$",
        transform=axes[1].transAxes,
        fontsize=10,
        verticalalignment="top",
    )
    axes[1].set_title(f"$B_{{simple}}$ = {B_simple:.0f}")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Step {step}", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"step_{step:05d}_fit.png"), dpi=150)
    plt.close(fig)


@hydra.main(config_path=".", config_name="measure_config", version_base=None)
def main(cfg: DictConfig):
    measure(cfg)


if __name__ == "__main__":
    main()

"""
Phase 2: Measure B_noise and B_simple from Phase 1 checkpoints.

Usage:
    python -m dev.critical_batchsize.transformer.measure \
        run_dir=dev/critical_batchsize/transformer/outputs/02-02_d4_adamw_8192
"""

import csv
import os
from contextlib import nullcontext
from dataclasses import dataclass

import hydra
import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from nanochat.common import autodetect_device_type, print0
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer


# =============================================================================
# Ray Actor for parallel evaluation
# =============================================================================


def _build_model_on_device(run_cfg_dict, device):
    """Build model on specified device (shared logic for main process and workers)."""
    tokenizer = get_tokenizer()
    run_cfg = OmegaConf.create(run_cfg_dict) if isinstance(run_cfg_dict, dict) else run_cfg_dict
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
    return model, trainable_params


def _build_optimizer(optimizer_type, params, lr):
    """Build optimizer (shared logic)."""
    if optimizer_type == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0)
    if optimizer_type == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.9), fused=False)
    raise ValueError(f"Unknown optimizer: {optimizer_type}")


def _init_optimizer_state(optimizer, optimizer_type):
    """Initialize optimizer state (shared logic)."""
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
                g = p.grad.to(dtype=p.dtype)
                state["exp_avg"] = g.clone()
                state["exp_avg_sq"] = g**2
                state["step"] = torch.tensor(0)


@ray.remote(num_gpus=1)
class EvalWorker:
    """Ray actor for parallel evaluation. Each worker holds a model copy on its GPU."""

    def __init__(self, run_cfg_dict, eval_batches_cpu, optimizer_type):
        self.device = torch.device("cuda")
        self.model, self.trainable_params = _build_model_on_device(run_cfg_dict, self.device)
        self.optimizer_type = optimizer_type

        # Move eval batches to GPU (eval_batches_cpu is already deserialized by Ray)
        self.eval_batches = [(x.to(self.device), y.to(self.device)) for x, y in eval_batches_cpu]
        self.autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    def eval_after_step(self, ckpt_state_cpu, grads_cpu, lr):
        """Load checkpoint, apply gradients, take optimizer step, and evaluate.

        Args:
            ckpt_state_cpu: checkpoint state dict (CPU tensors, auto-resolved by Ray)
            grads_cpu: gradients dict (CPU tensors, auto-resolved by Ray)
            lr: learning rate for this evaluation

        Returns:
            eval_loss: float
        """

        # Load checkpoint to GPU
        ckpt_state = {k: v.to(self.device) for k, v in ckpt_state_cpu.items()}
        self.model.load_state_dict(ckpt_state)
        self.model.train()

        # Restore gradients
        for n, p in self.model.named_parameters():
            if n in grads_cpu:
                p.grad = grads_cpu[n].to(device=self.device, dtype=p.dtype)

        # Create optimizer, init state, and step
        optimizer = _build_optimizer(self.optimizer_type, self.trainable_params, lr)
        _init_optimizer_state(optimizer, self.optimizer_type)
        optimizer.step()

        # Eval
        self.model.eval()
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad(), self.autocast_ctx:
            for x, y in self.eval_batches:
                num_tokens = (y >= 0).sum().item()
                total_loss += self.model(x, y).item() * num_tokens
                total_tokens += num_tokens
        return total_loss / total_tokens


# =============================================================================
# Data classes and builders
# =============================================================================


@dataclass
class MeasureContext:
    model: any
    trainable_params: list
    heldout_loader: any
    autocast_ctx: any
    run_cfg: any
    run_cfg_dict: dict  # for passing to workers
    cfg: any
    lrs: np.ndarray
    batch_sizes: list
    workers: list  # Ray actors


def build_model(run_cfg, device):
    model, trainable_params = _build_model_on_device(run_cfg, device)
    tokenizer = get_tokenizer()
    return model, trainable_params, tokenizer


def build_data(tokenizer, run_cfg, device):
    heldout_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, 1, run_cfg.seq_len, split="heldout", device=device
    )
    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, 1, run_cfg.eval_seq_len, split="val", device=device
    )
    return heldout_loader, val_loader


# Keep old names as aliases for compatibility
build_optimizer = _build_optimizer
init_optimizer_state = _init_optimizer_state


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


def process_checkpoint_parallel(ctx, ckpt_state, step, baseline_loss, out_dir):
    """Process checkpoint with parallel evaluation using Ray workers."""
    batch_sizes = ctx.batch_sizes
    lrs = ctx.lrs

    # Put checkpoint state in Ray object store (once, shared by all workers)
    ckpt_state_cpu = {k: v.cpu() for k, v in ckpt_state.items()}
    ckpt_state_ref = ray.put(ckpt_state_cpu)

    eps_opts = []
    all_coeffs = []
    all_grad_norm_sqs = []
    all_losses = np.zeros((len(batch_sizes), len(lrs)))

    # Process each batch_size: compute gradient, parallel eval all lrs, then print
    for b_idx, B in enumerate(batch_sizes):
        # Step 1: Compute gradient for this B
        print0(f"\n  [B={B}] Computing gradient...")
        ctx.model.load_state_dict(ckpt_state)
        ctx.model.train()
        grads, grad_norm_sq = compute_gradient(
            ctx.model, ctx.heldout_loader, B, ctx.run_cfg.seq_len, ctx.autocast_ctx
        )
        all_grad_norm_sqs.append(grad_norm_sq)
        print0(f"  [B={B}] |g|² = {grad_norm_sq:.6e}")

        # Step 2: Submit all lr tasks for this B in parallel
        grads_cpu = {k: v.cpu() for k, v in grads.items()}
        grads_ref = ray.put(grads_cpu)

        tasks = []
        for lr_idx, lr in enumerate(lrs):
            worker_idx = lr_idx % len(ctx.workers)
            task = ctx.workers[worker_idx].eval_after_step.remote(
                ckpt_state_ref, grads_ref, float(lr)
            )
            tasks.append(task)

        # Step 3: Wait for all lr evals to complete
        losses_for_B = ray.get(tasks)
        all_losses[b_idx] = losses_for_B

        # Step 4: Print results for this B
        print0(f"    {'lr':>8}  {'loss':>8}  {'ΔL':>10}")
        for lr_idx, lr in enumerate(lrs):
            loss = losses_for_B[lr_idx]
            delta_loss = loss - baseline_loss
            print0(f"    {lr:>8.4f}  {loss:>8.4f}  {delta_loss:>+10.4f}")

        # Step 5: Fit and print for this B
        eps_opt, r2, coeffs = find_optimal_lr(lrs, losses_for_B)
        eps_opts.append(eps_opt)
        all_coeffs.append(coeffs)
        c, a, b = coeffs
        print0(f"  [B={B}] Fit: L(ε) = {c:.4f} + {a:.4f}*ε + {b:.4f}*ε²")
        print0(f"  [B={B}] ε_opt = {eps_opt:.4f}, R² = {r2:.4f}")

    plot_lr_fits(batch_sizes, lrs, all_losses, all_coeffs, eps_opts, step, ctx.cfg.optimizer, out_dir)
    return eps_opts, all_grad_norm_sqs, all_losses


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
    fig.savefig(os.path.join(out_dir, f"step_{step:05d}_{optimizer}_lr_fits.png"), dpi=150)
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
    fig.savefig(os.path.join(out_dir, f"step_{step:05d}_{optimizer}_contour.png"), dpi=150)
    plt.close(fig)


def plot_fits(batch_sizes, eps_opts, grad_norm_sqs, B_noise, B_simple, step, optimizer, out_dir):
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

    fig.suptitle(f"Step {step} ({optimizer})", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"step_{step:05d}_{optimizer}_fit.png"), dpi=150)
    plt.close(fig)


@hydra.main(config_path=".", config_name="measure_config", version_base=None)
def main(cfg: DictConfig):
    device_type = autodetect_device_type()
    device = torch.device(device_type)
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    )

    run_cfg = OmegaConf.load(os.path.join(cfg.run_dir, "config.yaml"))
    print0(f"Run dir: {cfg.run_dir}, depth: {run_cfg.depth}, seq_len: {run_cfg.seq_len}")

    # Load baseline loss for the target step
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

    model, trainable_params, tokenizer = build_model(run_cfg, device)
    print0(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

    heldout_loader, val_loader = build_data(tokenizer, run_cfg, device)

    max_eval_tokens = cfg.max_eval_tokens or run_cfg.max_eval_tokens
    eval_batches = cache_eval_batches(val_loader, max_eval_tokens)
    print0(f"Cached {len(eval_batches)} eval batches ({max_eval_tokens} tokens)")

    # Initialize Ray and create workers
    ray.init(ignore_reinit_error=True)
    eval_batches_cpu = [(x.cpu(), y.cpu()) for x, y in eval_batches]
    run_cfg_dict = OmegaConf.to_container(run_cfg)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_workers = getattr(cfg, "num_workers", num_gpus)
    workers = [
        EvalWorker.remote(run_cfg_dict, eval_batches_cpu, cfg.optimizer)
        for _ in range(num_workers)
    ]
    print0(f"Created {num_workers} Ray workers for parallel eval ({num_gpus} GPUs detected)")

    lrs = np.linspace(cfg.lr_min, cfg.lr_max, cfg.lr_steps)
    batch_sizes = list(cfg.batch_sizes)
    lr_str = ", ".join(f"{lr:.4f}" for lr in lrs)
    print0(f"Batch sizes: {batch_sizes}")
    print0(f"LRs: [{lr_str}]")

    out_dir = cfg.run_dir

    ctx = MeasureContext(
        model=model,
        trainable_params=trainable_params,
        heldout_loader=heldout_loader,
        autocast_ctx=autocast_ctx,
        run_cfg=run_cfg,
        run_cfg_dict=run_cfg_dict,
        cfg=cfg,
        lrs=lrs,
        batch_sizes=batch_sizes,
        workers=workers,
    )

    ckpt_state = torch.load(ckpt_path, map_location=device, weights_only=True)
    eps_opts, grad_norm_sqs, all_losses = process_checkpoint_parallel(
        ctx, ckpt_state, step, baseline_loss, out_dir
    )

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

    plot_contour(batch_sizes, lrs, all_losses, step, cfg.optimizer, out_dir)
    plot_fits(batch_sizes, eps_opts, grad_norm_sqs, B_noise, B_simple, step, cfg.optimizer, out_dir)

    ray.shutdown()
    print0("\nDone.")


if __name__ == "__main__":
    main()

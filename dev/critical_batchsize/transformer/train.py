"""
Phase 1: Train transformer with different batch sizes for B_crit measurement.

Usage:
    python -m dev.critical_batchsize.transformer.train batch_size=1000000
"""

import os
import csv
import time
from contextlib import nullcontext

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type
from nanochat.tokenizer import get_tokenizer


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    train(cfg)


def train(cfg: DictConfig):
    # Setup
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    tokenizer = get_tokenizer()

    # Model
    model_dim = cfg.depth * 64
    num_heads = cfg.depth
    config = GPTConfig(
        sequence_len=cfg.seq_len,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=cfg.depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
    )
    model = GPT(config).to(device)
    model.init_weights()
    model.value_embeds = nn.ModuleDict({})  # 禁用 value_embeds
    model.resid_lambdas.requires_grad_(False)  # 禁用 scalar 学习
    model.x0_lambdas.requires_grad_(False)
    model.resid_lambdas.fill_(1.0)  # 固定为 1（恒等残差）
    model.x0_lambdas.fill_(0.0)     # 固定为 0（无 x0 混合）
    if ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
    raw_model = model.module if ddp else model

    param_counts = raw_model.num_scaling_params()
    print0(f"Param counts: {param_counts}")
    total_tokens = int(cfg.train_ratio * param_counts["transformer_matrices"])
    num_iterations = total_tokens // cfg.batch_size
    eval_every = max(1, num_iterations // cfg.eval_times)
    print0(f"Total tokens: {total_tokens:,} | Steps: {num_iterations} | Eval every: {eval_every}")

    # Gradient accumulation (micro batch = 1 sample with seqpack)
    tokens_per_micro = cfg.seq_len * ddp_world_size
    assert cfg.batch_size % tokens_per_micro == 0, f"batch_size {cfg.batch_size} must be divisible by {tokens_per_micro}"
    grad_accum_steps = cfg.batch_size // tokens_per_micro
    print0(f"Batch size {cfg.batch_size} => grad_accum: {grad_accum_steps}")

    # Optimizer - 分组设置不同 LR
    embd_params, lm_head_params, matrix_params = [], [], []
    for n, p in raw_model.named_parameters():
        if "wte" in n:
            embd_params.append(p)
        elif "lm_head" in n:
            lm_head_params.append(p)
        elif "lambda" in n:
            pass  # scalars 已禁用，不训练
        else:
            matrix_params.append(p)
    param_groups = [
        {"name": "embd", "params": embd_params, "lr": cfg.lr_embd},
        {"name": "lm_head", "params": lm_head_params, "lr": cfg.lr_lm_head},
        {"name": "matrix", "params": matrix_params, "lr": cfg.lr_matrix},
    ]
    for group in param_groups:
        print0(f"Group {group['name']}: {sum(p.numel() for p in group['params']):,} params")

    if cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(param_groups)
    elif cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.9))
    elif cfg.optimizer == "muon":
        optimizer = raw_model.setup_optimizer(
            matrix_lr=cfg.lr_matrix,
            embedding_lr=cfg.lr_embd,
            unembedding_lr=cfg.lr_lm_head,
            scalar_lr=0.0,
            weight_decay=0.0,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    # Dataloaders (batch_size=1 with seqpack)
    train_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, 1, cfg.seq_len, split="train", device=device
    )
    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, 1, cfg.eval_seq_len, split="val", device=device
    )

    # Output setup
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    print0(f"Output dir: {output_dir}")
    if master_process:
        os.makedirs(ckpt_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"))
        train_csv = open(os.path.join(output_dir, "loss_train.csv"), "w", newline="")
        eval_csv = open(os.path.join(output_dir, "loss_eval.csv"), "w", newline="")
        train_writer = csv.writer(train_csv)
        eval_writer = csv.writer(eval_csv)
        train_writer.writerow(["step", "loss"])
        eval_writer.writerow(["step", "loss"])

    def do_eval(step):
        t0 = time.time()
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad(), autocast_ctx:
            for x, y in val_loader:
                num_tokens = (y >= 0).sum().item()  # 排除 ignore_index
                loss = raw_model(x, y)
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                if total_tokens >= cfg.max_eval_tokens:
                    break
        val_loss = total_loss / total_tokens
        elapsed = time.time() - t0
        print0(f"Step {step:05d} | eval loss: {val_loss:.4f} | {elapsed:.1f}s")
        if master_process:
            eval_writer.writerow([step, val_loss])
            eval_csv.flush()
            ckpt_path = os.path.join(ckpt_dir, f"step_{step:05d}.pt")
            torch.save(raw_model.state_dict(), ckpt_path)
            print0(f"Saved: {ckpt_path}")
        model.train()

    # Training loop
    model.train()
    for step in range(num_iterations + 1):
        # Eval at regular intervals and always at last step (skip step 0)
        if (step > 0 and step % eval_every == 0) or step == num_iterations:
            do_eval(step)

        if step == num_iterations:
            break

        # Training step
        optimizer.zero_grad()
        total_loss = 0.0
        for _ in range(grad_accum_steps):
            x, y = next(train_loader)
            with autocast_ctx:
                loss = model(x, y)
            (loss / grad_accum_steps).backward()
            total_loss += loss.item()
        optimizer.step()
        avg_loss = total_loss / grad_accum_steps

        if master_process:
            train_writer.writerow([step, avg_loss])
            if step % 10 == 0 or step < 10 or step == num_iterations:
                train_csv.flush()
                print0(f"Step {step:05d} | train loss: {avg_loss:.4f}")

    # Cleanup
    if master_process:
        train_csv.close()
        eval_csv.close()
    compute_cleanup()
    print0("Done.")


if __name__ == "__main__":
    main()

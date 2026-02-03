import glob
import os
from dataclasses import dataclass

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import gaussian_filter1d


def smooth_loss(loss: np.ndarray, sigma_frac: float = 0.01) -> np.ndarray:
    """Gaussian smoothing with sigma = sigma_frac * len(loss)."""
    sigma = max(1, int(len(loss) * sigma_frac))
    return gaussian_filter1d(loss, sigma=sigma, mode="nearest")


@dataclass
class RunData:
    run_dir: str
    batch_size: int
    train_step: np.ndarray
    train_loss: np.ndarray
    train_loss_smooth: np.ndarray
    eval_step: np.ndarray
    eval_loss: np.ndarray


def infer_tag(pattern: str) -> str:
    # dev/critical_batchsize/transformer/outputs/02-01_d4_adamw_* -> transformer_02-01_d4_adamw
    parts = pattern.split("/outputs/")
    prefix = os.path.basename(parts[0].rstrip("/"))
    suffix = parts[1].rstrip("_*")
    return f"{prefix}_{suffix}"


def parse_batch_size(run_dir: str) -> int:
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    return cfg.batch_size


def read_loss_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def interpolate_step_for_loss(step: np.ndarray, loss: np.ndarray, target: float) -> float:
    loss_mono = np.minimum.accumulate(loss)
    return float(np.interp(target, loss_mono[::-1], step[::-1]))


def fit_smin_emin(E: np.ndarray, S: np.ndarray) -> tuple[float, float, float]:
    """Linear regression on 1/S vs 1/E.

    From: 1 = Emin/E + Smin/S
    We get: 1/S = 1/Smin - (Emin/Smin) * (1/E)

    So fitting 1/S = a + b * (1/E):
      - a = 1/Smin -> Smin = 1/a
      - b = -Emin/Smin -> Emin = -b * Smin = -b/a
      - Bcrit = Emin/Smin = -b
    """
    inv_E = 1.0 / E
    inv_S = 1.0 / S
    # Linear regression: inv_S = a + b * inv_E
    A = np.vstack([np.ones_like(inv_E), inv_E]).T
    (a, b), residuals, _, _ = np.linalg.lstsq(A, inv_S, rcond=None)
    smin = 1.0 / a
    emin = -b / a
    # R^2
    inv_S_pred = a + b * inv_E
    ss_res = np.sum((inv_S - inv_S_pred) ** 2)
    ss_tot = np.sum((inv_S - np.mean(inv_S)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(smin), float(emin), r2


def run(cfg: DictConfig) -> None:
    run_dirs = sorted({os.path.abspath(p) for p in glob.glob(cfg.runs_glob)})
    out_dir = cfg.out_dir or "dev/critical_batchsize/analysis"
    os.makedirs(out_dir, exist_ok=True)
    tag = cfg.tag or infer_tag(cfg.runs_glob)

    runs: list[RunData] = []
    for run_dir in run_dirs:
        train_step, train_loss = read_loss_csv(os.path.join(run_dir, "loss_train.csv"))
        eval_step, eval_loss = read_loss_csv(os.path.join(run_dir, "loss_eval.csv"))
        runs.append(
            RunData(
                run_dir=run_dir,
                batch_size=parse_batch_size(run_dir),
                train_step=train_step,
                train_loss=train_loss,
                train_loss_smooth=smooth_loss(train_loss),
                eval_step=eval_step,
                eval_loss=eval_loss,
            )
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for color, run in zip(plt.cm.viridis(np.linspace(0.1, 0.9, len(runs))), runs):
        # Train: raw with high transparency, smoothed with solid line
        axes[0].plot(run.train_step, run.train_loss, color=color, alpha=0.15, linewidth=0.8)
        axes[0].plot(run.train_step, run.train_loss_smooth, color=color, alpha=0.9, linewidth=1.5)
        # Eval
        axes[1].plot(run.eval_step, run.eval_loss, color=color, alpha=0.8, linewidth=1.5)
    axes[0].set_title("Train loss vs step (smoothed)")
    axes[1].set_title("Eval loss vs step")
    for ax in axes:
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tag}_losscurve.png"), dpi=160)
    plt.close(fig)

    if cfg.loss_source == "train":
        loss_lower = max(float(np.min(run.train_loss_smooth)) for run in runs)
    else:
        loss_lower = max(float(np.min(run.eval_loss)) for run in runs)
    loss_targets = np.linspace(loss_lower, cfg.loss_upper, cfg.loss_targets)
    points_rows = []
    result_rows = []

    for target in loss_targets:
        steps = []
        batches = []
        for run in runs:
            if cfg.loss_source == "train":
                step = interpolate_step_for_loss(run.train_step, run.train_loss_smooth, target)
            else:
                step = interpolate_step_for_loss(run.eval_step, run.eval_loss, target)
            steps.append(step)
            batches.append(run.batch_size)
            points_rows.append(
                {
                    "loss_target": target,
                    "batch_size": run.batch_size,
                    "step": step,
                    "E": run.batch_size * step,
                }
            )
        E = np.asarray(steps, dtype=float) * np.asarray(batches, dtype=float)
        smin, emin, r2 = fit_smin_emin(E, np.asarray(steps, dtype=float))
        result_rows.append(
            {
                "loss_target": target,
                "bcrit": emin / smin,
                "smin": smin,
                "emin": emin,
                "r2": r2,
                "num_runs": len(steps),
            }
        )

    if result_rows:
        # Pick 3 loss levels: low, mid, high
        n = len(result_rows)
        indices = [0, n // 2, n - 1]  # low, mid, high
        colors = ["tab:blue", "tab:orange", "tab:green"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Collect all inv_E for axis range
        all_inv_E = []
        for idx, color in zip(indices, colors):
            target = result_rows[idx]["loss_target"]
            E_arr = np.asarray([row["E"] for row in points_rows if row["loss_target"] == target], dtype=float)
            S_arr = np.asarray([row["step"] for row in points_rows if row["loss_target"] == target], dtype=float)
            smin, emin, r2 = fit_smin_emin(E_arr, S_arr)

            inv_E = 1.0 / E_arr
            inv_S = 1.0 / S_arr
            all_inv_E.extend(inv_E)

            # Scatter points
            axes[0].scatter(inv_E, inv_S, color=color, alpha=0.8, marker="o")

            # Fit line
            inv_E_line = np.linspace(inv_E.min() * 0.8, inv_E.max() * 1.2, 100)
            inv_S_line = 1.0 / smin - (emin / smin) * inv_E_line
            axes[0].plot(inv_E_line, inv_S_line, color=color, label=f"L={target:.2f} ($R^2$={r2:.4f})")

        axes[0].set_title("1/S vs 1/E @ different loss levels")
        axes[0].set_xlabel("1/E")
        axes[0].set_ylabel("1/S")
        axes[0].grid(True, alpha=0.2)
        axes[0].legend(fontsize=8)

        loss_arr = np.array([row["loss_target"] for row in result_rows])
        bcrit_arr = np.array([row["bcrit"] for row in result_rows])
        axes[1].plot(loss_arr, bcrit_arr, marker="o", color="tab:green", label="Measured")

        # Kaplan scaling law: B_crit = B* / L^(1/alpha_B)
        B_star = 2e8  # tokens
        alpha_B = 0.21
        loss_line = np.linspace(loss_arr.min(), loss_arr.max(), 100)
        bcrit_kaplan = B_star / (loss_line ** (1 / alpha_B))
        axes[1].plot(loss_line, bcrit_kaplan, "--", color="tab:red", label="Kaplan")

        axes[1].set_title("B_crit vs target loss")
        axes[1].set_xlabel("Target loss")
        axes[1].set_ylabel("B_crit")
        axes[1].grid(True, alpha=0.2)
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{tag}_bcrit.png"), dpi=160)
        plt.close(fig)

    print(f"[ok] {tag}_losscurve.png")
    print(f"[ok] {tag}_bcrit.png")


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()

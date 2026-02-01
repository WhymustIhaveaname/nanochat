import glob
import os
from dataclasses import dataclass

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import curve_fit


@dataclass
class RunData:
    run_dir: str
    batch_size: int
    train_step: np.ndarray
    train_loss: np.ndarray
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
    (smin, emin), _ = curve_fit(
        lambda E, smin, emin: smin * E / (E - emin),
        E, S,
        p0=[S.min(), E.min() * 0.5],
        bounds=([0, 0], [np.inf, E.min()]),
    )
    residual = S - smin * E / (E - emin)
    ss_res = np.sum(residual**2)
    ss_tot = np.sum((S - np.mean(S)) ** 2)
    return float(smin), float(emin), 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


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
                eval_step=eval_step,
                eval_loss=eval_loss,
            )
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for color, run in zip(plt.cm.viridis(np.linspace(0.1, 0.9, len(runs))), runs):
        axes[0].plot(run.train_step, run.train_loss, color=color, alpha=0.6, linewidth=1.2)
        axes[1].plot(run.eval_step, run.eval_loss, color=color, alpha=0.8, linewidth=1.5)
    axes[0].set_title("Train loss vs step")
    axes[1].set_title("Eval loss vs step")
    for ax in axes:
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tag}_losscurve.png"), dpi=160)
    plt.close(fig)

    if cfg.loss_source == "train":
        loss_lower = max(float(np.min(run.train_loss)) for run in runs)
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
                step = interpolate_step_for_loss(run.train_step, run.train_loss, target)
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
        mid_target = result_rows[len(result_rows) // 2]["loss_target"]
        E_mid = np.asarray([row["E"] for row in points_rows if row["loss_target"] == mid_target], dtype=float)
        S_mid = np.asarray([row["step"] for row in points_rows if row["loss_target"] == mid_target], dtype=float)
        smin, emin, _ = fit_smin_emin(E_mid, S_mid)
        E_line = np.linspace(min(E_mid) * 1.01, max(E_mid) * 0.99, 200)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(E_mid, S_mid, color="tab:blue", alpha=0.8, label="Runs")
        axes[0].plot(E_line, smin * E_line / (E_line - emin), color="tab:orange", label="Fit")
        axes[0].set_title(f"S vs E fit @ loss={mid_target:.3f}")
        axes[0].set_xlabel("E = B * S")
        axes[0].set_ylabel("S")
        axes[0].grid(True, alpha=0.2)
        axes[0].legend()

        axes[1].plot(
            [row["loss_target"] for row in result_rows],
            [row["bcrit"] for row in result_rows],
            marker="o",
            color="tab:green",
        )
        axes[1].set_title("B_crit vs target loss")
        axes[1].set_xlabel("Target loss")
        axes[1].set_ylabel("B_crit")
        axes[1].grid(True, alpha=0.2)

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

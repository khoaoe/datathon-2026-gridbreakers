"""
Training tracker: logs loss/scores per iteration, saves to CSV, plots learning curves.
"""
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from modeling.config import OUTPUT_DIR


TRACKER_DIR = OUTPUT_DIR / "tracking"
TRACKER_DIR.mkdir(parents=True, exist_ok=True)


class ExperimentTracker:
    """
    Tracks training metrics across iterations/epochs.

    Usage:
        tracker = ExperimentTracker("ex_03_lgbm")
        tracker.log_params({"lr": 0.03, "max_depth": 8})
        tracker.log_step(100, {"train_mae": 1234, "val_mae": 5678})
        tracker.log_step(200, {"train_mae": 1100, "val_mae": 5500})
        tracker.log_final({"mae": 5500, "rmse": 7000, "r2": 0.85})
        tracker.save()
    """

    def __init__(self, experiment_name):
        self.name = experiment_name
        self.start_time = time.time()
        self.params = {}
        self.steps = []          # list of {step, metric1, metric2, ...}
        self.final_scores = {}
        self.notes = ""
        self.save_dir = TRACKER_DIR / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def log_params(self, params: dict):
        """Log hyperparameters."""
        self.params.update(params)

    def log_step(self, step: int, metrics: dict):
        """Log metrics at a training step/iteration."""
        entry = {"step": step, "elapsed_sec": round(time.time() - self.start_time, 1)}
        entry.update(metrics)
        self.steps.append(entry)

    def log_final(self, scores: dict):
        """Log final validation scores."""
        self.final_scores = scores

    def add_note(self, note: str):
        self.notes += note + "\n"

    def elapsed(self):
        return round(time.time() - self.start_time, 1)

    def save(self):
        """Save all tracking data to disk."""
        # Save step-level metrics as CSV
        if self.steps:
            steps_df = pd.DataFrame(self.steps)
            steps_df.to_csv(self.save_dir / "training_log.csv", index=False)

        # Save summary as JSON
        summary = {
            "experiment": self.name,
            "timestamp": datetime.now().isoformat(),
            "elapsed_sec": self.elapsed(),
            "params": self.params,
            "final_scores": self.final_scores,
            "total_steps": len(self.steps),
            "notes": self.notes,
        }
        with open(self.save_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Plot learning curves
        self._plot_curves()

        print(f"Tracking saved → {self.save_dir}/")

    def _plot_curves(self):
        """Generate learning curve plots."""
        if not self.steps:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  (matplotlib not installed, skipping plots)")
            return

        df = pd.DataFrame(self.steps)
        metric_cols = [c for c in df.columns if c not in ("step", "elapsed_sec")]

        if not metric_cols:
            return

        fig, axes = plt.subplots(1, len(metric_cols), figsize=(6 * len(metric_cols), 4))
        if len(metric_cols) == 1:
            axes = [axes]

        for ax, col in zip(axes, metric_cols):
            ax.plot(df["step"], df[col], marker=".", markersize=3, linewidth=1)
            ax.set_xlabel("Step")
            ax.set_ylabel(col)
            ax.set_title(f"{self.name} — {col}")
            ax.grid(True, alpha=0.3)

            # Mark best value
            if "loss" in col.lower() or "mae" in col.lower() or "rmse" in col.lower():
                best_idx = df[col].idxmin()
                best_step = df.loc[best_idx, "step"]
                best_val = df.loc[best_idx, col]
                ax.axhline(y=best_val, color="red", linestyle="--", alpha=0.5)
                ax.annotate(f"best={best_val:,.0f} @step {best_step}",
                            xy=(best_step, best_val), fontsize=8, color="red")

        plt.tight_layout()
        plt.savefig(self.save_dir / "learning_curves.png", dpi=150)
        plt.close()


class LGBMCallback:
    """
    Custom LightGBM callback that logs to ExperimentTracker.

    Usage:
        tracker = ExperimentTracker("ex_03_lgbm")
        cb = LGBMCallback(tracker)
        model.fit(..., callbacks=[cb, lgb.early_stopping(100)])
    """

    def __init__(self, tracker: ExperimentTracker, log_every=50):
        self.tracker = tracker
        self.log_every = log_every

    def __call__(self, env):
        if env.iteration % self.log_every != 0 and env.iteration != env.end_iteration - 1:
            return

        metrics = {}
        for data_name, eval_name, result, _ in env.evaluation_result_list:
            metrics[f"{data_name}_{eval_name}"] = result

        self.tracker.log_step(env.iteration, metrics)


class XGBCallback:
    """
    Custom XGBoost callback that logs to ExperimentTracker.

    Usage:
        tracker = ExperimentTracker("ex_04_xgb")
        cb = XGBCallback(tracker)
        model.fit(..., callbacks=[cb])
    """

    def __init__(self, tracker: ExperimentTracker, log_every=50):
        self.tracker = tracker
        self.log_every = log_every

    def __call__(self, env):
        if env.iteration % self.log_every != 0:
            return

        metrics = {}
        for item in env.evaluation_result_list:
            # XGBoost format: (dataset, metric, value)
            metrics[f"{item[0]}_{item[1]}"] = item[2]

        self.tracker.log_step(env.iteration, metrics)


def print_scoreboard():
    """Print a summary scoreboard from all saved experiments."""
    if not TRACKER_DIR.exists():
        print("No experiments tracked yet.")
        return

    rows = []
    for exp_dir in sorted(TRACKER_DIR.iterdir()):
        summary_path = exp_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)
            scores = data.get("final_scores", {})
            rows.append({
                "Experiment": data["experiment"],
                "MAE": scores.get("mae", "—"),
                "RMSE": scores.get("rmse", "—"),
                "R²": scores.get("r2", "—"),
                "Time (s)": data.get("elapsed_sec", "—"),
                "Steps": data.get("total_steps", "—"),
            })

    if rows:
        df = pd.DataFrame(rows)
        # Format numeric columns
        for col in ["MAE", "RMSE"]:
            df[col] = df[col].apply(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
        for col in ["R²"]:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
        print("\n" + "=" * 70)
        print("EXPERIMENT SCOREBOARD")
        print("=" * 70)
        print(df.to_string(index=False))
        print("=" * 70)
    else:
        print("No experiments tracked yet.")


if __name__ == "__main__":
    print_scoreboard()

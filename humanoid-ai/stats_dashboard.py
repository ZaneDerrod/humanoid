# Comparison dashboard — reads TensorBoard event files and produces a 2x2 figure
import argparse
import os
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ALGO_COLORS = {
    "PPO/RPO": "#1f77b4",
    "SAC": "#ff7f0e",
    "TD3": "#2ca02c",
    "TQC": "#d62728",
}


def detect_algo(folder_name):
    if "tqc" in folder_name:
        return "TQC"
    if "sac" in folder_name:
        return "SAC"
    if "td3" in folder_name:
        return "TD3"
    return "PPO/RPO"


def rolling_mean(values, window):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def load_runs(runs_dir):
    runs = []
    for entry in sorted(os.listdir(runs_dir)):
        run_path = os.path.join(runs_dir, entry)
        if not os.path.isdir(run_path):
            continue
        event_files = glob.glob(os.path.join(run_path, "events.out.tfevents.*"))
        if not event_files:
            continue
        # pick the most recent event file
        event_file = max(event_files, key=os.path.getmtime)
        algo = detect_algo(entry)
        ea = EventAccumulator(run_path)
        ea.Reload()
        scalars = {}
        wall_times = {}
        available_tags = ea.Tags().get("scalars", [])
        for tag in available_tags:
            events = ea.Scalars(tag)
            scalars[tag] = {"steps": [e.step for e in events], "values": [e.value for e in events]}
            wall_times[tag] = [e.wall_time for e in events]
        runs.append(dict(
            name=entry,
            algo=algo,
            scalars=scalars,
            wall_times=wall_times,
            path=run_path,
        ))
    return runs


def main():
    parser = argparse.ArgumentParser(description="Generate comparison dashboard from TensorBoard logs")
    parser.add_argument("--runs_dir", type=str, default="runs/")
    parser.add_argument("--smoothing_window", type=int, default=20)
    parser.add_argument("--output_path", type=str, default="runs/comparison_dashboard.png")
    args = parser.parse_args()

    runs = load_runs(args.runs_dir)
    if not runs:
        print("No runs found. Exiting.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Humanoid-v4 Algorithm Comparison", fontsize=16, fontweight="bold")

    # ---- Panel 1: Reward Curves ----
    ax1 = axes[0, 0]
    ax1.set_title("Reward Curves (Episodic Return vs Step)")
    ax1.set_xlabel("Global Step")
    ax1.set_ylabel("Episodic Return")
    for run in runs:
        tag = "charts/episodic_return"
        if tag not in run["scalars"]:
            continue
        steps = np.array(run["scalars"][tag]["steps"])
        values = np.array(run["scalars"][tag]["values"])
        color = ALGO_COLORS.get(run["algo"], "gray")
        ax1.plot(steps, values, alpha=0.15, color=color, linewidth=0.5)
        smoothed = rolling_mean(values, args.smoothing_window)
        offset = len(values) - len(smoothed)
        ax1.plot(steps[offset:], smoothed, color=color, label=f"{run['algo']} ({run['name'][:30]}...)", linewidth=1.5)
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ---- Panel 2: Sample Efficiency ----
    ax2 = axes[0, 1]
    ax2.set_title("Sample Efficiency (Return per Million Steps)")
    ax2.set_xlabel("Million Environment Steps")
    ax2.set_ylabel("Episodic Return")
    for run in runs:
        tag = "charts/episodic_return"
        if tag not in run["scalars"]:
            continue
        steps = np.array(run["scalars"][tag]["steps"]) / 1_000_000
        values = np.array(run["scalars"][tag]["values"])
        color = ALGO_COLORS.get(run["algo"], "gray")
        ax2.plot(steps, values, alpha=0.15, color=color, linewidth=0.5)
        smoothed = rolling_mean(values, args.smoothing_window)
        offset = len(values) - len(smoothed)
        ax2.plot(steps[offset:], smoothed, color=color, label=f"{run['algo']}", linewidth=1.5)
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(True, alpha=0.3)

    # ---- Panel 3: Wall-Clock Time ----
    ax3 = axes[1, 0]
    ax3.set_title("Wall-Clock Time (Return vs Elapsed Minutes)")
    ax3.set_xlabel("Elapsed Minutes")
    ax3.set_ylabel("Episodic Return")
    for run in runs:
        tag = "charts/episodic_return"
        if tag not in run["scalars"] or tag not in run["wall_times"]:
            continue
        wt = np.array(run["wall_times"][tag])
        if len(wt) == 0:
            continue
        elapsed_min = (wt - wt[0]) / 60.0
        values = np.array(run["scalars"][tag]["values"])
        color = ALGO_COLORS.get(run["algo"], "gray")
        ax3.plot(elapsed_min, values, alpha=0.15, color=color, linewidth=0.5)
        smoothed = rolling_mean(values, args.smoothing_window)
        offset = len(values) - len(smoothed)
        ax3.plot(elapsed_min[offset:], smoothed, color=color, label=f"{run['algo']}", linewidth=1.5)
    ax3.legend(fontsize=7, loc="upper left")
    ax3.grid(True, alpha=0.3)

    # ---- Panel 4: Final Performance Bar Chart ----
    ax4 = axes[1, 1]
    ax4.set_title("Final Performance (Last 10% of Episodes)")
    bar_data = []
    for run in runs:
        tag = "charts/episodic_return"
        if tag not in run["scalars"]:
            continue
        values = np.array(run["scalars"][tag]["values"])
        if len(values) == 0:
            continue
        cutoff = max(1, int(len(values) * 0.1))
        final_values = values[-cutoff:]
        bar_data.append(dict(
            algo=run["algo"],
            name=run["name"],
            mean=np.mean(final_values),
            std=np.std(final_values),
        ))
    # sort highest to lowest
    bar_data.sort(key=lambda x: x["mean"], reverse=True)
    if bar_data:
        labels = [f"{d['algo']} (seed {d['name'].split('__')[2] if len(d['name'].split('__')) > 2 else '?'})" for d in bar_data]
        means = [d["mean"] for d in bar_data]
        stds = [d["std"] for d in bar_data]
        colors = [ALGO_COLORS.get(d["algo"], "gray") for d in bar_data]
        y_pos = np.arange(len(bar_data))
        ax4.barh(y_pos, means, xerr=stds, color=colors, alpha=0.8, capsize=3)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(labels, fontsize=8)
        ax4.set_xlabel("Mean Episodic Return")
        ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)
    fig.savefig(args.output_path, dpi=300, bbox_inches="tight")
    print(f"Dashboard saved to {args.output_path}")
    plt.close(fig)

    # ---- Console summary table ----
    print()
    header = f"{'Algorithm':<10} | {'Run Name':<50} | {'Max Return':>11} | {'Final Mean':>11} | {'Final Std':>10} | {'Mean SPS':>9} | {'Total Steps':>12} | {'Train Time (min)':>17}"
    print(header)
    print("-" * len(header))
    for run in runs:
        algo = run["algo"]
        name = run["name"]
        ret_tag = "charts/episodic_return"
        sps_tag = "charts/SPS"
        if ret_tag in run["scalars"]:
            values = np.array(run["scalars"][ret_tag]["values"])
            max_ret = np.max(values) if len(values) > 0 else 0.0
            cutoff = max(1, int(len(values) * 0.1))
            final_mean = np.mean(values[-cutoff:]) if len(values) > 0 else 0.0
            final_std = np.std(values[-cutoff:]) if len(values) > 0 else 0.0
            total_steps = run["scalars"][ret_tag]["steps"][-1] if len(run["scalars"][ret_tag]["steps"]) > 0 else 0
            wt = np.array(run["wall_times"][ret_tag])
            train_min = (wt[-1] - wt[0]) / 60.0 if len(wt) > 1 else 0.0
        else:
            max_ret = final_mean = final_std = total_steps = train_min = 0.0
        if sps_tag in run["scalars"]:
            sps_vals = np.array(run["scalars"][sps_tag]["values"])
            mean_sps = np.mean(sps_vals) if len(sps_vals) > 0 else 0.0
        else:
            mean_sps = 0.0
        print(f"{algo:<10} | {name:<50} | {max_ret:>11.1f} | {final_mean:>11.1f} | {final_std:>10.1f} | {mean_sps:>9.0f} | {total_steps:>12} | {train_min:>17.1f}")


if __name__ == "__main__":
    main()

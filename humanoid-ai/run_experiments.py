# Experiment runner — launches SAC, TD3, TQC sequentially then generates dashboard
import argparse
import subprocess
import sys
import time


ALGORITHMS = {
    "sac": "train_humanoid_sac.py",
    "td3": "train_humanoid_td3.py",
    "tqc": "train_humanoid_tqc.py",
}

DEFAULT_ORDER = ["sac", "td3", "tqc"]


def main():
    parser = argparse.ArgumentParser(description="Run all humanoid training experiments sequentially")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument("--algorithms", type=str, default=None, help="Comma-separated subset, e.g. 'sac,tqc'")
    parser.add_argument("--seed", type=int, default=1, help="Seed passed to all child scripts")
    parser.add_argument("--total_timesteps", type=int, default=3_000_000, help="Total timesteps per algorithm")
    args = parser.parse_args()

    if args.algorithms:
        algo_list = [a.strip().lower() for a in args.algorithms.split(",")]
        for a in algo_list:
            if a not in ALGORITHMS:
                print(f"Unknown algorithm: {a}. Choose from: {list(ALGORITHMS.keys())}")
                sys.exit(1)
    else:
        algo_list = DEFAULT_ORDER

    cumulative_time = 0.0
    times = {}

    for algo in algo_list:
        script = ALGORITHMS[algo]
        cmd = [
            sys.executable, script,
            "--seed", str(args.seed),
            "--total-timesteps", str(args.total_timesteps),
        ]
        print(f"\n{'='*60}")
        print(f"  Running {algo.upper()}: {' '.join(cmd)}")
        print(f"{'='*60}\n")

        if args.dry_run:
            times[algo] = 0.0
            continue

        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0
        times[algo] = elapsed
        cumulative_time += elapsed
        print(f"\n{algo.upper()} finished in {elapsed/60:.1f} min (exit code {result.returncode})")

    # generate dashboard
    dashboard_cmd = [sys.executable, "stats_dashboard.py"]
    print(f"\n{'='*60}")
    print(f"  Generating dashboard: {' '.join(dashboard_cmd)}")
    print(f"{'='*60}\n")

    if not args.dry_run:
        subprocess.run(dashboard_cmd)

    # summary
    print(f"\n{'='*60}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for algo in algo_list:
        print(f"  {algo.upper():<6}: {times[algo]/60:>8.1f} min")
    print(f"  {'TOTAL':<6}: {cumulative_time/60:>8.1f} min")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

'''
# Run all three new algorithms sequentially + generate dashboard
$env:MUJOCO_GL = "glfw"; uv run python run_experiments.py

# Or run individually
uv run python train_humanoid_sac.py
uv run python train_humanoid_td3.py
uv run python train_humanoid_tqc.py

# Generate dashboard from existing runs
uv run python stats_dashboard.py

# Resume RPO from checkpoint
uv run python train_humanoid.py --resume-path "runs/Humanoid-v4__train_humanoid__1__1775661229/agent_final.pt"
'''
"""Run the closed-loop DBS benchmark with all built-in controllers.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --duration 30 --seed 0
    python scripts/run_benchmark.py --controller bang-bang --plot

Results are saved to results/<timestamp>/
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from dbs_bench.controllers.bang_bang import BangBangController
from dbs_bench.controllers.pi_controller import PIController
from dbs_bench.simulation.simulate import SimulationRunner
from dbs_bench.synthetic.data_generator import generate_demo_patient

CONTROLLERS = {
    "bang-bang": lambda beta_0: BangBangController(beta_0=beta_0),
    "pi": lambda beta_0: PIController(beta_0=beta_0),
}

BETA_0 = 2.3  # Log-space threshold


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the benchmark script."""
    p = argparse.ArgumentParser(description="Run closed-loop DBS benchmark")
    p.add_argument("--controller", choices=list(CONTROLLERS) + ["all"], default="all")
    p.add_argument("--duration", type=float, default=60.0, help="Simulation duration (s)")
    p.add_argument("--dt", type=float, default=0.02, help="Sample period (s)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--n-state", type=int, default=15, help="History buffer length")
    p.add_argument("--plot", action="store_true", help="Plot results (requires matplotlib)")
    p.add_argument("--out", type=str, default=None, help="Output directory")
    return p.parse_args()


def run_one(
    name: str,
    patient,
    duration: float,
    dt: float,
    beta_0: float,
    out_dir: Path,
) -> dict:
    """Run one built-in controller and persist its arrays and metrics."""
    ctrl = CONTROLLERS[name](beta_0)
    runner = SimulationRunner(patient, dt=dt, beta_0=beta_0)

    t0 = time.perf_counter()
    result = runner.run(ctrl, duration=duration, controller_type=name, show_progress=True)
    elapsed = time.perf_counter() - t0

    metrics = result.metrics.copy()
    metrics["wall_clock_s"] = elapsed
    metrics["controller"] = name

    # Save arrays
    np.savez(
        out_dir / f"{name}_arrays.npz",
        time=result.time,
        y=result.y,
        u=result.u,
    )

    # Save metrics
    with open(out_dir / f"{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{name}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    return metrics


def plot_results(results: list, out_dir: Path) -> None:
    """Save a comparison plot for the completed controller runs."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    try:
        from dbs_bench.analysis.figure_style import apply_style
        apply_style()
    except Exception:
        pass

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for r in results:
        data = np.load(out_dir / f"{r['controller']}_arrays.npz")
        ax1.plot(data["time"], data["y"], label=r["controller"], alpha=0.8)
        ax2.plot(data["time"], data["u"], label=r["controller"], alpha=0.8)

    ax1.axhline(BETA_0, color="k", linestyle="--", label="threshold")
    ax1.set_ylabel("Beta power (log-space)")
    ax1.legend()
    ax1.set_title("Closed-loop DBS Benchmark")

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Stimulation amplitude")
    ax2.legend()

    plt.tight_layout()
    fig_path = out_dir / "benchmark.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Figure saved to {fig_path}")
    plt.close(fig)


def main() -> None:
    """Run the requested benchmark controllers on one synthetic patient."""
    args = parse_args()

    out_dir = Path(args.out) if args.out else Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    patient = generate_demo_patient(n_state_y=args.n_state, seed=args.seed)

    names = list(CONTROLLERS) if args.controller == "all" else [args.controller]
    all_metrics = []

    for name in names:
        metrics = run_one(name, patient, args.duration, args.dt, BETA_0, out_dir)
        all_metrics.append(metrics)

    # Summary table
    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nResults saved to {out_dir}/")

    if args.plot:
        plot_results(all_metrics, out_dir)


if __name__ == "__main__":
    main()

"""Quick demo: run bang-bang and PI controllers on synthetic beta data.

Should complete in under 30 seconds.

Run:
    python examples/quick_demo.py
"""
from __future__ import annotations

import numpy as np

from dbs_bench.controllers.bang_bang import BangBangController
from dbs_bench.controllers.pi_controller import PIController
from dbs_bench.simulation.simulate import SimulationRunner, compute_metrics
from dbs_bench.synthetic.data_generator import generate_demo_patient

BETA_0 = 2.3  # Log-space threshold
DURATION = 10.0  # seconds
DT = 0.02  # 50 Hz


def main() -> None:
    patient = generate_demo_patient(n_state_y=15, seed=42)
    runner = SimulationRunner(patient, dt=DT, beta_0=BETA_0)

    print("Running bang-bang controller ...")
    bb_result = runner.run(
        BangBangController(beta_0=BETA_0),
        duration=DURATION,
        controller_type="bang-bang",
        show_progress=False,
    )

    print("Running PI controller ...")
    pi_result = runner.run(
        PIController(beta_0=BETA_0),
        duration=DURATION,
        controller_type="pi",
        show_progress=False,
    )

    print("\n--- Results ---")
    for label, res in [("bang-bang", bb_result), ("PI", pi_result)]:
        m = res.metrics
        print(
            f"{label:12s}  tracking_error={m['tracking_error']:.4f}"
            f"  control_effort={m['control_effort']:.4f}"
            f"  time_above={m['time_above_threshold']:.2%}"
        )

    print("\nDemo complete.")


if __name__ == "__main__":
    main()

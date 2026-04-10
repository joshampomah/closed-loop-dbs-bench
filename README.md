# closed-loop-dbs-bench

Benchmark hub for closed-loop deep brain stimulation (DBS) control.

This repository provides:
- **Baseline controllers**: bang-bang on/off and PI, ready to run on synthetic data
- **Simulation harness**: `SimulationRunner` with `ControllerProtocol` for plugging in method-repo controllers
- **Shared utilities**: ARX predictor, ARI model, cost functions, evaluation metrics, figure style
- **Synthetic data**: AR-process beta trajectories with PRBS stimulation overlay — no patient data required

Method-specific implementations live in separate repositories:
- [dcnn-tube-mpc-dbs](../dcnn-tube-mpc-dbs) — DC neural network tube MPC
- [koopman-mpc-dbs](../koopman-mpc-dbs) — Koopman operator MPC
- [embedded-stable-neuron-mpc](../embedded-stable-neuron-mpc) — embedded stable-neuron QP solver

> **Disclaimer**: This is a research prototype and is **not a medical device**. It has not been approved, cleared, or certified by any regulatory authority and must not be used for clinical decision-making or patient treatment. See [DISCLAIMER.md](DISCLAIMER.md).

---

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python 3.10–3.12.

## Quick start

```python
from dbs_bench.controllers.bang_bang import BangBangController
from dbs_bench.simulation.simulate import SimulationRunner
from dbs_bench.synthetic.data_generator import generate_demo_patient

patient = generate_demo_patient(n_state_y=15)
runner  = SimulationRunner(patient, dt=0.02, beta_0=2.3)
result  = runner.run(BangBangController(beta_0=2.3), duration=10.0)
print(result.metrics)
```

Or run the benchmark script:

```bash
python scripts/run_benchmark.py --duration 60 --plot
```

## Plugging in a custom controller

Implement the `ControllerProtocol`:

```python
from dbs_bench.simulation.simulate import ControllerProtocol

class MyController:
    def compute_control(self, y_history, u_history, u_prev):
        # y_history: (n,) float32, most-recent first
        # u_history: (m,) float32, most-recent first
        u = ...
        info = {}   # optional solver diagnostics
        return float(u), info

    def reset(self):
        ...
```

Pass it to `runner.run(ctrl, controller_type="custom")`.

## Tests

```bash
pytest tests/ -v
```

## Background

The simulation model follows the CDC24/CDC25 framework for closed-loop DBS:

```
y(k) = y_β(k) · exp(−η(u(k)))
```

In log-domain: `log y(k) = log y_β(k) − η(k)`, where `y_β` follows an AR(3)
process and `η` is the response of a 2nd-order ZOH state-space system to
stimulation `u`.  Parameters are set to the published nominal values
(`gain=62.11`, `τ₁=0.05 s`, `τ₂=0.25 s`, `dt=0.02 s`).

## Citation

If you use this software, please cite:

```bibtex
@software{ampomah2025dbsbench,
  author = {Ampomah, Joshua},
  title  = {closed-loop-dbs-bench},
  year   = {2025},
}
```

See [CITATION.cff](CITATION.cff) for the full citation metadata.

## License

MIT — see [LICENSE](LICENSE).

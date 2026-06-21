# closed-loop-dbs-bench

Benchmark and baseline code for closed-loop deep brain stimulation (DBS)
control experiments.

This is the best starting point for understanding the public code release. It
defines the synthetic plant, the simulation loop, the controller interface, the
baseline controllers, and the common metrics used by the method repositories.

This code is a research prototype. It is not a medical device and must not be
used for clinical decision-making or patient treatment. See
[DISCLAIMER.md](DISCLAIMER.md).

## Repository Set

The project is split by responsibility:

| Repository | Purpose |
|---|---|
| [closed-loop-dbs-bench](https://github.com/joshampomah/closed-loop-dbs-bench) | Shared benchmark, synthetic DBS plant, metrics, plotting utilities, bang-bang/PI/linear baselines |
| [dcnn-tube-mpc-dbs](https://github.com/joshampomah/dcnn-tube-mpc-dbs) | DC neural network tube MPC method: predictor, SCP controller, uncertainty bounds, synthetic training/demo code |
| [koopman-mpc-dbs](https://github.com/joshampomah/koopman-mpc-dbs) | Koopman MPC method: lifted-linear predictor, dense QP builder, OLS training/demo code |
| [embedded-stable-neuron-mpc](https://github.com/joshampomah/embedded-stable-neuron-mpc) | C++/STM32 implementation of the stable-neuron and Koopman QP solvers, plus the final report PDF |

Use this repo to compare controllers under one benchmark. Use the method repos
to inspect, train, or modify the individual controllers.

## What Is In This Repo

- `src/dbs_bench/simulation/`: `SimulationRunner`, synthetic/real-data plant
  wrappers, result containers, logging helpers.
- `src/dbs_bench/synthetic/`: public-safe synthetic beta and stimulation
  generators; no patient recordings are included.
- `src/dbs_bench/controllers/`: baseline controllers and linear MPC support.
  The main ready-to-run baselines are bang-bang and PI.
- `src/dbs_bench/models/`: ARI/ARX/state-space utilities used by the benchmark.
- `src/dbs_bench/evaluation/`: cost and metric helpers.
- `src/dbs_bench/analysis/`: plotting style helpers for producing comparable
  figures.
- `scripts/run_benchmark.py`: baseline benchmark runner.
- `examples/quick_demo.py`: small programmatic demo.
- `tests/`: pytest coverage for the public-safe benchmark components.

## What Is Not In This Repo

- No patient recordings.
- No patient-derived trained model weights.
- No STM32 firmware or embedded C++ solver code.
- No DCNN or Koopman implementation internals beyond the common controller
  interface.

## Installation

Requires Python 3.10-3.12.

```bash
pip install -e ".[dev]"
```

## Quick Start

Run the built-in baseline benchmark:

```bash
python scripts/run_benchmark.py --duration 60 --plot
```

Or use the simulation runner directly:

```python
from dbs_bench.controllers.bang_bang import BangBangController
from dbs_bench.simulation.simulate import SimulationRunner
from dbs_bench.synthetic.data_generator import generate_demo_patient

patient = generate_demo_patient(n_state_y=15)
runner = SimulationRunner(patient, dt=0.02, beta_0=2.3)
result = runner.run(BangBangController(beta_0=2.3), duration=10.0)
print(result.metrics)
```

## Loading Method Controllers

`scripts/run_benchmark.py` intentionally runs only the built-in baselines. To
benchmark a controller from a method repo, install that sibling repo and pass a
controller object into `SimulationRunner`.

The runner supports two controller styles:

```python
class ScalarController:
    def compute_control(self, y):
        return 0.0

    def reset(self):
        pass
```

and the history-aware style used by MPC controllers:

```python
class HistoryController:
    def compute_control(self, y_history, u_history, u_prev):
        # y_history and u_history are newest-first float arrays.
        u = 0.0
        info = {"status": "ok"}
        return u, info

    def reset(self):
        pass
```

Example integration:

```python
from dbs_bench.simulation.simulate import SimulationRunner
from dbs_bench.synthetic.data_generator import generate_demo_patient

patient = generate_demo_patient(n_state_y=15)
runner = SimulationRunner(patient, dt=0.02, beta_0=2.3)

# ctrl can be a DCNN SCPController or a KoopmanControllerAdapter.
result = runner.run(ctrl, duration=60.0, controller_type="custom")
print(result.metrics)
```

## Using Your Own Data

The public release contains only synthetic/demo-safe data. For private
recordings, see [DATA.md](DATA.md). In short:

- raw 4YP-style `.mat` recordings should be processed into patient folders
  containing `beta_causal_RMS.csv` and `stimulation.csv`;
- the DCNN and Koopman training scripts can read those processed folders
  directly via `--data-dir`;
- benchmark replay uses a log-space beta trace passed to
  `SimulationRunner(..., real_beta_data=beta)`;
- runtime controller histories are newest-first, while training windows store
  each history oldest-to-newest.

## Tests

```bash
pytest tests/ -v
```

The GitHub Actions workflow runs the tests on Python 3.10, 3.11, and 3.12.

## Background

The synthetic model follows the closed-loop DBS setup used in the project:

```text
y(k) = y_beta(k) * exp(-eta(u(k)))
```

In log space this becomes:

```text
log y(k) = log y_beta(k) - eta(k)
```

`y_beta` is generated by an AR process and `eta` is produced by a second-order
zero-order-hold stimulation response model. The nominal public-safe parameters
are set in `src/dbs_bench/config/device_params.json`.

## Citation

```bibtex
@software{ampomah2025dbsbench,
  author = {Ampomah, Joshua},
  title  = {closed-loop-dbs-bench},
  year   = {2025},
}
```

See [CITATION.cff](CITATION.cff) for the full citation metadata.

## License

MIT. See [LICENSE](LICENSE).

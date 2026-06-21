# Using Your Own Data

The public repositories do not include patient recordings. Keep `.mat`, `.csv`,
model checkpoints, and result exports outside the public repos, for example
under a sibling `private_data/` directory.

## Raw `.mat` Recordings For DCNN/Koopman Training

The larger DCNN/Koopman training data came from the Cambium/MRC BNDU dataset
[STN local field potential recordings from awake patients with Parkinson's, ON
and OFF meds, and during 130 Hz DBS](https://data.mrc.ox.ac.uk/stn-lfp-on-off-and-dbs).
The dataset DOI is `10.5287/bodleian:mzJ7YwXvo`; registered/logged-in users can
download or request access to the raw data from that page.

The 4YP processing code expected each medication-state `.mat` file to contain
a `SmrData` struct:

- `SmrData.Fs`: original sample rate in Hz;
- `SmrData.WvData`: LFP data as `[n_channels x n_samples]`;
- `SmrData.WvTits`: channel names used to select the STN channel.

The same dataset also provides `MATRIX_DBS.mat` for the 130 Hz DBS recordings,
with `MATRIX_DBS.fs`, `MATRIX_DBS.signal_base`, and
`MATRIX_DBS.signal_dbs`. The public training examples below use the
medication/resting-state `.mat` files processed into patient folders.

The raw `.mat` files are not read directly by the public training scripts.
Process them once into the patient-folder format below, then pass the processed
root to the method repo.

The 4YP MATLAB processor did this for each recording:

1. Select the relevant STN LFP channel.
2. Band-pass filter 13-30 Hz with a 3rd-order Butterworth filter.
3. Use causal filtering, matching the real-time controller assumption.
4. Compute a 0.5 s RMS beta envelope.
5. Resample the beta envelope to 50 Hz.
6. Write one processed patient directory.

Expected processed output:

```text
private_data/processed/aperiodic/
├── patient_001/
│   ├── beta_causal_RMS.csv    # Linear beta RMS envelope at 50 Hz
│   ├── stimulation.csv        # Same length; zeros for resting-state files
│   └── metadata.json          # Optional processing metadata
├── patient_002/
│   └── ...
└── selected_patients.json     # Optional role labels for training/refinement
```

For resting-state recordings, `stimulation.csv` is usually all zeros. The DCNN
and Koopman trainers can overlay synthetic PRBS stimulation with
`--synthetic-stim`, matching the 4YP training setup.

Example `selected_patients.json`:

```json
{
  "patients": [
    {"directory": "patient_001", "role": "training"},
    {"directory": "patient_002", "role": "training"},
    {"directory": "patient_003", "role": "refinement"}
  ]
}
```

Train the DCNN from the processed root:

```bash
python -m dcnn_tube_mpc.training.train_predictor \
  --data-dir ../private_data/processed/aperiodic \
  --input-space linear \
  --patient-role training \
  --synthetic-stim \
  --horizon 5 \
  --save-dir models/dcnn_custom
```

Train the Koopman/ARX model from the same processed root:

```bash
python -m koopman_mpc.training.train_koopman_ols \
  --data-dir ../private_data/processed/aperiodic \
  --input-space linear \
  --patient-role training \
  --synthetic-stim \
  --horizon 7 \
  --model lasso46 \
  --save-dir models/koopman_custom
```

`--input-space linear` is correct for `beta_causal_RMS.csv` produced by the
4YP MATLAB processing; the training scripts convert it to log beta internally.
Use `--input-space log` only if your CSV already stores log beta.

## Benchmark Replay

The benchmark runner expects log-space beta when `real_beta_data` is supplied
to the default log-space simulator. To replay one processed patient folder,
convert its linear beta RMS before constructing the runner:

```python
import numpy as np

from dbs_bench.simulation.simulate import PatientData, SimulationRunner

data_dir = "../private_data/processed/aperiodic/patient_001"
beta_linear = np.loadtxt(f"{data_dir}/beta_causal_RMS.csv", delimiter=",")
stim = np.loadtxt(f"{data_dir}/stimulation.csv", delimiter=",")

n = min(len(beta_linear), len(stim))
beta_log = np.log(np.maximum(beta_linear[:n], 1e-10)).astype(np.float32)
stim = stim[:n].astype(np.float32)

n_state_y = 15
n_state_u = 15
patient = PatientData(
    y_history=beta_log[:n_state_y][::-1],
    u_history=stim[:n_state_u][::-1],
)

runner = SimulationRunner(
    patient_data=patient,
    dt=0.02,
    beta_0=2.3,
    real_beta_data=beta_log[n_state_y:],
)
result = runner.run(ctrl, duration=60.0, controller_type="custom")
print(result.metrics)
```

Runtime histories are newest-first because the controller updates index `0`
with the latest sample. Training windows are stored oldest-to-newest because
they are fixed supervised regressors.

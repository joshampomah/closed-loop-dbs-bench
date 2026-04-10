# Canonical owner: closed-loop-dbs-bench
"""Data schema for private patient data plug-in.

This module defines the exact shape that private patient data must conform to
in order to be used with the simulation harness.  All public repos work with
synthetic data (see data_generator.py).  Researchers with access to patient
recordings can provide a loader that returns these dataclasses.

Example (private loader, NOT included here):
    >>> from dbs_bench.synthetic.schema import PatientRecording, DataPlugIn
    >>> recording = PatientRecording(
    ...     beta=np.load("patient_beta.npy"),
    ...     sample_rate=50.0,
    ...     patient_id="ANON_001",
    ... )
    >>> plugin = DataPlugIn(recordings=[recording])
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class PatientRecording:
    """Single continuous recording session from one patient.

    Attributes:
        beta: Log-space beta power array, shape (n_samples,), dtype float32.
              Represents log(beta_band_power) in units consistent with the
              AR model in device_params.json.
        sample_rate: Sampling rate in Hz (must be 50.0 for current models).
        patient_id: Anonymised patient identifier string.
        stimulation: Optional stimulation signal applied during recording,
                     shape (n_samples,), dtype float32.  If provided, the
                     simulation harness will use the real data simulator
                     (RealBetaSimulator) rather than the synthetic one.
        metadata: Optional dict of arbitrary session metadata (age, condition,
                  electrode location, etc.)
    """

    beta: np.ndarray
    sample_rate: float
    patient_id: str
    stimulation: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.beta.ndim != 1:
            raise ValueError("beta must be a 1-D array")
        if self.beta.dtype != np.float32:
            self.beta = self.beta.astype(np.float32)
        if self.stimulation is not None:
            if self.stimulation.ndim != 1:
                raise ValueError("stimulation must be a 1-D array")
            if len(self.stimulation) != len(self.beta):
                raise ValueError("stimulation must have the same length as beta")
            if self.stimulation.dtype != np.float32:
                self.stimulation = self.stimulation.astype(np.float32)

    @property
    def n_samples(self) -> int:
        """Total number of samples."""
        return len(self.beta)

    @property
    def duration(self) -> float:
        """Recording duration in seconds."""
        return self.n_samples / self.sample_rate


@dataclass
class DataPlugIn:
    """Container for one or more patient recordings.

    Pass this to SimulationRunner or training pipelines to replace
    synthetic data with real patient data.

    Attributes:
        recordings: List of PatientRecording objects.
        split_ratios: Train/validation/test split ratios.  Must sum to 1.0.
    """

    recordings: List[PatientRecording]
    split_ratios: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15}
    )

    def __post_init__(self) -> None:
        if not self.recordings:
            raise ValueError("recordings list must not be empty")
        total = sum(self.split_ratios.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split_ratios must sum to 1.0, got {total}")

    @property
    def n_patients(self) -> int:
        """Number of unique patients."""
        return len({r.patient_id for r in self.recordings})

    @property
    def total_duration(self) -> float:
        """Total recording duration across all sessions, in seconds."""
        return sum(r.duration for r in self.recordings)

    def get_by_patient(self, patient_id: str) -> List[PatientRecording]:
        """Return all recordings for a given patient ID."""
        return [r for r in self.recordings if r.patient_id == patient_id]

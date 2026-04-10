"""Tests for synthetic data generation."""
import numpy as np
import pytest

from dbs_bench.synthetic.data_generator import (
    generate_demo_dataset,
    generate_demo_patient,
    generate_synthetic_beta,
    generate_synthetic_stimulation,
)


def test_synthetic_beta_shape():
    """Synthetic beta has correct shape."""
    beta = generate_synthetic_beta(n_steps=1000)
    assert beta.shape == (1000,)
    assert beta.dtype == np.float32


def test_synthetic_beta_reproducible():
    """Same seed gives same trajectory."""
    b1 = generate_synthetic_beta(n_steps=100, seed=42)
    b2 = generate_synthetic_beta(n_steps=100, seed=42)
    np.testing.assert_array_equal(b1, b2)


def test_synthetic_beta_reasonable_range():
    """Synthetic beta is in a reasonable range for log-space beta."""
    beta = generate_synthetic_beta(n_steps=5000, seed=42)
    assert np.mean(beta) > 0.0  # Log-space beta should be positive
    assert np.std(beta) < 5.0   # Not wildly variable


def test_synthetic_stimulation_shape():
    """Synthetic stimulation has correct shape."""
    u = generate_synthetic_stimulation(n_steps=1000)
    assert u.shape == (1000,)
    assert u.dtype == np.float32


def test_synthetic_stimulation_bounds():
    """Synthetic stimulation respects bounds."""
    u = generate_synthetic_stimulation(n_steps=1000, u_max=0.03)
    assert np.all(u >= 0.0)
    assert np.all(u <= 0.03)


def test_demo_patient():
    """Demo patient has valid structure."""
    patient = generate_demo_patient()
    assert len(patient.y_history) == 15
    assert len(patient.u_history) == 15
    assert patient.stim_gain > 0


def test_demo_dataset():
    """Demo dataset returns matching trajectory and patient."""
    beta, patient = generate_demo_dataset(duration=10.0, dt=0.02)
    assert len(beta) == 500  # 10s at 50Hz
    assert patient is not None

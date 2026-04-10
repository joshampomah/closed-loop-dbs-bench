"""Tests for the simulation harness."""
import numpy as np
import pytest

from dbs_bench.controllers.bang_bang import BangBangController
from dbs_bench.controllers.pi_controller import PIController
from dbs_bench.simulation.simulate import (
    BetaSimulator,
    PatientData,
    SimulationResult,
    SimulationRunner,
    compute_metrics,
)
from dbs_bench.synthetic.data_generator import generate_demo_patient


def test_beta_simulator_step():
    """BetaSimulator produces finite outputs."""
    sim = BetaSimulator(noise_std=0.001)
    y_history = np.full(5, 2.3, dtype=np.float32)
    y = sim.step(y_history, 0.0)
    assert np.isfinite(y)


def test_beta_simulator_warmup():
    """Warmup returns correct buffer shape."""
    sim = BetaSimulator()
    y_hist, params = sim.warm_up(steps=100, buffer_len=15)
    assert y_hist.shape == (15,)
    assert "gain" in params


def test_compute_metrics():
    """Metrics computation returns expected keys."""
    y = np.array([2.5, 2.4, 2.1, 2.0, 2.6])
    u = np.array([0.01, 0.02, 0.01, 0.0, 0.02])
    metrics = compute_metrics(y, u, beta_0=2.3, dt=0.02)
    assert "tracking_error" in metrics
    assert "control_effort" in metrics
    assert "time_above_threshold" in metrics
    assert "mean_stimulation" in metrics
    assert metrics["tracking_error"] >= 0


def test_simulation_runner_bang_bang():
    """SimulationRunner works with bang-bang controller."""
    patient = generate_demo_patient(n_state_y=15)
    runner = SimulationRunner(patient, dt=0.02, beta_0=2.3)
    ctrl = BangBangController(beta_0=2.3)
    result = runner.run(ctrl, duration=2.0, controller_type="bang-bang", show_progress=False)
    assert isinstance(result, SimulationResult)
    assert len(result.y) == 100  # 2s at 50Hz
    assert result.controller_type == "bang-bang"


def test_simulation_runner_pi():
    """SimulationRunner works with PI controller."""
    patient = generate_demo_patient(n_state_y=15)
    runner = SimulationRunner(patient, dt=0.02, beta_0=2.3)
    ctrl = PIController(beta_0=2.3)
    result = runner.run(ctrl, duration=2.0, controller_type="pi", show_progress=False)
    assert isinstance(result, SimulationResult)
    assert len(result.y) == 100


def test_patient_data_create_default():
    """PatientData.create_default works."""
    pd = PatientData.create_default(n_state_y=15, initial_beta=2.5)
    assert len(pd.y_history) == 15
    assert np.allclose(pd.y_history, 2.5)

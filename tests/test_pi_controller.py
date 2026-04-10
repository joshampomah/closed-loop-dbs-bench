"""Tests for PI controller."""
import numpy as np
import pytest

from dbs_bench.controllers.pi_controller import PIController


def test_pi_above_threshold():
    """PI controller applies positive control when y > beta_0."""
    ctrl = PIController(Kp=0.1, Ki=0.01, beta_0=2.3, dt=0.02)
    u = ctrl.compute_control(2.5)
    assert u > 0.0


def test_pi_below_threshold():
    """PI controller has zero error when y <= beta_0 (asymmetric)."""
    ctrl = PIController(Kp=0.1, Ki=0.01, beta_0=2.3, dt=0.02)
    u = ctrl.compute_control(2.0)
    # First step with zero error: du = Kp*(0-0) + Ki*dt*0 = 0
    assert u == pytest.approx(0.0)


def test_pi_rate_saturation():
    """PI controller saturates rate of change at delta_u_max."""
    ctrl = PIController(Kp=100.0, Ki=100.0, beta_0=2.3, delta_u_max=0.0024, dt=0.02)
    u = ctrl.compute_control(5.0)  # Large error
    assert u <= 0.0024 + 1e-10  # Should be rate-limited


def test_pi_amplitude_saturation():
    """PI controller saturates at u_max."""
    ctrl = PIController(Kp=0.1, Ki=0.01, beta_0=2.3, u_max=0.03, dt=0.02)
    # Run many steps to accumulate integral
    for _ in range(10000):
        u = ctrl.compute_control(3.0)
    assert u <= 0.03 + 1e-10


def test_pi_reset():
    """Reset clears state."""
    ctrl = PIController(beta_0=2.3)
    ctrl.compute_control(2.5)
    ctrl.reset()
    assert ctrl._u_prev == 0.0
    assert ctrl._e_prev == 0.0

"""Tests for bang-bang controller."""
import numpy as np
import pytest

from dbs_bench.controllers.bang_bang import BangBangController


def test_bang_bang_above_threshold():
    """Controller ramps up when y > beta_0."""
    ctrl = BangBangController(beta_0=2.3, u_max=0.03, delta_u_max=0.0024)
    u = ctrl.compute_control(2.5)
    assert u == pytest.approx(0.0024)


def test_bang_bang_below_threshold():
    """Controller ramps down when y <= beta_0."""
    ctrl = BangBangController(beta_0=2.3, u_max=0.03, delta_u_max=0.0024)
    ctrl.u_prev = 0.01
    u = ctrl.compute_control(2.0)
    assert u == pytest.approx(0.01 - 0.0024)


def test_bang_bang_saturates_at_max():
    """Controller saturates at u_max."""
    ctrl = BangBangController(beta_0=2.3, u_max=0.03, delta_u_max=0.03)
    ctrl.u_prev = 0.025
    u = ctrl.compute_control(2.5)
    assert u == pytest.approx(0.03)


def test_bang_bang_saturates_at_zero():
    """Controller saturates at 0."""
    ctrl = BangBangController(beta_0=2.3, u_max=0.03, delta_u_max=0.03)
    ctrl.u_prev = 0.001
    u = ctrl.compute_control(2.0)
    assert u == pytest.approx(0.0)


def test_bang_bang_reset():
    """Reset clears state."""
    ctrl = BangBangController(beta_0=2.3)
    ctrl.compute_control(2.5)
    ctrl.reset()
    assert ctrl.u_prev == 0.0

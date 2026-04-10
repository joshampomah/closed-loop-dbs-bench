"""Tests for Linear MPC controller."""
import numpy as np
import pytest

from dbs_bench.config.device_config import get_device_config


def test_device_config_loads():
    """Device config loads without error."""
    config = get_device_config()
    assert config.stimulation.gain > 0
    assert config.constraints.u_max > 0
    assert config.sample_time > 0
    assert len(config.beta.ar_coefficients) > 0

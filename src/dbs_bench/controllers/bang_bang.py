# Canonical owner: closed-loop-dbs-bench
"""Bang-bang (on-off) controller with ramp for closed-loop DBS (CDC25)."""
from __future__ import annotations

import numpy as np

from dbs_bench.config.device_config import get_device_config

_DEVICE_CONFIG = get_device_config()


class BangBangController:
    """On-off (bang-bang) controller with ramp (CDC25).

    Increments/decrements control by delta_u_max each step:
    u_k = min(u_{k-1} + delta_u_max, u_max) if y_k > beta_0
    u_k = max(u_{k-1} - delta_u_max, 0)     otherwise
    """

    def __init__(
        self,
        beta_0: float = 2.3,
        u_max: float = _DEVICE_CONFIG.constraints.u_max,
        delta_u_max: float = _DEVICE_CONFIG.constraints.delta_u_max,
    ):
        self.beta_0 = beta_0
        self.u_max = u_max
        self.delta_u_max = delta_u_max
        self.u_prev = 0.0

    def reset(self) -> None:
        self.u_prev = 0.0

    def compute_control(self, y: float, **kwargs) -> float:
        if y > self.beta_0:
            u = min(self.u_prev + self.delta_u_max, self.u_max)
        else:
            u = max(self.u_prev - self.delta_u_max, 0.0)
        self.u_prev = u
        return u

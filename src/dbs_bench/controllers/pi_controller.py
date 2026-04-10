# Canonical owner: closed-loop-dbs-bench
"""Proportional-Integral controller for closed-loop DBS (CDC24 Equation 14)."""
from __future__ import annotations

import numpy as np

from dbs_bench.config.device_config import get_device_config

_DEVICE_CONFIG = get_device_config()


class PIController:
    """Proportional-Integral controller (CDC24 Equation 14).

    Difference form: u_k = u_{k-1} + K_P(e_k - e_{k-1}) + K_I*T_s*e_k
    Asymmetric error: e_k = [y_k - beta_0]_{>=0}
    """

    def __init__(
        self,
        Kp: float = 0.1,
        Ki: float = 0.01,
        beta_0: float = 2.3,
        u_min: float = _DEVICE_CONFIG.constraints.u_min,
        u_max: float = _DEVICE_CONFIG.constraints.u_max,
        delta_u_max: float = _DEVICE_CONFIG.constraints.delta_u_max,
        dt: float = 0.02,
    ):
        self.Kp = Kp
        self.Ki = Ki
        self.beta_0 = beta_0
        self.u_min = u_min
        self.u_max = u_max
        self.delta_u_max = delta_u_max
        self.dt = dt
        self._u_prev = 0.0
        self._e_prev = 0.0

    def reset(self) -> None:
        self._u_prev = 0.0
        self._e_prev = 0.0

    def compute_control(self, y: float, **kwargs) -> float:
        e_k = max(0.0, y - self.beta_0)
        du = self.Kp * (e_k - self._e_prev) + self.Ki * self.dt * e_k
        du = np.clip(du, -self.delta_u_max, self.delta_u_max)
        u_raw = self._u_prev + du
        u = np.clip(u_raw, self.u_min, self.u_max)
        self._u_prev = u
        self._e_prev = e_k
        return u

# Canonical owner: closed-loop-dbs-bench
"""Simulation harness for closed-loop DBS benchmarking.

Supports bang-bang, PI, multi-step ARX, and custom (plug-in) controllers.
Method-repo controllers (DCNN-MPC, Koopman MPC) can be plugged in via the
ControllerProtocol interface.

Example:
    >>> from dbs_bench.simulation.simulate import simulate_trial, PatientData
    >>> from dbs_bench.synthetic.data_generator import generate_demo_patient
    >>> patient = generate_demo_patient()
    >>> result = simulate_trial("bang-bang", patient, duration=10.0)
"""
from __future__ import annotations

import math
import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Callable, Dict, List, Literal, Optional, Protocol, Tuple,
    Union, runtime_checkable,
)

import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from dbs_bench.config.device_config import get_device_config

_DEVICE_CONFIG = get_device_config()


# ---------------------------------------------------------------------------
# Controller Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class ControllerProtocol(Protocol):
    """Interface for external controllers (e.g. from method repos).

    Any object with compute_control and reset satisfies this protocol.
    """

    def compute_control(self, *args, **kwargs): ...
    def reset(self) -> None: ...


def _controller_accepts_history(controller: object) -> bool:
    """Return true if compute_control expects y/u histories positionally."""
    method = getattr(controller, "compute_control", None)
    if method is None:
        return False
    try:
        sig = inspect.signature(method)
    except (TypeError, ValueError):
        return False

    required_positional = [
        p for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.default is p.empty
    ]
    return len(required_positional) >= 3


def _call_controller(
    controller: Union[ControllerProtocol, Callable],
    y_obs: float,
    y_history: np.ndarray,
    u_history: np.ndarray,
    u_prev: float,
):
    """Call scalar or history-aware controllers with a common interface."""
    if hasattr(controller, "compute_control"):
        method = controller.compute_control
        if _controller_accepts_history(controller):
            return method(y_history.copy(), u_history.copy(), u_prev)

        try:
            return method(
                y_obs,
                y_history=y_history.copy(),
                u_history=u_history.copy(),
                u_prev=u_prev,
            )
        except TypeError:
            return method(y_obs)

    if callable(controller):
        return controller(y_obs)

    raise ValueError("Controller must be callable or have compute_control method")


def _normalise_control_output(control_output):
    """Accept either u or (u, info) controller outputs."""
    if isinstance(control_output, tuple):
        if len(control_output) == 0:
            raise ValueError("Controller returned an empty tuple")
        u_k = control_output[0]
        info = control_output[1] if len(control_output) > 1 else None
    else:
        u_k = control_output
        info = None
    return float(u_k), info


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Result of a closed-loop simulation trial.

    Attributes:
        time: Time vector of shape (n_steps,).
        y: Output (beta power) of shape (n_steps,).
        u: Control input (stimulation) of shape (n_steps,).
        y_ref: Reference threshold (beta_0).
        metrics: Performance metrics dictionary.
        controller_type: Type of controller used.
        solver_info: Optional solver diagnostics.
        params: Controller parameters used.
        eta: Optional stimulation effect trajectory (from RealBetaSimulator).
    """
    time: np.ndarray
    y: np.ndarray
    u: np.ndarray
    y_ref: float
    metrics: Dict[str, float]
    controller_type: str
    solver_info: Optional[Dict] = None
    params: Dict = field(default_factory=dict)
    eta: Optional[np.ndarray] = None
    final_y_history: Optional[np.ndarray] = None
    final_u_history: Optional[np.ndarray] = None
    final_u_prev: Optional[float] = None
    final_sim_state: Optional[Dict] = None

    @property
    def n_steps(self) -> int:
        return len(self.time)

    @property
    def duration(self) -> float:
        return self.time[-1] - self.time[0] if len(self.time) > 1 else 0.0


@dataclass
class PatientData:
    """Patient data for simulation.

    Attributes:
        y_history: Initial state history of shape (n_state_y,).
        u_history: Initial control history of shape (n_state_u,).
        stim_gain: Stimulation gain (nominal: 62.11).
        stim_tau1: Time constant 1 (nominal: 0.05).
        stim_tau2: Time constant 2 (nominal: 0.25).
        beta_ar_coeffs: AR coefficients for beta dynamics.
        noise_std: Process noise standard deviation.
    """
    y_history: np.ndarray
    u_history: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float32))
    stim_gain: float = 62.11
    stim_tau1: float = 0.05
    stim_tau2: float = 0.25
    beta_ar_coeffs: Tuple[float, ...] = (0.35, -0.08, 0.04)
    noise_std: float = 0.0012

    @classmethod
    def create_default(cls, n_state_y: int = 15, initial_beta: float = 2.5) -> "PatientData":
        """Create default patient data with given initial conditions."""
        y_history = np.full(n_state_y, initial_beta, dtype=np.float32)
        return cls(y_history=y_history)

    @classmethod
    def from_ari_model(cls, mat_path: str, n_state_y: int = 15,
                       initial_beta: Optional[float] = None) -> "PatientData":
        """Load patient-specific parameters from ARI model .mat file.

        Note: .mat files are not included in this repository. Provide your
        own data via the PatientRecording schema in dbs_bench.synthetic.schema.
        """
        import h5py
        with h5py.File(mat_path, 'r') as f:
            stim_gain = float(f["stim_params"]["k"][0, 0])
            stim_tau1 = float(f["stim_params"]["tau1"][0, 0])
            stim_tau2 = float(f["stim_params"]["tau2"][0, 0])
            alpha_params = np.array(f['alpha_params']).flatten()
            beta_0 = float(f["beta_0"][0, 0])
            if initial_beta is None:
                initial_beta = beta_0 + 0.1
        y_history = np.full(n_state_y, initial_beta, dtype=np.float32)
        return cls(
            y_history=y_history,
            stim_gain=stim_gain,
            stim_tau1=stim_tau1,
            stim_tau2=stim_tau2,
            beta_ar_coeffs=tuple(alpha_params),
        )


# ---------------------------------------------------------------------------
# Simulators
# ---------------------------------------------------------------------------

class BetaSimulator:
    """Simulates beta power dynamics with stimulation effect (synthetic).

    Uses an AR process for natural beta and a 2nd-order ZOH state-space
    model for the stimulation attenuation effect.
    """

    def __init__(
        self,
        stim_gain: float = 62.11,
        stim_tau1: float = 0.05,
        stim_tau2: float = 0.25,
        beta_ar_coeffs: Tuple[float, ...] = (0.35, -0.08, 0.04),
        noise_std: float = 0.0012,
        dt: float = 0.02,
    ):
        self.base_params = {"gain": stim_gain, "tau1": stim_tau1, "tau2": stim_tau2}
        self.params = self.base_params.copy()
        self.beta_ar_coeffs = beta_ar_coeffs
        self.noise_std = noise_std
        self.dt = dt
        self._update_matrices()
        self._stim_state = np.zeros((2, 1), dtype=np.float32)
        self._natural_history = np.zeros(len(beta_ar_coeffs), dtype=np.float32)

    def _update_matrices(self) -> None:
        g, t1, t2, dt = (self.params["gain"], self.params["tau1"],
                         self.params["tau2"], self.dt)
        e1, e2 = math.exp(-dt / t1), math.exp(-dt / t2)
        inv_diff = 1.0 / (1.0 / t2 - 1.0 / t1)
        ad10 = (g / t2) * (e1 - e2) * inv_diff
        bd0 = 1.0 - e1
        bd1 = g * (1.0 - e1) - (t2 / t1) * ad10
        self.Ad = np.array([[e1, 0.0], [ad10, e2]])
        self.Bd = np.array([[bd0], [bd1]])
        self.Cd = np.array([[0.0, 1.0]])

    def reset(self, initial_y_history: Optional[np.ndarray] = None,
              initial_params: Optional[Dict] = None,
              initial_state: Optional[Dict] = None) -> None:
        self._stim_state = np.zeros((2, 1), dtype=np.float32)
        self.params = (initial_params or self.base_params).copy()
        self._update_matrices()
        n_ar = len(self.beta_ar_coeffs)
        if initial_y_history is not None:
            if len(initial_y_history) >= n_ar:
                self._natural_history = initial_y_history[:n_ar].copy().astype(np.float32)
            else:
                self._natural_history = np.zeros(n_ar, dtype=np.float32)
                self._natural_history[:len(initial_y_history)] = initial_y_history
        else:
            self._natural_history = np.zeros(n_ar, dtype=np.float32)
        if initial_state is not None:
            if initial_state.get("params"):
                self.params = dict(initial_state["params"])
                self._update_matrices()
            if initial_state.get("stim_state") is not None:
                self._stim_state = np.asarray(initial_state["stim_state"], dtype=np.float32).reshape(2, 1)
            if initial_state.get("natural_history") is not None:
                nh = np.asarray(initial_state["natural_history"], dtype=np.float32)
                if nh.shape == self._natural_history.shape:
                    self._natural_history = nh.copy()

    def get_state(self) -> Dict:
        return {"params": self.params.copy(),
                "stim_state": self._stim_state.copy(),
                "natural_history": self._natural_history.copy()}

    def warm_up(self, steps: int = 250, buffer_len: int = 15) -> Tuple[np.ndarray, Dict]:
        history_buffer: List[float] = []
        for _ in range(steps):
            y_val = self.step(np.zeros(1, dtype=np.float32), 0.0)
            history_buffer.append(y_val)
            if len(history_buffer) > buffer_len:
                history_buffer.pop(0)
        if len(history_buffer) < buffer_len:
            last = history_buffer[-1] if history_buffer else 0.0
            history_buffer = [last] * (buffer_len - len(history_buffer)) + history_buffer
        return np.array(history_buffer)[::-1], self.params.copy()

    def step(self, y_history: np.ndarray, u: float) -> float:
        ar_term = sum(c * self._natural_history[i] for i, c in enumerate(self.beta_ar_coeffs))
        noise = np.random.randn() * self.noise_std
        y_natural_new = ar_term + noise + 0.015
        stim_effect = (self.Cd @ self._stim_state).item()
        self._stim_state = self.Ad @ self._stim_state + self.Bd * u
        y_observed = y_natural_new - stim_effect
        self._natural_history = np.roll(self._natural_history, 1)
        self._natural_history[0] = y_natural_new
        return y_observed


class RealBetaSimulator:
    """Simulates beta dynamics using real patient data with stimulation effect.

    Natural beta comes from real recordings; stimulation suppresses beta
    via 2nd-order state-space dynamics.

    Note: Patient data is not included in this repository. Use generate_demo_dataset()
    for synthetic data, or provide your own data via the PatientRecording schema.
    """

    def __init__(
        self,
        beta_data: np.ndarray,
        stim_gain: float = 62.11,
        stim_tau1: float = 0.05,
        stim_tau2: float = 0.25,
        dt: float = 0.02,
        eta_matrices: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        output_space: str = "log",
        stim_drift_fn: Optional[Callable[[int, dict], Optional[dict]]] = None,
    ):
        self.beta_data = beta_data
        self.n_samples = len(beta_data)
        self.dt = dt
        self._step = 0
        self.output_space = output_space
        self._stim_drift_fn = stim_drift_fn
        self._use_external_matrices = eta_matrices is not None
        if output_space not in ["log", "linear"]:
            raise ValueError(f"output_space must be 'log' or 'linear', got {output_space}")
        self.base_params = {"gain": stim_gain, "tau1": stim_tau1, "tau2": stim_tau2}
        self.params = self.base_params.copy()
        if eta_matrices is not None:
            Ad, Bd, Cd = eta_matrices
            self.Ad = np.asarray(Ad)
            self.Bd = np.asarray(Bd).flatten()
            self.Cd = np.asarray(Cd).flatten()
            self.Dd = 0.0
            self._n_eta_states = self.Ad.shape[0]
        else:
            self._build_eta_model(stim_gain, stim_tau1, stim_tau2, dt)
            self._n_eta_states = 2
        self._eta_state = np.zeros(self._n_eta_states, dtype=np.float32)
        self._current_eta = 0.0

    def _build_eta_model(self, k, tau1, tau2, dt):
        a1 = 1.0 / tau1 + 1.0 / tau2
        a0 = 1.0 / (tau1 * tau2)
        b0 = k / (tau1 * tau2)
        e1, e2 = math.exp(-dt / tau1), math.exp(-dt / tau2)
        d = 1.0 / tau1 - 1.0 / tau2
        ad00 = (e2 / tau1 - e1 / tau2) / d
        ad01 = (e2 - e1) / d
        ad10 = a0 * (e1 - e2) / d
        ad11 = (e1 / tau1 - e2 / tau2) / d
        v0 = ad01 * b0
        v1 = (ad11 - 1.0) * b0
        bd0 = -(tau1 + tau2) * v0 - tau1 * tau2 * v1
        bd1 = v0
        self.Ad = np.array([[ad00, ad01], [ad10, ad11]])
        self.Bd = np.array([bd0, bd1])
        self.Cd = np.array([1.0, 0.0])
        self.Dd = 0.0

    def _update_matrices(self):
        if not self._use_external_matrices:
            self._build_eta_model(self.params["gain"], self.params["tau1"],
                                   self.params["tau2"], self.dt)

    def reset(self, initial_y_history=None, initial_params=None, initial_state=None):
        self._step = 0
        self._eta_state = np.zeros(self._n_eta_states, dtype=np.float32)
        self._current_eta = 0.0
        self.params = (initial_params or self.base_params).copy()
        self._update_matrices()
        if initial_state is not None:
            if initial_state.get("params"):
                self.params = dict(initial_state["params"])
            if initial_params is not None:
                self.params = initial_params.copy()
            self._update_matrices()
            if initial_state.get("step") is not None:
                self._step = int(np.clip(int(initial_state["step"]), 0, self.n_samples))
            if initial_state.get("eta_state") is not None:
                es = np.asarray(initial_state["eta_state"], dtype=np.float32).reshape(-1)
                if es.shape == self._eta_state.shape:
                    self._eta_state = es.copy()
            self._current_eta = float(initial_state.get("current_eta",
                                                          float(self.Cd @ self._eta_state)))

    def step(self, u: float) -> float:
        if self._step >= self.n_samples:
            raise IndexError(f"Exceeded patient data length ({self.n_samples} samples)")
        if self._stim_drift_fn is not None:
            new_params = self._stim_drift_fn(self._step, self.base_params)
            if new_params is not None:
                self.params = new_params
                self._update_matrices()
        beta_log = self.beta_data[self._step]
        eta = float(self.Cd @ self._eta_state + self.Dd * u)
        self._current_eta = eta
        self._eta_state = self.Ad @ self._eta_state + self.Bd * u
        self._step += 1
        if self.output_space == "log":
            return float(beta_log - eta)
        else:
            return float(np.exp(beta_log - eta))

    def warm_up(self, steps: int = 250, buffer_len: int = 15) -> Tuple[np.ndarray, Dict]:
        history_buffer: List[float] = []
        for _ in range(steps):
            y_val = self.step(0.0)
            history_buffer.append(y_val)
            if len(history_buffer) > buffer_len:
                history_buffer.pop(0)
        if len(history_buffer) < buffer_len:
            last = history_buffer[-1] if history_buffer else 0.0
            history_buffer = [last] * (buffer_len - len(history_buffer)) + history_buffer
        return np.array(history_buffer)[::-1], self.params.copy()

    def get_state(self) -> Dict:
        return {"params": self.params.copy(), "step": int(self._step),
                "eta_state": self._eta_state.copy(), "current_eta": float(self._current_eta)}

    @property
    def current_step(self) -> int:
        return self._step

    @property
    def remaining_steps(self) -> int:
        return self.n_samples - self._step

    @property
    def current_eta(self) -> float:
        return self._current_eta


def generate_per_trial_params(
    n_trials: int,
    base_gain: float = 62.11,
    base_tau1: float = 0.05,
    base_tau2: float = 0.25,
    max_total_deviation: float = 0.40,
    seed: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Generate i.i.d. per-trial stimulation parameters (±40% uniform)."""
    rng = np.random.default_rng(seed)
    bases = {"gain": base_gain, "tau1": base_tau1, "tau2": base_tau2}
    result: List[Dict[str, float]] = []
    for _ in range(n_trials):
        new_params: Dict[str, float] = {}
        for key, base_val in bases.items():
            lower = base_val * (1 - max_total_deviation)
            upper = base_val * (1 + max_total_deviation)
            new_params[key] = float(rng.uniform(lower, upper))
        result.append(new_params)
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y: np.ndarray, u: np.ndarray, beta_0: float, dt: float) -> Dict[str, float]:
    """Compute performance metrics for a simulation."""
    above_threshold = np.maximum(y - beta_0, 0)
    tracking_error = np.mean(above_threshold ** 2)
    control_effort = np.sum(u ** 2) * dt
    time_above = np.mean(y > beta_0)
    mean_stim = np.mean(u)
    mean_u_squared = np.mean(u ** 2)
    suppression_efficiency = ((1 - time_above) / (control_effort + 1e-10)
                               if control_effort > 0 else 0.0)
    return {
        "tracking_error": float(tracking_error),
        "control_effort": float(control_effort),
        "time_above_threshold": float(time_above),
        "mean_stimulation": float(mean_stim),
        "mean_u_squared": float(mean_u_squared),
        "suppression_efficiency": float(suppression_efficiency),
        "mean_beta": float(np.mean(y)),
        "max_beta": float(np.max(y)),
        "min_beta": float(np.min(y)),
        "std_beta": float(np.std(y)),
    }


# ---------------------------------------------------------------------------
# Simulation Runner
# ---------------------------------------------------------------------------

class SimulationRunner:
    """Runs closed-loop simulations with any controller implementing ControllerProtocol."""

    def __init__(
        self,
        patient_data: PatientData,
        dt: float = 0.02,
        beta_0: float = 2.3,
        real_beta_data: Optional[np.ndarray] = None,
        eta_matrices: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        output_space: str = "log",
        stim_drift_fn: Optional[Callable[[int, dict], Optional[dict]]] = None,
    ):
        self.patient_data = patient_data
        self.dt = dt
        self.beta_0 = beta_0
        self.use_real_data = real_beta_data is not None
        self.output_space = output_space
        if real_beta_data is not None:
            self.simulator = RealBetaSimulator(
                beta_data=real_beta_data,
                stim_gain=patient_data.stim_gain,
                stim_tau1=patient_data.stim_tau1,
                stim_tau2=patient_data.stim_tau2,
                dt=dt,
                eta_matrices=eta_matrices,
                output_space=output_space,
                stim_drift_fn=stim_drift_fn,
            )
        else:
            self.simulator = BetaSimulator(
                stim_gain=patient_data.stim_gain,
                stim_tau1=patient_data.stim_tau1,
                stim_tau2=patient_data.stim_tau2,
                beta_ar_coeffs=patient_data.beta_ar_coeffs,
                noise_std=patient_data.noise_std,
                dt=dt,
            )

    def run(
        self,
        controller: Union[ControllerProtocol, Callable],
        duration: float = 220.0,
        controller_type: str = "unknown",
        verbose: bool = False,
        initial_params: Optional[Dict[str, float]] = None,
        initial_y_history: Optional[np.ndarray] = None,
        initial_u_history: Optional[np.ndarray] = None,
        initial_u_prev: Optional[float] = None,
        initial_sim_state: Optional[Dict] = None,
        reset_controller: bool = True,
        show_progress: bool = True,
        warmup_steps: int = 0,
        step_callback: Optional[Callable[[int, object], None]] = None,
    ) -> SimulationResult:
        """Run closed-loop simulation."""
        n_steps = int(duration / self.dt)
        total_steps = warmup_steps + n_steps

        time_arr = np.zeros(n_steps, dtype=np.float32)
        y = np.zeros(n_steps, dtype=np.float32)
        u = np.zeros(n_steps, dtype=np.float32)
        eta_arr = np.zeros(n_steps, dtype=np.float32) if self.use_real_data else None
        solver_info_steps: List[object] = []

        y_history = (np.asarray(initial_y_history, dtype=np.float32).copy()
                     if initial_y_history is not None
                     else self.patient_data.y_history.copy())
        u_history = (np.asarray(initial_u_history, dtype=np.float32).copy()
                     if initial_u_history is not None
                     else self.patient_data.u_history.copy())
        u_prev = float(initial_u_prev) if initial_u_prev is not None else 0.0

        self.simulator.reset(
            initial_y_history=y_history,
            initial_params=initial_params,
            initial_state=initial_sim_state,
        )

        if reset_controller and hasattr(controller, 'reset'):
            controller.reset()

        use_tqdm = show_progress and TQDM_AVAILABLE
        step_iterator = range(total_steps)
        if use_tqdm:
            tqdm_bar = tqdm(step_iterator, desc=f"  {controller_type}",
                            unit="step", ncols=80, leave=True)
            step_iterator = tqdm_bar
        else:
            tqdm_bar = None

        for k_total in step_iterator:
            is_warmup = k_total < warmup_steps
            k = k_total - warmup_steps

            y_obs = y_history[0]
            if not is_warmup:
                time_arr[k] = k * self.dt
                y[k] = y_obs

            control_output = _call_controller(
                controller, y_obs, y_history, u_history, u_prev
            )
            u_k, info = _normalise_control_output(control_output)
            if info is not None and not is_warmup:
                solver_info_steps.append(info)

            if not is_warmup:
                u[k] = u_k

            # Plant step
            if self.use_real_data:
                y_new = self.simulator.step(u_k)
                if not is_warmup and eta_arr is not None:
                    eta_arr[k] = self.simulator.current_eta
            else:
                y_new = self.simulator.step(y_history, u_k)

            # Update histories
            y_history = np.roll(y_history, 1)
            y_history[0] = y_new
            u_history = np.roll(u_history, 1)
            u_history[0] = u_k
            u_prev = u_k

            if step_callback is not None and not is_warmup:
                step_callback(k, controller)

            if verbose and k_total % 1000 == 0:
                print(f"Step {k_total}/{total_steps}, y={y_obs:.3f}, u={u_k:.4f}")

        metrics = compute_metrics(y, u, self.beta_0, self.dt)
        final_sim_state = self.simulator.get_state() if hasattr(self.simulator, "get_state") else None

        return SimulationResult(
            time=time_arr,
            y=y,
            u=u,
            y_ref=self.beta_0,
            metrics=metrics,
            controller_type=controller_type,
            solver_info=({"steps": solver_info_steps} if solver_info_steps else None),
            eta=eta_arr,
            final_y_history=y_history.copy(),
            final_u_history=u_history.copy(),
            final_u_prev=float(u_prev),
            final_sim_state=final_sim_state,
        )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

ControllerType = Literal["bang-bang", "pi", "multistep-arx", "custom"]


def simulate_trial(
    controller_type: ControllerType,
    patient_data: Optional[PatientData] = None,
    duration: float = 220.0,
    params: Optional[Dict] = None,
    predictor=None,
    dt: float = 0.02,
    beta_0: float = 2.3,
    verbose: bool = False,
    initial_params: Optional[Dict[str, float]] = None,
    initial_y_history: Optional[np.ndarray] = None,
    initial_u_history: Optional[np.ndarray] = None,
    initial_u_prev: Optional[float] = None,
    initial_sim_state: Optional[Dict] = None,
    reset_controller: bool = True,
    real_beta_data: Optional[np.ndarray] = None,
    show_progress: bool = True,
    warmup_steps: int = 0,
    stim_drift_fn: Optional[Callable[[int, dict], Optional[dict]]] = None,
    custom_controller=None,
) -> SimulationResult:
    """Unified simulation interface for baseline controllers.

    For DCNN-MPC, Koopman MPC, or other method-repo controllers, use
    SimulationRunner directly with your controller instance.

    Args:
        controller_type: One of "bang-bang", "pi", "multistep-arx", "custom".
        patient_data: PatientData instance. If None, creates a default.
        predictor: Required for "multistep-arx" (ARXModel instance).
        custom_controller: Required for "custom" (must implement ControllerProtocol).
    """
    from dbs_bench.controllers.bang_bang import BangBangController
    from dbs_bench.controllers.pi_controller import PIController

    params = params or {}

    if patient_data is None:
        patient_data = PatientData.create_default(initial_beta=beta_0 + 0.2)

    runner = SimulationRunner(
        patient_data, dt=dt, beta_0=beta_0,
        real_beta_data=real_beta_data,
        output_space="log",
        stim_drift_fn=stim_drift_fn,
    )

    if controller_type == "bang-bang":
        controller = BangBangController(
            beta_0=beta_0,
            u_max=params.get("u_max", _DEVICE_CONFIG.constraints.u_max),
            delta_u_max=params.get("delta_u_max", _DEVICE_CONFIG.constraints.delta_u_max),
        )

    elif controller_type == "pi":
        controller = PIController(
            Kp=params.get("Kp", 0.1),
            Ki=params.get("Ki", 0.01),
            beta_0=beta_0,
            u_min=params.get("u_min", _DEVICE_CONFIG.constraints.u_min),
            u_max=params.get("u_max", _DEVICE_CONFIG.constraints.u_max),
            delta_u_max=params.get("delta_u_max", _DEVICE_CONFIG.constraints.delta_u_max),
            dt=dt,
        )

    elif controller_type == "multistep-arx":
        if predictor is None:
            raise ValueError("predictor (ARXModel) required for multistep-arx")
        # ARX controller wraps the model with a simple greedy step
        class _ARXController:
            def __init__(self, arx_model, config):
                self._model = arx_model
                self._cfg = config
                self._u_prev = 0.0

            def reset(self):
                self._u_prev = 0.0

            def compute_control(self, y: float, y_history=None, u_history=None, **kwargs) -> float:
                # Greedy: use u=0 nominal and return u_max/2 if above threshold
                u = _DEVICE_CONFIG.constraints.u_max / 2.0 if y > self._cfg["beta_0"] else 0.0
                u = float(np.clip(u, 0, _DEVICE_CONFIG.constraints.u_max))
                self._u_prev = u
                return u

        controller = _ARXController(predictor, {"beta_0": beta_0})

    elif controller_type == "custom":
        if custom_controller is None:
            raise ValueError("custom_controller required for custom type")
        controller = custom_controller

    else:
        raise ValueError(f"Unknown controller_type: {controller_type!r}. "
                         "Use 'bang-bang', 'pi', 'multistep-arx', or 'custom'.")

    return runner.run(
        controller,
        duration=duration,
        controller_type=controller_type,
        verbose=verbose,
        initial_params=initial_params,
        initial_y_history=initial_y_history,
        initial_u_history=initial_u_history,
        initial_u_prev=initial_u_prev,
        initial_sim_state=initial_sim_state,
        reset_controller=reset_controller,
        show_progress=show_progress,
        warmup_steps=warmup_steps,
    )

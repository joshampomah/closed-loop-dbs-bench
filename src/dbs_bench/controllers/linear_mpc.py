# Canonical owner: closed-loop-dbs-bench
"""Linear MPC controller for DBS in velocity form.

This module implements the Linear MPC controller matching the MATLAB dbsMPC.m
implementation for CDC25 comparison.

The controller solves a finite-horizon optimal control problem with asymmetric
cost that only penalizes beta above the pathological threshold.

Cost function (CDC24 Equation 11):
    J = Σ_{i=0}^{N-1} ( Q·[y_{k+i} - β₀]²_{≥0} + R_Δ·(Δu_{k+i})² + R·u_{k+i}² )
        + Q_f·||x_{k+N}||²

Example:
    >>> from dbs_bench.controllers.linear_mpc import LinearMPC, LinearMPCConfig
    >>> config = LinearMPCConfig(Q=50000.0, R_delta=1.0)
    >>> mpc = LinearMPC(A_delta, B_delta, C_delta, config)
    >>> u = mpc.compute_control(chi_delta)
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple

import cvxpy as cp
import numpy as np

from dbs_bench.config.device_config import get_device_config

_DEVICE_CONFIG = get_device_config()


@dataclass
class LinearMPCConfig:
    """Configuration for Linear MPC controller.

    Attributes:
        N: Prediction horizon (default: 5).
        Q: Output tracking weight for asymmetric cost (default: 50000.0).
        R_delta: Control rate penalty for Δu (default: 0.0).
        R: Control magnitude penalty for u (default: 1.0).
        Q_f: Terminal state weight (default: 0.0).
        threshold: Pathological beta threshold (default: 2.3).
            IMPORTANT: This must be in the same domain as the plant output C_delta.
            For strict CDC25 compliance, this should be ξ₀ (log-domain threshold).
        u_min: Minimum stimulation (default: 0.0).
        u_max: Maximum stimulation (default: 0.02).
        delta_u_max: Maximum control rate change per step (default: 0.02).
        y_min: Optional minimum output constraint.
        y_max: Optional maximum output constraint.
        solver: CVXPY solver to use (default: "OSQP").
        solver_verbose: Enable verbose solver output (default: False).
        solver_warm_start: Enable solver warm start (default: True).
    """

    N: int = 5
    Q: float = 50000.0
    R_delta: float = 0.0
    R: float = 1.0  # Was R_mag
    Q_f: float = 0.0
    threshold: float = 2.3  # Was beta_0
    u_min: float = _DEVICE_CONFIG.constraints.u_min
    u_max: float = _DEVICE_CONFIG.constraints.u_max
    delta_u_max: float = _DEVICE_CONFIG.constraints.delta_u_max
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    solver: str = "OSQP"
    solver_verbose: bool = False
    solver_warm_start: bool = True

    # Legacy aliases (InitVar won't be stored in instance)
    beta_0: dataclasses.InitVar[Optional[float]] = None
    R_mag: dataclasses.InitVar[Optional[float]] = None

    def __post_init__(self, beta_0: Optional[float], R_mag: Optional[float]):
        """Handle legacy aliases."""
        if beta_0 is not None:
            if self.threshold != 2.3 and self.threshold != beta_0:
                raise ValueError(
                    f"Conflicting values for threshold ({self.threshold}) and beta_0 ({beta_0})"
                )
            self.threshold = beta_0

        if R_mag is not None:
            if self.R != 1.0 and self.R != R_mag:
                raise ValueError(
                    f"Conflicting values for R ({self.R}) and R_mag ({R_mag})"
                )
            self.R = R_mag

        # Scale Q, R, R_delta so that R·u_max² is at least MIN_R_COST.
        # Only the Q/R ratio matters for the optimal solution, but the absolute
        # magnitude of R·u² must exceed the solver's optimality tolerance (~1e-7
        # for MOSEK). Without this, the solver treats tiny R·u² costs as zero,
        # causing a "control floor" where MPC maintains unnecessary stimulation.
        MIN_R_COST = 0.01  # R * u_max² should be at least this
        r_cost = self.R * self.u_max ** 2
        if r_cost > 0 and r_cost < MIN_R_COST:
            scale = MIN_R_COST / r_cost
            self.Q *= scale
            self.R *= scale
            self.R_delta *= scale
            self.Q_f *= scale


class LinearMPC:
    """Linear MPC controller for DBS in velocity form.

    Implements the QP-based MPC controller from MATLAB dbsMPC.m using CVXPY.

    The controller operates in velocity form where the decision variable is
    Δu (change in control) rather than u directly. This provides better
    numerical properties and natural rate limiting.

    State dynamics (velocity form):
        χ^Δ_{k+1} = A^Δ χ^Δ_k + B^Δ Δu_k
        y_k = C^Δ χ^Δ_k

    Paper Alignment Note (CDC25):
        The paper describes Linear MPC operating on ξ = ln(y).
        Ensure that the matrix model (A_delta, B_delta, C_delta) represents
        the log-beta dynamics and that `config.threshold` is set to ξ₀.
        If the model represents raw envelope dynamics, the controller will
        still function but as a "Linear Envelope MPC" rather than strict CDC25.

    The asymmetric cost is implemented using slack variables:
        s_i ≥ y_i - threshold  (slack exceeds positive error)
        s_i ≥ 0                (slack non-negative)

    Attributes:
        config: MPC configuration parameters.
        A_delta: Velocity form state matrix.
        B_delta: Velocity form input matrix.
        C_delta: Velocity form output matrix.
    """

    def __init__(
        self,
        A_delta: np.ndarray,
        B_delta: np.ndarray,
        C_delta: np.ndarray,
        config: Optional[LinearMPCConfig] = None,
    ):
        """Initialize Linear MPC controller.

        Args:
            A_delta: Velocity form state matrix of shape (n_states, n_states).
            B_delta: Velocity form input matrix of shape (n_states, 1).
            C_delta: Velocity form output matrix of shape (1, n_states).
            config: MPC configuration. Uses defaults if None.
        """
        self.config = config or LinearMPCConfig()
        self.A_delta = np.asarray(A_delta)
        self.B_delta = np.asarray(B_delta).reshape(-1, 1)

        # Handle C matrix (allow multi-output)
        self.C_delta = np.asarray(C_delta)
        if self.C_delta.ndim == 1:
            self.C_delta = self.C_delta.reshape(1, -1)

        self.n_states = self.A_delta.shape[0]
        self.N = self.config.N

        # Build prediction matrices (cached)
        self._build_prediction_matrices()

        # Build cumulative sum matrix for u from Δu
        self.Cum = np.tril(np.ones((self.N, self.N)))

        # Setup CVXPY problem (will be parameterized)
        self._setup_qp_problem()

        # State tracking
        self._u_prev = 0.0

    def _build_prediction_matrices(self) -> None:
        """Build MPC prediction matrices.

        Constructs Psi, Gamma, Theta, Phi matrices for:
            y = Psi * x0 + Gamma * U
            x_N = Theta * x0 + Phi * U

        where U = [Δu_0; Δu_1; ...; Δu_{N-1}]
        """
        A = self.A_delta
        B = self.B_delta
        C = self.C_delta
        N = self.N

        n_states = self.n_states
        n_outputs = C.shape[0]

        # Psi: maps initial state to outputs
        # y_i = C @ A^i @ x0
        self.Psi = np.zeros((N * n_outputs, n_states), dtype=np.float32)
        A_power = np.eye(n_states, dtype=np.float32)
        for i in range(N):
            self.Psi[i * n_outputs : (i + 1) * n_outputs, :] = C @ A_power
            A_power = A_power @ A

        # Gamma: maps inputs to outputs (Toeplitz-like structure)
        # y_i = C @ Σ_{j=0}^{i-1} A^{i-1-j} @ B @ Δu_j
        self.Gamma = np.zeros((N * n_outputs, N), dtype=np.float32)
        A_powers = [np.eye(n_states, dtype=np.float32)]
        for i in range(N):
            A_powers.append(A_powers[-1] @ A)

        for i in range(N):
            for j in range(i + 1):
                power_idx = i - j
                self.Gamma[
                    i * n_outputs : (i + 1) * n_outputs, j
                ] = (C @ A_powers[power_idx] @ B).reshape(-1)

        # Theta: maps initial state to terminal state
        # x_N = A^N @ x0
        self.Theta = A_powers[N]

        # Phi: maps inputs to terminal state
        # x_N = Σ_{j=0}^{N-1} A^{N-1-j} @ B @ Δu_j
        self.Phi = np.zeros((n_states, N), dtype=np.float32)
        for j in range(N):
            power_idx = N - 1 - j
            self.Phi[:, j] = (A_powers[power_idx] @ B).flatten()

    def _setup_qp_problem(self) -> None:
        """Setup parameterized CVXPY problem for efficient re-solving."""
        N = self.N
        config = self.config

        # Decision variables: [Δu_0, ..., Δu_{N-1}, s_0, ..., s_{N*n_outputs-1}]
        n_outputs = self.C_delta.shape[0]
        self.du = cp.Variable(N, name="du")
        self.s = cp.Variable(N * n_outputs, name="s", nonneg=True)

        # Parameters (updated each call)
        self.chi_param = cp.Parameter(self.n_states, name="chi")
        self.u_prev_param = cp.Parameter(name="u_prev")

        # Predicted outputs without control: y_free = Psi @ chi
        y_free = self.Psi @ self.chi_param

        # Predicted outputs with control: y = y_free + Gamma @ du
        y_pred = y_free + self.Gamma @ self.du

        # Control trajectory: u = u_prev + Cum @ du
        u_traj = self.u_prev_param + self.Cum @ self.du

        # Cost function (CDC24 Equation 11 + CDC25 updates)
        # J = Q·Σs_i² + R_delta·ΣΔu_i² + R·Σu_i² + Q_f·||x_N||²
        cost = (
            config.Q * cp.sum_squares(self.s)
            + config.R_delta * cp.sum_squares(self.du)
            + config.R * cp.sum_squares(u_traj)
        )

        # Terminal cost (if Q_f > 0)
        if config.Q_f > 0:
            x_terminal = self.Theta @ self.chi_param + self.Phi @ self.du
            cost += config.Q_f * cp.sum_squares(x_terminal)

        # Constraints
        constraints = [
            # Slack must exceed positive error: s >= y - threshold
            self.s >= y_pred - config.threshold,
            # Amplitude constraints: u in [0, u_max]
            u_traj >= config.u_min,
            u_traj <= config.u_max,
            # Rate constraints: |Δu| <= delta_u_max
            self.du >= -config.delta_u_max,
            self.du <= config.delta_u_max,
        ]

        # Optional output constraints
        if config.y_min is not None:
            constraints.append(y_pred >= config.y_min)
        if config.y_max is not None:
            constraints.append(y_pred <= config.y_max)

        # Create problem
        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def reset(self, u_prev: float = 0.0) -> None:
        """Reset controller state.

        Args:
            u_prev: Initial previous control value.
        """
        self._u_prev = u_prev

    def compute_control(
        self, chi_delta: np.ndarray, u_prev: Optional[float] = None
    ) -> float:
        """Compute optimal control input.

        Args:
            chi_delta: Current velocity-form state vector.
            u_prev: Previous control input. If None, uses internal state.

        Returns:
            Optimal control input u_k.
        """
        chi_delta = np.asarray(chi_delta).flatten()

        if u_prev is None:
            u_prev = self._u_prev

        # Update parameters
        self.chi_param.value = chi_delta
        self.u_prev_param.value = u_prev

        # Solve QP
        try:
            if self.config.solver == "MOSEK":
                self.problem.solve(
                    solver=cp.MOSEK,
                    verbose=self.config.solver_verbose,
                    warm_start=self.config.solver_warm_start,
                )
            elif self.config.solver == "CLARABEL":
                self.problem.solve(
                    solver=cp.CLARABEL,
                    verbose=self.config.solver_verbose,
                    warm_start=self.config.solver_warm_start,
                )
            elif self.config.solver == "OSQP":
                self.problem.solve(
                    solver=cp.OSQP,
                    verbose=self.config.solver_verbose,
                    warm_start=self.config.solver_warm_start,
                )
            elif self.config.solver == "ECOS":
                self.problem.solve(
                    solver=cp.ECOS,
                    verbose=self.config.solver_verbose,
                    warm_start=self.config.solver_warm_start,
                )
            else:
                self.problem.solve(
                    verbose=self.config.solver_verbose,
                    warm_start=self.config.solver_warm_start,
                )
        except cp.SolverError as e:
            # Fallback on solver failure
            print(f"LinearMPC solver error: {e}. Using fallback.")
            return self._fallback_control(u_prev)

        # Check solution status
        if self.problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"LinearMPC: {self.problem.status}. Using fallback.")
            return self._fallback_control(u_prev)

        # Extract optimal Δu_0
        du_0 = self.du.value[0]

        # Compute u_k = u_{k-1} + Δu_0
        u = u_prev + du_0

        # Safety saturation
        u = np.clip(u, self.config.u_min, self.config.u_max)

        # Update internal state
        self._u_prev = u

        return float(u)

    def _fallback_control(self, u_prev: float) -> float:
        """Fallback control when QP fails.

        Args:
            u_prev: Previous control value.

        Returns:
            Safe fallback control (maintain previous, clamped to bounds).
        """
        u = np.clip(u_prev, self.config.u_min, self.config.u_max)
        self._u_prev = u
        return float(u)

    @property
    def last_solution(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get last optimal solution (du, slack) if available."""
        if self.du.value is not None and self.s.value is not None:
            return self.du.value.copy(), self.s.value.copy()
        return None


def load_velocity_form_model(
    mat_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load velocity-form state-space model from MATLAB .mat file.

    Args:
        mat_path: Path to .mat file containing A_delta, B_delta, C_delta.

    Returns:
        Tuple of (A_delta, B_delta, C_delta) arrays.
    """
    from scipy.io import loadmat

    data = loadmat(mat_path, squeeze_me=True)

    # Try different variable names
    if "A_delta" in data:
        A_delta = data["A_delta"]
        B_delta = data["B_delta"]
        C_delta = data["C_delta"]
    elif "model" in data:
        model = data["model"]
        if hasattr(model, "dtype") and model.dtype.names:
            A_delta = model["A_delta"].item()
            B_delta = model["B_delta"].item()
            C_delta = model["C_delta"].item()
        else:
            raise ValueError("Cannot parse model structure from .mat file")
    else:
        raise ValueError("Expected A_delta/B_delta/C_delta or model struct in .mat")

    return A_delta, B_delta, C_delta


def build_velocity_form(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, forgetting_factor: float = 0.999
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert standard state-space to velocity form.

    Given a discrete-time system:
        x_{k+1} = A·x_k + B·u_k
        y_k = C·x_k

    Returns the velocity-form system:
        χ^Δ_{k+1} = A^Δ·χ^Δ_k + B^Δ·Δu_k
        y_k = C^Δ·χ^Δ_k

    where χ^Δ = [x; u_{k-1}] is the augmented state.

    Args:
        A: State matrix of shape (n, n).
        B: Input matrix of shape (n, 1).
        C: Output matrix of shape (1, n).
        forgetting_factor: Factor for integrator stability (default: 0.999).
            MATLAB uses 0.999 to prevent numerical drift.
            Setting to 1.0 gives pure integration (may drift).

    Returns:
        Tuple of (A_delta, B_delta, C_delta) for velocity form.
    """
    A = np.asarray(A)
    B = np.asarray(B).reshape(-1, 1)
    C = np.asarray(C).reshape(1, -1)

    n = A.shape[0]

    # Augmented state: χ = [x; u_{k-1}]
    # Augmented dynamics:
    #   x_{k+1} = A·x_k + B·u_k = A·x_k + B·(u_{k-1} + Δu_k)
    #   u_k = λ·u_{k-1} + Δu_k  (with forgetting factor λ for stability)
    #
    # So: [x_{k+1}]   [A  B] [x_k    ]   [B]
    #     [u_k    ] = [0  λ] [u_{k-1}] + [1] Δu_k

    A_delta = np.block([[A, B], [np.zeros((1, n), dtype=np.float32), np.array([[forgetting_factor]], dtype=np.float32)]])

    B_delta = np.vstack([B, np.array([[1.0]], dtype=np.float32)])

    C_delta = np.hstack([C, np.array([[0.0]], dtype=np.float32)])

    return A_delta, B_delta, C_delta

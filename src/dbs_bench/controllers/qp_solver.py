# Canonical owner: closed-loop-dbs-bench
"""QP formulation for SCP subproblems in DCNN Tube MPC.

This module implements the **DC-Convex** QP subproblem for the SCP algorithm,
following equations (6) and (7) from CDC25.

Key Features:
- **Proper DC Constraints**: Linearizes only the concave part of each constraint,
  keeping the convex ICNN evaluation exact using CVXPY expressions.
- **Asymmetric Bounds**: Upper bound uses f1(u) exactly, lower bound uses f2(u) exactly.
- **Cached Problem**: Pre-builds the CVXPY problem for fast re-solving.
- **Tube Regularization**: Penalizes tube width to prevent artificial inflation.

CDC25 Equations (6) and (7):
    s_max >= f1(z,u) - f2(z,u^0) - J_f2·v + w_max - y^0
    s_min <= -f2(z,u) + f1(z,u^0) + J_f1·v + w_min - y^0

Simplifying (since y^0 = f1(z,u^0) - f2(z,u^0)):
    s_max >= f1(z,u) - f1(z,u^0) - J_f2·v + w_max
    s_min <= -(f2(z,u) - f2(z,u^0)) + J_f1·v + w_min

Example:
    >>> qp = QPSubproblem(N=5, config=scp_config, predictor=model)
    >>> solution = qp.solve(
    ...     z_k=state,
    ...     y_nominal=y_nom,
    ...     u_nominal=u_nom,
    ...     u_prev=0.01,
    ...     jacobians_f1=jac_f1,
    ...     jacobians_f2=jac_f2,
    ...     f1_nominal=f1_nom,
    ...     f2_nominal=f2_nom,
    ...     W_bounds=W,
    ... )
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import cvxpy as cp

if TYPE_CHECKING:
    from .scp_config import SCPConfig
    from .dcnn_models import MultiStepDCNN


@dataclass
class QPSolution:
    """Result of QP subproblem solve.

    Attributes:
        u_optimal: Optimal control sequence of shape (N_ctrl,) for control horizon,
            or (N_pred,) if extended (with frozen tail).
        s_max_optimal: Optimal upper perturbation bounds of shape (N_pred,).
        s_min_optimal: Optimal lower perturbation bounds of shape (N_pred,).
        cost: Optimal cost value.
        status: Solver status string.
        solve_time: Time to solve in seconds.
        is_feasible: True if problem was solved successfully.
    """

    u_optimal: np.ndarray
    s_max_optimal: np.ndarray
    s_min_optimal: np.ndarray
    cost: float
    status: str
    solve_time: float
    is_feasible: bool


class QPSubproblem:
    """DC-Convex QP solver for SCP iterations (CDC25 formulation).

    This class implements the proper DC decomposition where convex parts
    are evaluated exactly using CVXPY expressions, and only the concave
    parts are linearized.

    The problem must be rebuilt when z_k changes because the ICNN expressions
    depend on z_k. However, within an SCP iteration (where z_k is fixed),
    parameter updates allow fast re-solving.
    """

    def __init__(
        self,
        N: int,
        config: "SCPConfig",
        predictor: "MultiStepDCNN" = None,
        weights_f1: List[List[np.ndarray]] = None,
        weights_f2: List[List[np.ndarray]] = None,
    ):
        """Initialize QP subproblem.

        Args:
            N: Prediction horizon (N_pred). For backward compatibility, this is
                the full prediction horizon. Control horizon is read from config.
            config: SCP configuration (contains control_horizon and prediction_horizon).
            predictor: MultiStepDCNN model (extracts weights automatically).
                Should have control_horizon networks.
            weights_f1: Pre-extracted weights for f1 networks (alternative to predictor).
            weights_f2: Pre-extracted weights for f2 networks (alternative to predictor).
        """
        self.N = N  # N_pred: prediction horizon (for backward compatibility)
        self.N_pred = N
        self.N_ctrl = getattr(config, 'control_horizon', N)  # Control horizon
        self.config = config

        # Performance tracking
        self.build_count = 0
        self.solve_count = 0

        # Check if we're using extended horizon (move blocking)
        self.uses_extended_horizon = self.N_pred > self.N_ctrl

        # Extract and store ICNN weights (only for control horizon networks)
        if predictor is not None:
            from experiments.dcnn_mpc.core.jacobian import extract_weights_from_convex_nn
            # Predictor only has N_ctrl networks
            n_networks = min(self.N_ctrl, len(predictor.networks))
            self.weights_f1 = [
                extract_weights_from_convex_nn(predictor.networks[i].f1)
                for i in range(n_networks)
            ]
            self.weights_f2 = [
                extract_weights_from_convex_nn(predictor.networks[i].f2)
                for i in range(n_networks)
            ]
            self.n_state = predictor.n_state
        elif weights_f1 is not None and weights_f2 is not None:
            self.weights_f1 = weights_f1
            self.weights_f2 = weights_f2
            # Infer n_state from first layer weight shape
            self.n_state = weights_f1[0][0].shape[1] - 1  # input_dim - 1 control
        else:
            # Fallback for backward compatibility (linearized-only mode)
            self.weights_f1 = None
            self.weights_f2 = None
            self.n_state = getattr(config, 'n_state_y', 15) + getattr(config, 'n_state_u', 1)

        # Check if we can use DC mode or must fall back to linearized mode
        self.use_dc_mode = self.weights_f1 is not None

        # Problem will be built on first solve (needs z_k)
        self._problem = None
        self._current_z_k = None

    def solve(
        self,
        z_k: np.ndarray,
        y_nominal: np.ndarray,
        u_nominal: np.ndarray,
        u_prev: float,
        jacobians_f1: List[np.ndarray],
        jacobians_f2: List[np.ndarray],
        W_bounds: np.ndarray,
        f1_nominal: np.ndarray = None,
        f2_nominal: np.ndarray = None,
        device: str = "cpu",
        force_rebuild: bool = False,
    ) -> QPSolution:
        """Solve the QP subproblem.

        Args:
            z_k: Current state vector of shape (n_state,).
            y_nominal: Nominal predictions y^0 of shape (N_pred,).
            u_nominal: Nominal control u^0 of shape (N_ctrl,) for control horizon.
            u_prev: Previous control u_{k-1}.
            jacobians_f1: List of N_ctrl Jacobians for f1, each shape (1, i+1).
            jacobians_f2: List of N_ctrl Jacobians for f2, each shape (1, i+1).
            W_bounds: Disturbance bounds of shape (N_pred, 2).
            f1_nominal: f1 evaluations at nominal, shape (N_ctrl,). Required for DC mode.
            f2_nominal: f2 evaluations at nominal, shape (N_ctrl,). Required for DC mode.
            device: Unused (for API compatibility).
            force_rebuild: If True, force problem rebuild even if z_k unchanged.
                Use on first SCP iteration of a new control step.

        Returns:
            QPSolution object with u_optimal extended to N_pred (frozen tail).
        """
        start_time = time.time()

        # Rebuild problem if z_k changed (ICNN expressions depend on z_k)
        # Within SCP iterations at the same timestep, z_k is fixed, so skip rebuild
        needs_rebuild = (
            self._problem is None
            or force_rebuild
            or not np.allclose(z_k, self._current_z_k, rtol=1e-10)
        )
        if needs_rebuild:
            self._build_problem(z_k)

        # Update Parameters
        self.param_y_nominal.value = y_nominal  # Shape (N_pred,)
        self.param_u_nominal.value = u_nominal[:self.N_ctrl]  # Shape (N_ctrl,)
        self.param_u_prev.value = u_prev
        self.param_w_min.value = W_bounds[:, 0]  # Shape (N_pred,)
        self.param_w_max.value = W_bounds[:, 1]

        # Update Jacobian and nominal evaluation parameters (only for control horizon)
        for i in range(self.N_ctrl):
            self.params_J_f1[i].value = np.asarray(jacobians_f1[i]).reshape(1, i + 1)
            self.params_J_f2[i].value = np.asarray(jacobians_f2[i]).reshape(1, i + 1)

            if self.use_dc_mode:
                if f1_nominal is None or f2_nominal is None:
                    raise ValueError("f1_nominal and f2_nominal required for DC mode")
                self.params_f1_nom[i].value = f1_nominal[i]
                self.params_f2_nom[i].value = f2_nominal[i]
            else:
                # Linearized mode: set combined Jacobian
                J_diff = jacobians_f1[i] - jacobians_f2[i]
                self.params_J[i].value = np.asarray(J_diff).reshape(1, i + 1)

        # Update extended horizon parameters (frozen nominal values)
        if self.uses_extended_horizon:
            for i in range(self.N_ctrl, self.N_pred):
                # Extended steps use frozen values from control horizon boundary
                self.params_y_nom_ext[i - self.N_ctrl].value = y_nominal[i]

        # Solve with hard constraints first, then soft if needed
        HARD_PENALTY = 1e9
        SOFT_PENALTY = 1e4

        self.param_slack_penalty.value = HARD_PENALTY

        try:
            self._problem.solve(
                solver=getattr(cp, self.config.solver, cp.CLARABEL),
                warm_start=True,
                verbose=self.config.solver_verbose
            )

            total_slack = np.sum(self.slack_max.value) + np.sum(self.slack_min.value)

            if self._problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] or total_slack > 1e-3:
                if self.config.constraint_softening:
                    self.param_slack_penalty.value = SOFT_PENALTY
                    self._problem.solve(
                        solver=getattr(cp, self.config.solver, cp.CLARABEL),
                        warm_start=True,
                        verbose=self.config.solver_verbose
                    )

            solve_time = time.time() - start_time
            self.solve_count += 1

            is_feasible = self._problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]

            if is_feasible:
                # Extend u_optimal to N_pred by repeating the last control value
                u_ctrl = self.u.value  # Shape (N_ctrl,)
                if self.uses_extended_horizon:
                    u_extended = np.zeros(self.N_pred, dtype=np.float32)
                    u_extended[:self.N_ctrl] = u_ctrl
                    u_extended[self.N_ctrl:] = u_ctrl[-1]  # Frozen tail
                else:
                    u_extended = np.asarray(u_ctrl, dtype=np.float32)

                return QPSolution(
                    u_optimal=u_extended,
                    s_max_optimal=np.asarray(self.s_max.value, dtype=np.float32),
                    s_min_optimal=np.asarray(self.s_min.value, dtype=np.float32),
                    cost=self._problem.value,
                    status=self._problem.status,
                    solve_time=solve_time,
                    is_feasible=True
                )
            else:
                return self._failure_result(u_nominal, solve_time, self._problem.status)

        except Exception as e:
            solve_time = time.time() - start_time
            return self._failure_result(u_nominal, solve_time, f"ERROR: {str(e)}")

    def _failure_result(self, u_nominal, solve_time, status):
        """Helper to return uniform failure result."""
        # Extend u_nominal to N_pred if needed
        if len(u_nominal) < self.N_pred:
            u_extended = np.zeros(self.N_pred, dtype=np.float32)
            u_extended[:len(u_nominal)] = u_nominal
            u_extended[len(u_nominal):] = u_nominal[-1]
        else:
            u_extended = np.asarray(u_nominal[:self.N_pred], dtype=np.float32)

        return QPSolution(
            u_optimal=u_extended,
            s_max_optimal=np.zeros(self.N_pred, dtype=np.float32),
            s_min_optimal=np.zeros(self.N_pred, dtype=np.float32),
            cost=np.inf,
            status=status,
            solve_time=solve_time,
            is_feasible=False
        )

    def _build_problem(self, z_k: np.ndarray):
        """Construct the CVXPY problem graph.

        This must be called when z_k changes, as the ICNN expressions
        depend on z_k values.

        Extended Horizon (Move Blocking):
        - Decision variable u has shape (N_ctrl,)
        - Tube variables s_max, s_min have shape (N_pred,)
        - For steps 0 to N_ctrl-1: full DC constraints
        - For steps N_ctrl to N_pred-1: frozen control at u[N_ctrl-1]

        Args:
            z_k: Current state vector of shape (n_state,).
        """
        self.build_count += 1
        self._current_z_k = z_k.copy()

        # --- Define Parameters ---
        self.param_u_prev = cp.Parameter(name="u_prev")
        self.param_u_nominal = cp.Parameter(self.N_ctrl, name="u_nominal")  # Control horizon
        self.param_y_nominal = cp.Parameter(self.N_pred, name="y_nominal")  # Full prediction
        self.param_w_max = cp.Parameter(self.N_pred, name="w_max")
        self.param_w_min = cp.Parameter(self.N_pred, name="w_min")

        # Parameters for extended horizon nominal y values (for frozen steps)
        if self.uses_extended_horizon:
            self.params_y_nom_ext = [
                cp.Parameter(name=f"y_nom_ext_{i}")
                for i in range(self.N_pred - self.N_ctrl)
            ]
        else:
            self.params_y_nom_ext = []

        # --- Define Variables ---
        # Decision variable: only N_ctrl free controls
        self.u = cp.Variable(self.N_ctrl, name="u")
        # Tube variables: full prediction horizon
        self.s_max = cp.Variable(self.N_pred, name="s_max")
        self.s_min = cp.Variable(self.N_pred, name="s_min")

        # Slack variables for soft constraints (full horizon)
        self.slack_max = cp.Variable(self.N_pred, nonneg=True, name="slack_max")
        self.slack_min = cp.Variable(self.N_pred, nonneg=True, name="slack_min")
        self.param_slack_penalty = cp.Parameter(nonneg=True, name="slack_penalty")

        constraints = []

        # 1. Input Constraints (only for control horizon)
        constraints.append(self.u >= self.config.u_min)
        constraints.append(self.u <= self.config.u_max)

        # 2. Rate Constraints (only for control horizon)
        constraints.append(self.u[0] - self.param_u_prev <= self.config.delta_u_max)
        constraints.append(self.u[0] - self.param_u_prev >= -self.config.delta_u_max)
        for i in range(1, self.N_ctrl):
            constraints.append(self.u[i] - self.u[i-1] <= self.config.delta_u_max)
            constraints.append(self.u[i] - self.u[i-1] >= -self.config.delta_u_max)

        # 3. Robust Tube Constraints
        # 3a. Control horizon: DC formulation from CDC25
        if self.use_dc_mode:
            constraints.extend(self._build_dc_constraints(z_k))
        else:
            constraints.extend(self._build_linearized_constraints())

        # 3b. Extended horizon: frozen control constraints
        if self.uses_extended_horizon:
            constraints.extend(self._build_extended_constraints())

        # 4. Tube Consistency and Positivity
        constraints.append(self.s_max >= 0)
        constraints.append(self.s_min <= 0)
        constraints.append(self.s_max >= self.s_min)

        # 5. Output Constraints (Softened) - full horizon
        if self.config.y_max is not None:
            constraints.append(
                self.param_y_nominal + self.s_max <= self.config.y_max + self.slack_max
            )
        if self.config.y_min is not None:
            constraints.append(
                self.param_y_nominal + self.s_min >= self.config.y_min - self.slack_min
            )

        # --- Objective ---
        Q = self.config.Q
        R = self.config.R
        R_delta = getattr(self.config, "R_delta", 0.0)  # Rate penalty (CDC24 consistency)
        beta_0 = self.config.beta_0
        gamma = getattr(self.config, "tube_weight", 0.0)

        # Tracking cost over full prediction horizon
        tracking_error = self.param_y_nominal + self.s_max - beta_0
        tracking_cost = Q * cp.sum_squares(cp.pos(tracking_error))

        # Control cost only over control horizon (u has shape N_ctrl)
        control_cost = R * cp.sum_squares(self.u)

        # For extended horizon: add cost for frozen control repeated
        if self.uses_extended_horizon:
            n_extended = self.N_pred - self.N_ctrl
            # u[-1] is repeated n_extended times, so add n_extended * R * u[-1]^2
            control_cost += R * n_extended * cp.square(self.u[-1])

        # Rate penalty: penalize changes in control (CDC24 Eq. 11)
        # Only within control horizon (no rate penalty for frozen tail)
        if R_delta > 0:
            delta_u = cp.diff(self.u)  # u[1:] - u[:-1] within N_ctrl
            delta_u_0 = self.u[0] - self.param_u_prev  # First change from previous control
            rate_cost = R_delta * (cp.sum_squares(delta_u_0) + cp.sum_squares(delta_u))
        else:
            rate_cost = 0

        # PE excitation reward: -pe_gamma * sum(delta_u^2)
        pe_gamma_val = getattr(self.config, "pe_gamma", 0.0)
        if pe_gamma_val > 0:
            delta_u_pe = cp.diff(self.u)
            delta_u_0_pe = self.u[0] - self.param_u_prev
            pe_cost = -pe_gamma_val * (cp.sum_squares(delta_u_0_pe) + cp.sum_squares(delta_u_pe))
        else:
            pe_cost = 0

        tube_cost = gamma * cp.sum_squares(self.s_max - self.s_min)
        slack_cost = self.param_slack_penalty * (cp.sum(self.slack_max) + cp.sum(self.slack_min))

        objective = cp.Minimize(tracking_cost + control_cost + rate_cost + pe_cost + tube_cost + slack_cost)

        self._problem = cp.Problem(objective, constraints)

    def _build_dc_constraints(self, z_k: np.ndarray) -> List:
        """Build DC constraints using CVXPY ICNN expressions for control horizon.

        Implements CDC25 equations (6) and (7):
            s_max >= f1(z,u) - f1_nom - J_f2·v + w_max
            s_min <= -(f2(z,u) - f2_nom) + J_f1·v + w_min

        Where:
            - f1(z,u) is evaluated exactly using CVXPY (convex in u)
            - f2(z,u) is evaluated exactly using CVXPY (convex in u)
            - J_f1, J_f2 are Jacobians evaluated at u_nominal
            - v = u - u_nominal

        For CVXPY to verify DCP compliance, the hidden/output layer weights
        are declared as non-negative Parameters.

        Note: Only builds constraints for steps 0 to N_ctrl-1 (control horizon).

        Args:
            z_k: Current state vector.

        Returns:
            List of CVXPY constraints.
        """
        from experiments.dcnn_mpc.core.jacobian import forward_from_weights_cvxpy, build_icnn_cvxpy_params

        constraints = []

        # Parameters for nominal evaluations and Jacobians (control horizon only)
        self.params_f1_nom = [cp.Parameter(name=f"f1_nom_{i}") for i in range(self.N_ctrl)]
        self.params_f2_nom = [cp.Parameter(name=f"f2_nom_{i}") for i in range(self.N_ctrl)]
        self.params_J_f1 = [cp.Parameter((1, i + 1), name=f"J_f1_{i}") for i in range(self.N_ctrl)]
        self.params_J_f2 = [cp.Parameter((1, i + 1), name=f"J_f2_{i}") for i in range(self.N_ctrl)]

        # Convert z_k to parameter (fixed during this problem instance)
        z_k_const = z_k.astype(np.float64)

        # Build non-negative Parameters for ICNN weights (for DCP compliance)
        # These tell CVXPY that hidden/output layer weights are non-negative
        self._nonneg_params_f1 = []
        self._nonneg_params_f2 = []

        for i in range(self.N_ctrl):
            n_u = i + 1

            # Control perturbation: v = u[:n_u] - u_nominal[:n_u]
            v = self.u[:n_u] - self.param_u_nominal[:n_u]

            # Create z_k parameter for this step
            z_k_param = cp.Parameter(self.n_state, name=f"z_k_{i}")
            z_k_param.value = z_k_const

            # Build non-negative parameters for f1 and f2 weights
            nonneg_f1 = build_icnn_cvxpy_params(self.weights_f1[i], f"f1_{i}")
            nonneg_f2 = build_icnn_cvxpy_params(self.weights_f2[i], f"f2_{i}")
            self._nonneg_params_f1.append(nonneg_f1)
            self._nonneg_params_f2.append(nonneg_f2)

            # Build CVXPY expressions for f1(z_k, u) and f2(z_k, u)
            f1_expr = forward_from_weights_cvxpy(
                z_k_param, self.u[:n_u], self.weights_f1[i], nonneg_f1
            )
            f2_expr = forward_from_weights_cvxpy(
                z_k_param, self.u[:n_u], self.weights_f2[i], nonneg_f2
            )

            # Upper bound (eq. 6): s_max >= f1(u) - f1_nom - J_f2·v + w_max
            # f1 is convex (kept), -f2 is concave (linearized as -f2_nom - J_f2·v)
            upper_bound = f1_expr - self.params_f1_nom[i] - self.params_J_f2[i] @ v + self.param_w_max[i]
            constraints.append(self.s_max[i] >= upper_bound)

            # Lower bound (eq. 7): s_min <= -(f2(u) - f2_nom) + J_f1·v + w_min
            # -f2 is concave (kept as -(f2(u) - f2_nom)), f1 is convex (linearized)
            lower_bound = -(f2_expr - self.params_f2_nom[i]) + self.params_J_f1[i] @ v + self.param_w_min[i]
            constraints.append(self.s_min[i] <= lower_bound)

        return constraints

    def _build_linearized_constraints(self) -> List:
        """Build fully linearized constraints (fallback mode) for control horizon.

        This is the original implementation that linearizes the entire model.
        Used when ICNN weights are not available.

        Note: Only builds constraints for steps 0 to N_ctrl-1 (control horizon).

        Returns:
            List of CVXPY constraints.
        """
        constraints = []

        # Combined Jacobian parameters (J = J_f1 - J_f2) for control horizon only
        self.params_J = [cp.Parameter((1, i + 1), name=f"J_{i}") for i in range(self.N_ctrl)]

        # Also define the split parameters for API compatibility
        self.params_J_f1 = self.params_J  # Will be set to J_f1 - J_f2
        self.params_J_f2 = [cp.Parameter((1, i + 1), name=f"J_f2_{i}") for i in range(self.N_ctrl)]
        self.params_f1_nom = [cp.Parameter(name=f"f1_nom_{i}") for i in range(self.N_ctrl)]
        self.params_f2_nom = [cp.Parameter(name=f"f2_nom_{i}") for i in range(self.N_ctrl)]

        for i in range(self.N_ctrl):
            n_u = i + 1
            v = self.u[:n_u] - self.param_u_nominal[:n_u]

            # Fully linearized: y ≈ y_nom + J·v, so perturbation ≈ J·v
            linearized_delta = self.params_J[i] @ v

            constraints.append(self.s_max[i] >= linearized_delta + self.param_w_max[i])
            constraints.append(self.s_min[i] <= linearized_delta + self.param_w_min[i])

        return constraints

    def _build_extended_constraints(self) -> List:
        """Build constraints for extended horizon (frozen control) steps.

        For steps N_ctrl to N_pred-1, the control is frozen at u[N_ctrl-1].
        Since control is frozen, there's no optimization variable sensitivity
        for these steps. The constraints become:
            s_max[i] >= w_max[i]  (no control perturbation effect)
            s_min[i] <= w_min[i]

        This implements the "frozen bounds" approach where:
        - Prediction is frozen at y_nominal[N_ctrl-1]
        - Bounds are frozen at W_bounds[N_ctrl-1] (set in config)
        - No decision variable affects these steps

        Returns:
            List of CVXPY constraints for extended horizon.
        """
        constraints = []

        for i in range(self.N_ctrl, self.N_pred):
            # For frozen steps: perturbation bounds are just the disturbance bounds
            # since control is frozen (no control perturbation term)
            constraints.append(self.s_max[i] >= self.param_w_max[i])
            constraints.append(self.s_min[i] <= self.param_w_min[i])

        return constraints


def create_qp_subproblem(
    config: "SCPConfig",
    predictor: "MultiStepDCNN" = None,
) -> QPSubproblem:
    """Factory function to create QP subproblem.

    Args:
        config: SCP configuration (contains both control_horizon and prediction_horizon).
        predictor: MultiStepDCNN model for DC mode. If None, uses linearized mode.
            Should have control_horizon networks.

    Returns:
        Configured QPSubproblem instance.
    """
    return QPSubproblem(
        N=config.prediction_horizon,  # Pass prediction horizon
        config=config,
        predictor=predictor,
    )

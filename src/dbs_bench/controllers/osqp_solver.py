# Canonical owner: closed-loop-dbs-bench
"""OSQP solver backend for DC-MPC.

Uses the QPMatrixBuilder from direct_qp_solver.py and calls OSQP (ADMM-based
first-order solver) directly. Key advantages over CLARABEL:
- Warm-starting between SCP iterations (primal + dual)
- Persistent solver instance with update() instead of rebuild
- Potentially faster for this QP size (~360 vars, ~375 constraints)

OSQP format:
    min  0.5 x'Px + q'x
    s.t. l <= Ax <= u

Equalities: l = u = b_eq
Inequalities: l = -inf, u = h
"""
from __future__ import annotations

import time
from typing import List

import numpy as np
from scipy import sparse

from dbs_bench.controllers.direct_qp_solver import (
    DirectQPSolution,
    QPMatrixBuilder,
    _create_builder,
)


class OSQPSolver:
    """OSQP solver for DC-MPC with warm-starting support.

    Maintains a persistent OSQP solver instance that is set up once and
    updated between SCP iterations for maximum speed.
    """

    def __init__(self, builder: QPMatrixBuilder):
        self._builder = builder
        self._setup_done = False
        self._solver = None
        self._prev_x = None
        self._prev_y = None

        # Expose variable indices from builder
        self.N = builder.N
        self.idx_u = builder.idx_u
        self.idx_s_max = builder.idx_s_max
        self.idx_s_min = builder.idx_s_min

    def setup(self):
        """Build the QP structure and create persistent OSQP solver."""
        import osqp

        start = time.perf_counter()

        if not self._builder._setup_done:
            self._builder.setup()

        b = self._builder

        # Build combined constraint matrix: [A_eq; A_ineq]
        # OSQP format: l <= Ax <= u
        if b.A_eq is not None:
            self._A_combined = sparse.vstack([b.A_eq, b.A_ineq]).tocsc()
            n_eq = b.n_eq
        else:
            self._A_combined = b.A_ineq.tocsc()
            n_eq = 0
        n_ineq = b.n_ineq

        # Build l, u vectors
        # Equalities: l = u = b_eq
        # Inequalities: -inf <= Ax <= h
        if b.b_eq_template is not None:
            l_eq = b.b_eq_template.copy()
            u_eq = b.b_eq_template.copy()
        else:
            l_eq = np.array([])
            u_eq = np.array([])

        l_ineq = np.full(n_ineq, -np.inf)
        u_ineq = b.h_template.copy()

        self._l = np.concatenate([l_eq, l_ineq])
        self._u = np.concatenate([u_eq, u_ineq])
        self._n_eq = n_eq

        # Regularize zero-diagonal entries of P to make it positive definite.
        # Use a fraction of the minimum nonzero diagonal entry so the
        # regularization adapts to the actual cost scales (Q, R, tube_weight).
        # This keeps the condition number reasonable for ADMM convergence.
        P = b.P.copy()
        P_diag = P.diagonal()
        nonzero_diag = P_diag[P_diag > 0]
        if len(nonzero_diag) > 0:
            reg_value = float(nonzero_diag.min()) * 1e-4
        else:
            reg_value = 1e-6
        reg_value = max(reg_value, 1e-7)  # Floor to avoid numerical issues

        reg_diag = np.array([reg_value if P_diag[i] == 0 else 0.0
                             for i in range(b.n_vars)])
        P_reg = P + sparse.diags(reg_diag)

        # OSQP requires upper-triangular P
        P_upper = sparse.triu(P_reg).tocsc()

        # Initial q (will be updated)
        q = np.zeros(b.n_vars)

        # Store sparsity pattern info for the combined constraint matrix.
        # OSQP update(Ax=...) requires values in the same order as the
        # CSC data array. We need to map from our COO-based updates to this.
        self._A_csc_data_len = len(self._A_combined.data)

        # Create solver.
        # The QP lives inside an SCP loop (delta_J_min ~1e-5), so
        # tolerances must be tight enough that objective noise doesn't
        # prevent SCP convergence. Cap max_iter to keep solve times
        # bounded; warm-starting across SCP iterations helps convergence.
        self._solver = osqp.OSQP()
        self._solver.setup(
            P=P_upper,
            q=q,
            A=self._A_combined,
            l=self._l,
            u=self._u,
            eps_abs=1e-4,
            eps_rel=1e-4,
            max_iter=10000,
            warm_starting=True,
            polishing=True,
            adaptive_rho=True,
            adaptive_rho_interval=25,
            rho=1.0,
            sigma=1e-6,
            scaling=100,
            verbose=False,
        )

        self._setup_done = True
        self._setup_time = time.perf_counter() - start

    def solve(
        self,
        z_k: np.ndarray,
        u_prev: float,
        u_nominal: np.ndarray,
        y_nominal: np.ndarray,
        jacobians_f1: List[np.ndarray],
        jacobians_f2: List[np.ndarray],
        f1_nominal: np.ndarray,
        f2_nominal: np.ndarray,
        W_bounds: np.ndarray,
        linear_jacobians: List[np.ndarray] = None,
    ) -> DirectQPSolution:
        """Solve the QP with updated parameters."""
        if not self._setup_done:
            self.setup()

        start = time.perf_counter()

        b = self._builder

        # Update linear cost
        q = b.compute_linear_cost(y_nominal, u_prev)

        # Update constraints
        b_eq, h, A_ineq = b.update_constraints(
            z_k, u_prev, u_nominal, y_nominal,
            jacobians_f1, jacobians_f2, f1_nominal, f2_nominal, W_bounds,
            linear_jacobians=linear_jacobians,
        )

        # Build updated combined constraint matrix
        if b.A_eq is not None:
            A_new = sparse.vstack([b.A_eq, A_ineq]).tocsc()
        else:
            A_new = A_ineq.tocsc()

        # Build updated l, u vectors
        if b_eq is not None:
            l_eq = b_eq.copy()
            u_eq = b_eq.copy()
        else:
            l_eq = np.array([])
            u_eq = np.array([])

        l_ineq = np.full(b.n_ineq, -np.inf)
        u_ineq = h

        l_new = np.concatenate([l_eq, l_ineq])
        u_new = np.concatenate([u_eq, u_ineq])

        update_time = time.perf_counter() - start

        # Update OSQP solver
        self._solver.update(q=q, l=l_new, u=u_new, Ax=A_new.data)

        # Warm-start from previous solution
        if self._prev_x is not None:
            self._solver.warm_start(x=self._prev_x, y=self._prev_y)

        # Solve
        solve_start = time.perf_counter()
        result = self._solver.solve()
        solve_time_inner = time.perf_counter() - solve_start

        total_time = time.perf_counter() - start

        # Map OSQP status
        # OSQP v1.0+: status_val 1 = solved, 2 = solved_inaccurate
        is_feasible = result.info.status_val in (1, 2)

        if is_feasible:
            x = result.x
            # Store for warm-starting
            self._prev_x = x.copy()
            self._prev_y = result.y.copy()

            return DirectQPSolution(
                u_optimal=np.asarray(x[self.idx_u], dtype=np.float32),
                s_max_optimal=np.asarray(x[self.idx_s_max], dtype=np.float32),
                s_min_optimal=np.asarray(x[self.idx_s_min], dtype=np.float32),
                cost=result.info.obj_val,
                status=result.info.status,
                solve_time=total_time,
                is_feasible=True,
                setup_time=getattr(self, '_setup_time', 0),
                update_time=update_time,
            )
        else:
            return DirectQPSolution(
                u_optimal=np.asarray(u_nominal, dtype=np.float32),
                s_max_optimal=np.zeros(self.N, dtype=np.float32),
                s_min_optimal=np.zeros(self.N, dtype=np.float32),
                cost=np.inf,
                status=result.info.status,
                solve_time=total_time,
                is_feasible=False,
                update_time=update_time,
            )


def create_osqp_solver(predictor, config):
    """Factory function to create OSQPSolver from predictor and config.

    Returns:
        OSQPSolver instance, or None if extended horizon is detected.
    """
    builder = _create_builder(predictor, config)
    if builder is None:
        return None
    return OSQPSolver(builder)

# Canonical owner: closed-loop-dbs-bench
"""ARX (AutoRegressive with eXogenous input) model for direct multi-step prediction.

This module implements a linear ARX model that operates in the same framework
as the DCNN multi-step predictor. Each step k has its own coefficient vector,
fitted independently via least squares (direct multi-step, not recursive).

The ARX model captures bulk linear dynamics, while a residual DC-NN captures
the nonlinear remainder. Together they form the Residual DCNN-MPC.

All operations are in log space (matching the DCNN training convention).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class ARXModel:
    """Direct multi-step ARX model with per-step coefficient vectors.

    For each step k in {1, ..., horizon}, the model predicts:
        y_hat[k] = W_z[k] @ z + W_u[k] @ u[:k] + b[k]

    where z is the state vector (past y and u values) and u[:k] are the
    first k future control inputs.

    Attributes:
        horizon: Number of prediction steps.
        n_state: State dimension (n_state_y + n_state_u).
        coefficients: Per-step fitted parameters.
    """

    def __init__(self, n_state: int, horizon: int = 5):
        self.n_state = n_state
        self.horizon = horizon
        # Per-step coefficients: {k: {"W": array, "b": float}}
        self.coefficients: Dict[int, Dict] = {}
        self._is_fitted = False

    def fit(
        self,
        x: np.ndarray,
        u: np.ndarray,
        y: np.ndarray,
        k: int,
        regularization: float = 0.0,
    ) -> Dict[str, float]:
        """Fit ARX model for step k using least squares.

        Args:
            x: State vectors of shape (N, n_state).
            u: Control sequences of shape (N, horizon).
            y: Target outputs of shape (N, horizon).
            k: Step index (1-indexed, 1..horizon).
            regularization: L2 regularization strength (ridge regression).

        Returns:
            Dict with fitting metrics (r2, mse, n_samples).
        """
        # Build regressor: [z, u[:k]]
        regressor = np.hstack([x, u[:, :k]])
        target = y[:, k - 1]

        n_features = regressor.shape[1]

        # Add bias column
        regressor_bias = np.hstack([regressor, np.ones((len(regressor), 1))])

        if regularization > 0:
            # Ridge regression: (X^T X + lambda I)^{-1} X^T y
            A = regressor_bias.T @ regressor_bias
            A += regularization * np.eye(A.shape[0])
            A[-1, -1] -= regularization  # Don't regularize bias
            b = regressor_bias.T @ target
            w = np.linalg.solve(A, b)
        else:
            # Ordinary least squares
            w, residuals, rank, sv = np.linalg.lstsq(regressor_bias, target, rcond=None)

        # Store coefficients
        self.coefficients[k] = {
            "W": w[:n_features].copy(),
            "b": float(w[n_features]),
        }

        # Compute metrics
        y_pred = regressor_bias @ w
        ss_res = np.sum((target - y_pred) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-10)
        mse = np.mean((target - y_pred) ** 2)

        if len(self.coefficients) == self.horizon:
            self._is_fitted = True

        return {"r2": float(r2), "mse": float(mse), "n_samples": len(target)}

    def fit_all(
        self,
        x: np.ndarray,
        u: np.ndarray,
        y: np.ndarray,
        regularization: float = 0.0,
    ) -> List[Dict[str, float]]:
        """Fit ARX model for all steps k=1..horizon.

        Args:
            x: State vectors of shape (N, n_state).
            u: Control sequences of shape (N, horizon).
            y: Target outputs of shape (N, horizon).
            regularization: L2 regularization strength.

        Returns:
            List of per-step fitting metrics.
        """
        metrics = []
        for k in range(1, self.horizon + 1):
            m = self.fit(x, u, y, k, regularization=regularization)
            metrics.append(m)
        return metrics

    def predict(self, z: np.ndarray, u: np.ndarray, k: int) -> np.ndarray:
        """Predict output at step k.

        Args:
            z: State vector of shape (n_state,) or (N, n_state).
            u: Control sequence of shape (horizon,) or (N, horizon).
            k: Step index (1-indexed).

        Returns:
            Predictions of shape () for single sample or (N,) for batch.
        """
        if k not in self.coefficients:
            raise ValueError(f"Step k={k} not fitted. Fitted steps: {list(self.coefficients.keys())}")

        coeff = self.coefficients[k]
        W = coeff["W"]
        b = coeff["b"]

        single = z.ndim == 1
        if single:
            z = z.reshape(1, -1)
            u = u.reshape(1, -1)

        regressor = np.hstack([z, u[:, :k]])
        result = regressor @ W + b

        return result[0] if single else result

    def predict_all(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Predict outputs for all steps k=1..horizon.

        Args:
            z: State vector of shape (n_state,) or (N, n_state).
            u: Control sequence of shape (horizon,) or (N, horizon).

        Returns:
            Predictions of shape (horizon,) or (N, horizon).
        """
        single = z.ndim == 1
        if single:
            z = z.reshape(1, -1)
            u = u.reshape(1, -1)

        N = z.shape[0]
        preds = np.zeros((N, self.horizon))
        for k in range(1, self.horizon + 1):
            preds[:, k - 1] = self.predict(z, u, k)

        return preds[0] if single else preds

    def get_u_jacobian(self, k: int) -> np.ndarray:
        """Get the u-coefficients (Jacobian w.r.t. u) for step k.

        The ARX model is linear in u, so the Jacobian is constant
        (independent of state). This returns the portion of W
        corresponding to u[:k].

        Args:
            k: Step index (1-indexed).

        Returns:
            Jacobian of shape (1, k) — the u-coefficients.
        """
        if k not in self.coefficients:
            raise ValueError(f"Step k={k} not fitted")

        W = self.coefficients[k]["W"]
        # W has shape (n_state + k,). The last k entries are u-coefficients.
        u_coeffs = W[self.n_state:self.n_state + k]
        return u_coeffs.reshape(1, k)

    def cross_validate_order(
        self,
        x_full: np.ndarray,
        u_full: np.ndarray,
        y_full: np.ndarray,
        candidates: List[Tuple[int, int]],
        n_folds: int = 5,
        regularization: float = 0.0,
    ) -> Tuple[Tuple[int, int], Dict]:
        """Cross-validate ARX order (n_state_y, n_state_u) candidates.

        Args:
            x_full: Full state vectors (will be truncated per candidate).
            u_full: Full control sequences of shape (N, horizon).
            y_full: Target outputs of shape (N, horizon).
            candidates: List of (n_state_y, n_state_u) tuples to try.
            n_folds: Number of CV folds.
            regularization: L2 regularization strength.

        Returns:
            Tuple of (best_order, results_dict) where results_dict maps
            each candidate to its avg validation MSE.
        """
        N = len(x_full)
        fold_size = N // n_folds
        results = {}

        for n_y, n_u in candidates:
            n_state_candidate = n_y + n_u
            if n_state_candidate > x_full.shape[1]:
                continue

            # Truncate state to candidate dimensions
            x_cand = x_full[:, :n_state_candidate]

            fold_mses = []
            for fold in range(n_folds):
                val_start = fold * fold_size
                val_end = val_start + fold_size

                x_train = np.vstack([x_cand[:val_start], x_cand[val_end:]])
                u_train = np.vstack([u_full[:val_start], u_full[val_end:]])
                y_train = np.vstack([y_full[:val_start], y_full[val_end:]])
                x_val = x_cand[val_start:val_end]
                u_val = u_full[val_start:val_end]
                y_val = y_full[val_start:val_end]

                arx = ARXModel(n_state=n_state_candidate, horizon=self.horizon)
                arx.fit_all(x_train, u_train, y_train, regularization=regularization)

                # Evaluate on validation set
                mses = []
                for k in range(1, self.horizon + 1):
                    y_pred = arx.predict(x_val, u_val, k)
                    mse_k = np.mean((y_val[:, k - 1] - y_pred) ** 2)
                    mses.append(mse_k)
                fold_mses.append(np.mean(mses))

            results[(n_y, n_u)] = {
                "avg_mse": float(np.mean(fold_mses)),
                "std_mse": float(np.std(fold_mses)),
                "n_state": n_state_candidate,
            }

        # Select best by average validation MSE
        best_order = min(results.keys(), key=lambda k: results[k]["avg_mse"])
        return best_order, results

    def save(self, path: Path) -> None:
        """Save ARX model coefficients to JSON.

        Args:
            path: Output file path (should end in .json).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "n_state": self.n_state,
            "horizon": self.horizon,
            "coefficients": {},
        }
        for k, coeff in self.coefficients.items():
            data["coefficients"][str(k)] = {
                "W": coeff["W"].tolist(),
                "b": coeff["b"],
            }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ARXModel":
        """Load ARX model from JSON file.

        Args:
            path: Path to saved model file.

        Returns:
            Loaded ARXModel instance.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        model = cls(n_state=data["n_state"], horizon=data["horizon"])
        for k_str, coeff in data["coefficients"].items():
            k = int(k_str)
            model.coefficients[k] = {
                "W": np.array(coeff["W"]),
                "b": coeff["b"],
            }

        if len(model.coefficients) == model.horizon:
            model._is_fitted = True

        return model

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

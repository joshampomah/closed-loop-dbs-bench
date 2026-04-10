# Canonical owner: closed-loop-dbs-bench
"""ARI model loading and conversion utilities.

This module provides functions to load ARI (AutoRegressive with Inputs) models
from MATLAB .mat files and convert them to velocity form for MPC.

Example:
    >>> from dbs_bench.models.ari_model import load_ari_model
    >>> A_delta, B_delta, C_delta = load_ari_model(
    ...     "matlab/results/data/ari_model.mat"
    ... )
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.io import loadmat


def load_ari_model(
    mat_path: Union[str, Path],
    convert_to_velocity_form: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ARI model from MATLAB .mat file.

    Note: .mat files are not included in this repository. Use synthetic
    defaults or provide your own data.

    Args:
        mat_path: Path to .mat file containing the ARI model.
        convert_to_velocity_form: If True, returns velocity-form matrices.
            If False, returns standard state-space matrices.

    Returns:
        Tuple of (A, B, C) or (A_delta, B_delta, C_delta) arrays.

    Raises:
        FileNotFoundError: If mat_path does not exist.
        ValueError: If required fields are not found in .mat file.
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"ARI model file not found: {mat_path}")

    # Try to load as scipy .mat file first
    try:
        data = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        # MATLAB v7.3 format requires h5py
        import h5py

        with h5py.File(str(mat_path), "r") as f:
            # Look for velocity form matrices
            if "A_delta" in f:
                A_delta = np.array(f["A_delta"])
                B_delta = np.array(f["B_delta"]).T
                C_delta = np.array(f["C_delta"]).T
                return A_delta, B_delta, C_delta

            # Look in params struct
            if "params" in f:
                params = f["params"]
                if "A_delta" in params:
                    A_delta = np.array(params["A_delta"])
                    B_delta = np.array(params["B_delta"]).T
                    C_delta = np.array(params["C_delta"]).T
                    return A_delta, B_delta, C_delta

            # Look for standard form matrices
            if "A" in f and "B" in f and "C" in f:
                A = np.array(f["A"]).T
                B = np.array(f["B"]).T
                C = np.array(f["C"]).T

                if convert_to_velocity_form:
                    return build_velocity_form(A, B, C)
                return A, B, C

            raise ValueError(
                f"Could not find ARI model matrices in {mat_path}. "
                "Expected A_delta/B_delta/C_delta or A/B/C or params struct."
            )

    # Try different variable naming conventions
    if "A_delta" in data and "B_delta" in data and "C_delta" in data:
        # Already in velocity form
        A_delta = np.asarray(data["A_delta"])
        B_delta = np.asarray(data["B_delta"]).reshape(-1, 1)
        C_delta = np.asarray(data["C_delta"]).reshape(1, -1)
        return A_delta, B_delta, C_delta

    elif "A" in data and "B" in data and "C" in data:
        # Standard state-space form
        A = np.asarray(data["A"])
        B = np.asarray(data["B"]).reshape(-1, 1)
        C = np.asarray(data["C"]).reshape(1, -1)

        if convert_to_velocity_form:
            return build_velocity_form(A, B, C)
        return A, B, C

    elif "model" in data:
        # MATLAB struct containing model
        model = data["model"]

        if hasattr(model, "A_delta"):
            A_delta = np.asarray(model.A_delta)
            B_delta = np.asarray(model.B_delta).reshape(-1, 1)
            C_delta = np.asarray(model.C_delta).reshape(1, -1)
            return A_delta, B_delta, C_delta

        elif hasattr(model, "A"):
            A = np.asarray(model.A)
            B = np.asarray(model.B).reshape(-1, 1)
            C = np.asarray(model.C).reshape(1, -1)

            if convert_to_velocity_form:
                return build_velocity_form(A, B, C)
            return A, B, C

    # Try params struct (common in older files)
    elif "params" in data:
        params = data["params"]

        if hasattr(params, "A_delta"):
            A_delta = np.asarray(params.A_delta)
            B_delta = np.asarray(params.B_delta).reshape(-1, 1)
            C_delta = np.asarray(params.C_delta).reshape(1, -1)
            return A_delta, B_delta, C_delta

    raise ValueError(
        f"Could not find ARI model matrices in {mat_path}. "
        "Expected A/B/C or A_delta/B_delta/C_delta or model/params struct."
    )


def build_velocity_form(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, forgetting_factor: float = 0.999
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert standard state-space to velocity form.

    Given a discrete-time system:
        x_{k+1} = A·x_k + B·u_k
        y_k = C·x_k

    Returns the velocity-form system:
        chi^Delta_{k+1} = A^Delta·chi^Delta_k + B^Delta·Delta_u_k
        y_k = C^Delta·chi^Delta_k

    where chi^Delta = [x; u_{k-1}] is the augmented state.

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

    # Augmented state: chi = [x; u_{k-1}]
    # Augmented dynamics:
    #   x_{k+1} = A·x_k + B·u_k = A·x_k + B·(u_{k-1} + Delta_u_k)
    #   u_k = lambda·u_{k-1} + Delta_u_k  (with forgetting factor lambda for stability)
    #
    # So: [x_{k+1}]   [A  B] [x_k    ]   [B]
    #     [u_k    ] = [0  lambda] [u_{k-1}] + [1] Delta_u_k

    A_delta = np.block([[A, B], [np.zeros((1, n)), np.array([[forgetting_factor]])]])

    B_delta = np.vstack([B, np.array([[1.0]])])

    C_delta = np.hstack([C, np.array([[0.0]])])

    return A_delta, B_delta, C_delta


def get_initial_state(
    mat_path: Union[str, Path],
    y_history: Optional[np.ndarray] = None,
    u_prev: float = 0.0,
) -> np.ndarray:
    """Get initial velocity-form state for simulation.

    Args:
        mat_path: Path to .mat file containing model info.
        y_history: Initial output history. If None, uses steady state.
        u_prev: Initial previous control.

    Returns:
        Initial velocity-form state chi^Delta_0.
    """
    A_delta, B_delta, C_delta = load_ari_model(mat_path)
    n_states = A_delta.shape[0]

    if y_history is not None:
        # Reconstruct state from output history (simplified)
        # For observable systems, use observer canonical form
        chi_delta = np.zeros(n_states)
        chi_delta[-1] = u_prev

        # Set first state element to current output (approximation)
        chi_delta[0] = y_history[0] if len(y_history) > 0 else 0.0
    else:
        # Zero initial state
        chi_delta = np.zeros(n_states)
        chi_delta[-1] = u_prev

    return chi_delta


def validate_velocity_form(
    A_delta: np.ndarray, B_delta: np.ndarray, C_delta: np.ndarray
) -> Dict[str, bool]:
    """Validate velocity-form state-space matrices.

    Args:
        A_delta: Velocity form state matrix.
        B_delta: Velocity form input matrix.
        C_delta: Velocity form output matrix.

    Returns:
        Dictionary with validation results.
    """
    results = {}

    # Check dimensions
    n = A_delta.shape[0]
    results["dims_consistent"] = (
        A_delta.shape == (n, n)
        and B_delta.shape[0] == n
        and C_delta.shape[1] == n
    )

    # Check stability (eigenvalues)
    eigs = np.linalg.eigvals(A_delta)
    max_eig = np.max(np.abs(eigs))
    results["stable"] = max_eig <= 1.0
    results["max_eigenvalue"] = float(max_eig)

    # Velocity form should have eigenvalue at 1 (integrator)
    has_integrator = np.any(np.abs(eigs - 1.0) < 1e-6)
    results["has_integrator"] = has_integrator

    # Check controllability (simplified)
    # Full check would use controllability matrix rank
    results["B_nonzero"] = np.any(B_delta != 0)

    # Check observability (simplified)
    results["C_nonzero"] = np.any(C_delta != 0)

    return results

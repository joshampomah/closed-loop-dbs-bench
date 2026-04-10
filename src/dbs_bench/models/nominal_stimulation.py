# Canonical owner: closed-loop-dbs-bench
"""
Nominal stimulation response model for DBS.

From CDC24 paper Equation 2:
    x_dot_c(t) = [[-1/tau_1,    0   ],  x_c(t) + [[k/tau_1],  u(t)
              [ 1/tau_2,  -1/tau_2 ]]            [  0  ]]

    eta(t) = [0, 1] x_c(t)

Nominal parameters (from averaged patient data):
    k = 62.11
    tau_1 = 0.05 s
    tau_2 = 0.25 s

This module:
- Discretizes the continuous system with zero-order hold
- Handles parameter perturbations with stability checking
"""

import numpy as np
from scipy.signal import cont2discrete


def create_nominal_stimulation_model(Ts=0.02, k=62.11, tau1=0.05, tau2=0.25):
    """
    Discretize the nominal stimulation response model.

    Continuous system (Eq. 2):
        x_dot_c = A_c x_c + B_c u
        eta = C_c x_c

    where:
        A_c = [[-1/tau_1,    0   ],
               [ 1/tau_2,  -1/tau_2 ]]
        B_c = [[k/tau_1],
               [  0  ]]
        C_c = [0, 1]

    Parameters:
    -----------
    Ts : float
        Sampling period (default: 0.02s for 50 Hz)
    k : float
        Gain parameter (default: 62.11)
    tau1 : float
        First time constant (default: 0.05 s)
    tau2 : float
        Second time constant (default: 0.25 s)

    Returns:
    --------
    A_eta : ndarray (2, 2)
        Discrete state matrix
    B_eta : ndarray (2, 1)
        Discrete input matrix
    C_eta : ndarray (1, 2)
        Output matrix (same for continuous/discrete)
    params : dict
        Parameters {k, tau1, tau2, Ts}
    """
    # Continuous-time matrices (CDC24 Equation 2)
    A_c = np.array([[-1/tau1,      0    ],
                    [ 1/tau2,  -1/tau2  ]], dtype=np.float32)
    B_c = np.array([[k/tau1],
                    [  0   ]], dtype=np.float32)
    C_c = np.array([[0, 1]], dtype=np.float32)
    D_c = np.array([[0]], dtype=np.float32)

    # Discretize with zero-order hold (scipy returns float64, cast to float32)
    sys_discrete = cont2discrete((A_c, B_c, C_c, D_c), Ts, method='zoh')
    A_eta = np.asarray(sys_discrete[0], dtype=np.float32)
    B_eta = np.asarray(sys_discrete[1], dtype=np.float32)
    C_eta = np.asarray(sys_discrete[2], dtype=np.float32)

    # Store parameters
    params = {
        'k': k,
        'tau1': tau1,
        'tau2': tau2,
        'Ts': Ts,
        'A_c': A_c,
        'B_c': B_c,
        'C_c': C_c,
        'A_eta': A_eta,
        'B_eta': B_eta,
        'C_eta': C_eta,
    }

    return A_eta, B_eta, C_eta, params


def check_stability(A):
    """
    Check if discrete-time system is stable.

    For discrete-time: eigenvalues must be inside unit circle.

    Parameters:
    -----------
    A : ndarray
        State matrix

    Returns:
    --------
    is_stable : bool
        True if all eigenvalues satisfy |lambda| < 1
    max_abs_eig : float
        Maximum absolute eigenvalue
    """
    eigs = np.linalg.eigvals(A)
    abs_eigs = np.abs(eigs)
    max_abs_eig = np.max(abs_eigs)
    is_stable = max_abs_eig < 1.0

    return is_stable, max_abs_eig


def compute_static_gain(A_c, B_c, C_c):
    """
    Compute DC gain of continuous-time system.

    DC gain = C @ (-A)^{-1} @ B

    For the stimulation response, this should equal k.

    Parameters:
    -----------
    A_c, B_c, C_c : ndarrays
        Continuous-time system matrices

    Returns:
    --------
    dc_gain : float
        Static gain
    """
    dc_gain = -C_c @ np.linalg.inv(A_c) @ B_c
    return dc_gain[0, 0]


def perturb_stimulation_params(params_nominal, max_deviation=0.30, seed=None, max_attempts=1000):
    """
    Generate perturbed parameters with constraints.

    Paper constraints:
    - Uniform distribution within +/-max_deviation
    - Open-loop stable (eigenvalues inside unit circle for discrete)
    - Static gain change <= max_deviation

    Parameters:
    -----------
    params_nominal : dict
        Nominal parameters {k, tau1, tau2, Ts}
    max_deviation : float
        Maximum relative deviation (default: 0.30 = 30%)
    seed : int or None
        Random seed for reproducibility
    max_attempts : int
        Maximum sampling attempts

    Returns:
    --------
    params_perturbed : dict
        Perturbed parameters (same structure as params_nominal)
    is_valid : bool
        Whether perturbation satisfies constraints
    """
    if seed is not None:
        np.random.seed(seed)

    k_nom = params_nominal['k']
    tau1_nom = params_nominal['tau1']
    tau2_nom = params_nominal['tau2']
    Ts = params_nominal['Ts']

    # Nominal DC gain
    A_c_nom = np.array([[-1/tau1_nom, 0], [1/tau2_nom, -1/tau2_nom]], dtype=np.float32)
    B_c_nom = np.array([[k_nom/tau1_nom], [0]], dtype=np.float32)
    C_c_nom = np.array([[0, 1]], dtype=np.float32)
    dc_gain_nom = compute_static_gain(A_c_nom, B_c_nom, C_c_nom)

    for attempt in range(max_attempts):
        # Sample uniformly
        k_pert = k_nom * (1 + np.random.uniform(-max_deviation, max_deviation))
        tau1_pert = tau1_nom * (1 + np.random.uniform(-max_deviation, max_deviation))
        tau2_pert = tau2_nom * (1 + np.random.uniform(-max_deviation, max_deviation))

        # Build perturbed continuous system
        A_c_pert = np.array([[-1/tau1_pert, 0],
                             [1/tau2_pert, -1/tau2_pert]], dtype=np.float32)
        B_c_pert = np.array([[k_pert/tau1_pert], [0]], dtype=np.float32)
        C_c_pert = np.array([[0, 1]], dtype=np.float32)

        # Check continuous-time stability (eigenvalues must have negative real part)
        eigs_c = np.linalg.eigvals(A_c_pert)
        if not np.all(np.real(eigs_c) < 0):
            continue

        # Discretize
        A_eta_pert, B_eta_pert, C_eta_pert, _ = create_nominal_stimulation_model(
            Ts=Ts, k=k_pert, tau1=tau1_pert, tau2=tau2_pert
        )

        # Check discrete-time stability
        is_stable, max_eig = check_stability(A_eta_pert)
        if not is_stable:
            continue

        # Check static gain change
        dc_gain_pert = compute_static_gain(A_c_pert, B_c_pert, C_c_pert)
        gain_change = abs(dc_gain_pert - dc_gain_nom) / abs(dc_gain_nom)
        if gain_change > max_deviation:
            continue

        # Valid perturbation found
        params_perturbed = {
            'k': k_pert,
            'tau1': tau1_pert,
            'tau2': tau2_pert,
            'Ts': Ts,
            'A_c': A_c_pert,
            'B_c': B_c_pert,
            'C_c': C_c_pert,
            'A_eta': A_eta_pert,
            'B_eta': B_eta_pert,
            'C_eta': C_eta_pert,
            'dc_gain': dc_gain_pert,
            'max_eig': max_eig,
        }
        return params_perturbed, True

    # Failed to find valid perturbation
    print(f"Warning: Could not find valid perturbation after {max_attempts} attempts")
    return params_nominal, False


def simulate_step_response(A_eta, B_eta, C_eta, T_sim=2.0, Ts=0.02, u_step=1.0):
    """
    Simulate step response of discretized system.

    Parameters:
    -----------
    A_eta, B_eta, C_eta : ndarrays
        Discrete system matrices
    T_sim : float
        Simulation duration (default: 2.0 s)
    Ts : float
        Sampling period
    u_step : float
        Step input magnitude

    Returns:
    --------
    t : ndarray
        Time vector
    y : ndarray
        Output response eta(t)
    x : ndarray
        State trajectory
    """
    N = int(T_sim / Ts)
    t = np.arange(N) * Ts

    # Initialize
    x = np.zeros((2, N))
    y = np.zeros(N)
    u = u_step * np.ones(N)

    # Simulate
    for k in range(N-1):
        y[k] = (C_eta @ x[:, k])[0]
        x[:, k+1] = A_eta @ x[:, k] + B_eta.flatten() * u[k]

    y[-1] = (C_eta @ x[:, -1])[0]

    return t, y, x

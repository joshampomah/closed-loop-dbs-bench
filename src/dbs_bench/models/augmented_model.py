# Canonical owner: closed-loop-dbs-bench
"""
Augmented state-space model for DBS closed-loop control.

From CDC24 paper Equation 8:
    chi_{k+1} = A(theta_beta) chi_k + B u_k + w_k
    y_k = [C_eta, C_beta] chi_k

State vector (8 states for n_beta=5):
    chi_k = [x_{eta,k}^T, beta_k, Delta_beta_k, Delta_beta_{k-1}, ..., Delta_beta_{k-n_beta+1}]^T
         = [x_eta_1, x_eta_2, beta, Delta_beta_0, Delta_beta_-1, Delta_beta_-2, Delta_beta_-3, Delta_beta_-4]^T

Velocity form (Equation 10) adds u_{k-1} to state:
    chi^Delta_k = [chi_k^T, u_{k-1}]^T  (9 states)
"""

import numpy as np


def build_augmented_model(alpha_params, A_eta, B_eta, C_eta):
    """
    Build augmented state-space model (Eq. 8).

    Combines:
    - Stimulation response: x_{eta,k+1} = A_eta x_{eta,k} + B_eta u_k
    - ARI beta dynamics: Delta_beta_k = Sum alpha_i Delta_beta_{k-i} + delta_k

    State vector:
        chi_k = [x_{eta,k}^T, beta_k, Delta_beta_k, Delta_beta_{k-1}, ..., Delta_beta_{k-n_beta+1}]^T

    Parameters:
    -----------
    alpha_params : ndarray (n_beta,)
        ARI coefficients [alpha_1, alpha_2, ..., alpha_{n_beta}]
    A_eta, B_eta, C_eta : ndarrays
        Stimulation response matrices

    Returns:
    --------
    A_aug : ndarray (n_states, n_states)
        Augmented state matrix
    B_aug : ndarray (n_states, 1)
        Augmented input matrix
    C_aug : ndarray (1, n_states)
        Augmented output matrix [C_eta, C_beta]
    state_info : dict
        Information about state structure
    """
    n_beta = len(alpha_params)
    n_eta = A_eta.shape[0]  # Should be 2

    # Total states: n_eta + (n_beta + 1)
    # Structure: [x_eta_1, x_eta_2, beta, Delta_beta_0, Delta_beta_-1, ..., Delta_beta_-(n_beta-1)]
    n_states = n_eta + n_beta + 1

    # Initialize augmented matrices
    A_aug = np.zeros((n_states, n_states))
    B_aug = np.zeros((n_states, 1))

    # ===== Top-left block: stimulation dynamics =====
    A_aug[:n_eta, :n_eta] = A_eta
    B_aug[:n_eta, 0] = B_eta.flatten()

    # ===== Bottom-right block: ARI dynamics =====
    idx_beta = n_eta  # Index for beta_k
    idx_delta_start = n_eta + 1  # Start of Delta_beta states

    # Row for beta_k: beta_{k+1} = beta_k + Delta_beta_k
    A_aug[idx_beta, idx_beta] = 1.0  # beta_k term
    A_aug[idx_beta, idx_delta_start:idx_delta_start+n_beta] = alpha_params  # Sum alpha_i Delta_beta_{k-i} term

    # Rows for Delta_beta states
    for i in range(n_beta):
        if i == 0:
            # Delta_beta_{k+1} = Sum alpha_j Delta_beta_{k+1-j}
            A_aug[idx_delta_start, idx_delta_start:idx_delta_start+n_beta] = alpha_params
        else:
            # Shift register: Delta_beta_{k+1-i} = Delta_beta_{k-i}
            A_aug[idx_delta_start+i, idx_delta_start+i-1] = 1.0

    # ===== Output equation: y_k = beta_k - eta_k =====
    C_aug = np.zeros((1, n_states))
    C_aug[0, idx_beta] = 1.0  # beta_k term
    C_aug[0, :n_eta] = -C_eta.flatten()  # -eta_k term

    # State information for debugging/visualization
    state_names = [f'x_eta_{i+1}' for i in range(n_eta)]
    state_names.append('beta')
    state_names.extend([f'Delta_beta_k-{i}' for i in range(n_beta)])

    state_info = {
        'n_states': n_states,
        'n_eta': n_eta,
        'n_beta': n_beta,
        'idx_beta': idx_beta,
        'idx_delta_start': idx_delta_start,
        'state_names': state_names,
        'alpha_params': alpha_params,
    }

    return A_aug, B_aug, C_aug, state_info


def build_velocity_form_model(A_aug, B_aug, C_aug):
    """
    Build velocity form augmented model (Eq. 10).

    Augments state with u_{k-1} to enable Delta_u as input.

    State vector:
        chi^Delta_k = [chi_k^T, u_{k-1}]^T

    Dynamics:
        chi^Delta_{k+1} = A^Delta chi^Delta_k + B^Delta Delta_u_k
        y_k = C^Delta chi^Delta_k

    where Delta_u_k = u_k - u_{k-1}

    Parameters:
    -----------
    A_aug, B_aug, C_aug : ndarrays
        Augmented model matrices

    Returns:
    --------
    A_delta, B_delta, C_delta : ndarrays
        Velocity form matrices
    """
    n_states = A_aug.shape[0]

    # Augment with u_{k-1}
    A_delta = np.zeros((n_states + 1, n_states + 1))
    A_delta[:n_states, :n_states] = A_aug
    A_delta[:n_states, n_states] = B_aug.flatten()  # chi_{k+1} depends on u_{k-1}
    A_delta[n_states, n_states] = 1.0  # u_k = u_{k-1} + Delta_u_k

    B_delta = np.zeros((n_states + 1, 1))
    B_delta[:n_states, 0] = B_aug.flatten()  # chi_{k+1} depends on Delta_u_k
    B_delta[n_states, 0] = 1.0  # u_k = u_{k-1} + Delta_u_k

    C_delta = np.zeros((1, n_states + 1))
    C_delta[0, :n_states] = C_aug.flatten()
    # C_delta[0, n_states] = 0  (y doesn't directly depend on u_{k-1})

    return A_delta, B_delta, C_delta


def simulate_augmented_system(A_aug, B_aug, C_aug, x0, u_seq, state_info):
    """
    Simulate augmented system with given input sequence.

    Parameters:
    -----------
    A_aug, B_aug, C_aug : ndarrays
        Augmented system matrices
    x0 : ndarray (n_states,)
        Initial state
    u_seq : ndarray (N,)
        Input sequence
    state_info : dict
        State structure information

    Returns:
    --------
    t : ndarray (N,)
        Time indices
    x_traj : ndarray (N, n_states)
        State trajectory
    y_traj : ndarray (N,)
        Output trajectory
    """
    N = len(u_seq)
    n_states = len(x0)

    x_traj = np.zeros((N, n_states))
    y_traj = np.zeros(N)

    x_traj[0, :] = x0
    y_traj[0] = (C_aug @ x0)[0]

    for k in range(N-1):
        x_traj[k+1, :] = A_aug @ x_traj[k, :] + B_aug.flatten() * u_seq[k]
        y_traj[k+1] = (C_aug @ x_traj[k+1, :])[0]

    t = np.arange(N)

    return t, x_traj, y_traj

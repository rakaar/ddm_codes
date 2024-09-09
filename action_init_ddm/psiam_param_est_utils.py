import numpy as np
from psiam_utils import rho_A_t_fn, cum_A_t_fn
from psiam_utils import rho_E_minus_small_t_NORM_fn, P_small_t_btn_x1_x2

def calculate_abort_loglike(t, V_A, theta_A, t_a):
    P_A = rho_A_t_fn(t, V_A, theta_A, t_a)
    # p_abort = (P_A * (1 - C_E) + P_E * (1 - C_A))
    # Since, C_E = 0 and P_E = 0
    p_abort = P_A
    if p_abort <= 0 or np.isnan(p_abort):
        p_abort = 1e-6
    return np.log(p_abort)

def calculate_correct_loglike(t, V_A, theta_A, t_a, V_E, theta_E, Z, K_max, t_stim, t_E):
    P_A = rho_A_t_fn(t, V_A, theta_A, t_a)
    P_E_btn_1_2 = P_small_t_btn_x1_x2(1, 2, t, V_E, theta_E, Z, K_max, t_stim)
    P_E_plus = rho_E_minus_small_t_NORM_fn(t, -V_E, theta_E, K_max, t_stim, -Z, t_E)
    C_A = cum_A_t_fn(t, V_A, theta_A, t_a)
    p_correct = (P_A * P_E_btn_1_2 + P_E_plus * (1 - C_A))
    if p_correct <= 0 or np.isnan(p_correct):
        p_correct = 1e-6
    return np.log(p_correct)

def calculate_wrong_loglike(t, V_A, theta_A, t_a, V_E, theta_E, Z, K_max, t_stim, t_E):
    P_A = rho_A_t_fn(t, V_A, theta_A, t_a)
    P_E_btn_0_1 = P_small_t_btn_x1_x2(0, 1, t, V_E, theta_E, Z, K_max, t_stim)
    P_E_minus = rho_E_minus_small_t_NORM_fn(t, V_E, theta_E, K_max, t_stim, Z, t_E)
    C_A = cum_A_t_fn(t, V_A, theta_A, t_a)
    p_wrong = (P_A * P_E_btn_0_1 + P_E_minus * (1 - C_A))
    if p_wrong <= 0 or np.isnan(p_wrong):
        p_wrong = 1e-6
    return np.log(p_wrong)
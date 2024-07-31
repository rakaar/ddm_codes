import numpy as np
from scipy import integrate
from scipy.integrate import quad
from scipy.special import erf

def rho_A_t_fn(t, V_A, theta_A, t_a = 0):
    """
    For AI,prob density of t given V_A, theta_A
    """
    return (theta_A*1/np.sqrt(2*np.pi*(t - t_a)**3))*np.exp(-0.5 * (V_A**2) * (((t - t_a) - (theta_A/V_A))**2)/(t - t_a))

def rho_A_t_arr_fn(t_arr, V_A, theta_A, t_a=0):
    """
    For AI, prob density of t arr
    """
    return np.array([rho_A_t_fn(t, V_A, theta_A, t_a) for t in t_arr])


def rho_E_minus_t_fn(t, V_E, theta_E, K_max, Z_E=0, t_E=0):
    """
    for EA, prob density of t given V_E, theta_E
    """
    non_sum_term = (np.pi / (2 * theta_E)**2) * np.exp(-V_E * (Z_E + theta_E) - 0.5*(V_E**2) * (t - t_E))

    k_pts = np.arange(1, K_max + 1)
    sine_term = np.sin(k_pts * np.pi * (Z_E + theta_E) / (2 * theta_E))
    exp_term = np.exp(-((k_pts**2 * np.pi**2) / (2 * (2 * theta_E)**2)) * (t - t_E))
    sum_term = np.sum(k_pts * sine_term * exp_term)    
    p_E_minus = non_sum_term * sum_term
    
    return p_E_minus

def rho_E_t_fn(t, V_E, theta_E, K_max, Z_E=0, t_E=0):
    """
    for EA, prob density of t given V_E, theta_E
    """
    return  rho_E_minus_t_fn(t, V_E, theta_E, K_max, Z_E, t_E) + rho_E_minus_t_fn(t, -V_E, theta_E, K_max, -Z_E, t_E)

def rho_E_minus_t_arr_fn(t_arr, V_E, theta_E, K_max, Z_E=0, t_E=0):
    """
    for EA, prob density of t arr
    """
    return np.array([rho_E_minus_t_fn(t, V_E, theta_E, K_max, Z_E, t_E) for t in t_arr])


def rho_E_t_arr_fn(t_arr, V_E, theta_E, K_max, Z_E=0, t_E=0):
    """
    for EA, prob density of t arr
    """
    rho_E_minus = rho_E_minus_t_arr_fn(t_arr, V_E, theta_E, K_max, Z_E, t_E)
    rho_E_plus = rho_E_minus_t_arr_fn(t_arr, -V_E, theta_E, K_max, -Z_E, t_E)
    return  rho_E_minus + rho_E_plus


def Phi(x):
    """
    Define the normal cumulative distribution function Φ(x) using erf
    """
    return 0.5 * (1 + erf(x / np.sqrt(2)))

def cum_A_t_fn(t, V_A, theta_A, t_A=0):
    """
    For AI, calculate cummulative distrn of a time t given V_A, theta_A
    """
    term1 = Phi(V_A * ((t - t_A) - (theta_A/V_A)) / np.sqrt(t - t_A))
    term2 = np.exp(-2 * V_A * theta_A) * Phi(-V_A * ((t - t_A) + (theta_A / V_A)) / np.sqrt(t - t_A))
    
    return term1 + term2

def cum_A_t_arr_fn(t_arr, V_A, theta_A, t_A=0):
    """
    For AI, calculate cummulative distrn of a time arr given V_A, theta_A
    """
    return np.array([cum_A_t_fn(t, V_A, theta_A, t_A) for t in t_arr])  


def cum_E_minus_t_non_norm_fn(t, V_E, theta_E, K_max):
    """
    For EA, calculate cummulative distrn of a time t for reaching lower bound given V_E, theta_E for  K_max
    """
    term1 = (np.pi/(2 * theta_E)**2) * np.exp(-V_E * theta_E)

    k_terms = np.arange(1, K_max + 1)
    sum_term = 0
    for k in k_terms:
        sine_term = np.sin(k*np.pi/2)
        alpha = -((V_E**2)/2)  - ((k**2) * (np.pi**2) / (2 * (2 * theta_E)**2))
        sum_term += k*sine_term*(1/alpha)*(np.exp(alpha*t) - 1)
    
    return term1 * sum_term

def cum_E_t_fn(t, V_E, theta_E, K_max):
    """
    For EA, calculate cummulative distrn of a time arr for reaching both bounds given V_E, theta_E for  K_max
    """
    return cum_E_minus_t_non_norm_fn(t, V_E, theta_E, K_max) + cum_E_minus_t_non_norm_fn(t, -V_E, theta_E, K_max)

def cum_E_t_arr_fn(t_arr, V_E, theta_E, K_max, min_val):
    """
    For EA, calculate cummulative distrn of a time arr given V_E, theta_E for  K_max
    """
    # norm_const, _ = quad(rho_E_t_fn, 0, np.inf, args=(V_E, theta_E, K_max))
    # normalized_pdf = lambda x: rho_E_t_fn(x, V_E, theta_E, K_max) / norm_const

    cdf_arr = np.zeros((len(t_arr),1))
    for idx, t in enumerate(t_arr):
        cdf_arr[idx], _ = quad(rho_E_t_fn, min_val, t, args=(V_E, theta_E, K_max))

    return cdf_arr
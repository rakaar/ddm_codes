import numpy as np
from scipy import integrate
from scipy.special import erf
from scipy.integrate import quad

def calculate_histogram(x_axis, y_axis):
    x_axis = np.sort(x_axis)
    histcounts, _ = np.histogram(y_axis, bins=x_axis)
    prob = histcounts/np.sum(histcounts)
    return prob

def density_A_fn(t, V_A, theta_A,t_a = 0):
    return (theta_A*1/np.sqrt(2*np.pi*(t - t_a)**3))*np.exp(-0.5 * (V_A**2) * (((t - t_a) - (theta_A/V_A))**2)/(t - t_a))

def prob_A_fn(t_arr, V_A, theta_A):
    N_t = len(t_arr)
    prob_arr = np.zeros((N_t-1,1))
    for i in range(0, N_t-1):
        prob_arr[i] = integrate.quad(density_A_fn, t_arr[i], t_arr[i+1], args=(V_A, theta_A))[0]
    return prob_arr


def density_E_minus_fn(t, V_E, theta_E, K_max, Z_E=0, t_E=0):
    non_sum_term = (np.pi / (2 * theta_E)**2) * np.exp(-V_E * (Z_E + theta_E) - 0.5*(V_E**2) * (t - t_E))

    k_pts = np.arange(1, K_max + 1)
    sine_term = np.sin(k_pts * np.pi * (Z_E + theta_E) / (2 * theta_E))
    exp_term = np.exp(-((k_pts**2 * np.pi**2) / (2 * (2 * theta_E)**2)) * (t - t_E))
    sum_term = np.sum(k_pts * sine_term * exp_term)    
    p_E_minus = non_sum_term * sum_term
    
    return p_E_minus

def Phi(x):
    # Define the normal cumulative distribution function Φ(x) using erf
    return 0.5 * (1 + erf(x / np.sqrt(2)))

def c_A(t, V_A, theta_A, t_A=0):
    term1 = Phi(V_A * ((t - t_A) - (theta_A/V_A)) / np.sqrt(t - t_A))
    term2 = np.exp(-2 * V_A * theta_A) * Phi(-V_A * ((t - t_A) + (theta_A / V_A)) / np.sqrt(t - t_A))
    
    return term1 + term2



def c_E(t, V_E, theta_E, K_max, Z_E=0, t_E=0):
    term1 = np.pi/(2 * theta_E)**2
    term2 = np.exp(-V_E * (Z_E + theta_E))

    k_terms = np.arange(1, K_max + 1)
    inside_sum_term = 0
    t_pre_term1 = -(V_E**2)/2
    for k in k_terms:
        sine_term = np.sin(k * np.pi * (Z_E + theta_E) / (2 * theta_E))
        t_pre_term2 = -k**2 * (np.pi**2) / (2 * (2 * theta_E)**2)
        t_pre_term = t_pre_term1 + t_pre_term2

        inside_sum_term += (1/t_pre_term) * (np.exp(t_pre_term * (t - t_E)) - np.exp(0)) * k * sine_term


    return term1 * term2 * inside_sum_term


def c_E_minimal(t, V_E, theta_E, K_max):
    term1 = (np.pi/(2 * theta_E)**2) * np.exp(-V_E * theta_E)

    k_terms = np.arange(1, K_max + 1)
    sum_term = 0
    for k in k_terms:
        sine_term = np.sin(k*np.pi/2)
        alpha = -((V_E**2)/2)  - ((k**2) * (np.pi**2) / (2 * (2 * theta_E)**2))
        sum_term += k*sine_term*(1/alpha)*(np.exp(alpha*t) - 1)
    
    return term1 * sum_term
    
def p_E_minus_and_plus(t, V_E, theta_E, K_max):
    return p_E_minus(t, V_E, theta_E, K_max) + p_E_minus(t, -V_E, theta_E, K_max)


def p_E_minus(t, V_E, theta_E, K_max):
    """
    p_E(t | V_E, theta_E) for K_max terms, t_E = 0, Z_E = 0
    """
    term1 = np.pi/(2 * theta_E)**2
    term2 = np.exp(-(V_E * theta_E) - 0.5*(V_E**2)*t)
    
    k_range = np.arange(1, K_max + 1)
    sine_terms = np.sin(k_range * np.pi /2)
    k_range_sq = k_range**2
    exp_term = np.exp(-t*(k_range_sq * np.pi**2) / (2 * (2 * theta_E)**2))
    sum_term = np.sum(k_range * sine_terms * exp_term)

    return term1 * term2 * sum_term

def cdf_E_both_bounds(t, V_E, theta_E, K_max):
    """
    CDF of normalized p_E_minimal from 0 to t
    """
    norm_const = normalization_constant(V_E, theta_E, K_max)
    normalized_pdf = lambda x: p_E_minus_and_plus(x, V_E, theta_E, K_max) / norm_const
    cdf, _ = quad(normalized_pdf, 0, t)
    return cdf


def normalization_constant(V_E, theta_E, K_max):
    """
    Compute the normalization constant for p_E_minimal over [0, ∞)
    """
    integral, _ = quad(p_E_minus_and_plus, 0, np.inf, args=(V_E, theta_E, K_max))
    return integral


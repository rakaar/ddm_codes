import numpy as np
from scipy import integrate
from scipy.integrate import quad
from scipy.special import erf
from numba import jit

@jit
def simulate_psiam(V_A, theta_A, V_E, theta_E, Z_E, t_stim):
    AI = 0; DV = Z_E; t = 0; dt = 1e-6; dB = dt**0.5
    is_act = 0
    while True:
        if t*dt > t_stim:
            DV += V_E*dt + np.random.normal(0, dB)
        
        AI += V_A*dt + np.random.normal(0, dB)
        
        t += 1
        
        if DV >= theta_E:
            choice = +1; RT = t*dt
            break
        elif DV <= -theta_E:
            choice = -1; RT = t*dt
            break
        
        if AI >= theta_A:
            is_act = 1
            if DV > 0:
                choice = 1; RT = t*dt
            elif DV < 0:
                choice = -1; RT = t*dt
            elif DV == 0:
                if t % 2 == 0:
                    choice = +1; RT = t*dt
                else:
                    choice = -1; RT = t*dt
            break
        
        

    return choice, RT, is_act

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



def rho_E_t_fn(t, V_E, theta_E, K_max, t_stim, Z_E=0, t_E=0):
    """
    for EA, prob density of t given V_E, theta_E
    """
    return  rho_E_minus_t_fn(t, V_E, theta_E, K_max, t_stim, Z_E, t_E) + rho_E_minus_t_fn(t, -V_E, theta_E, K_max, t_stim, -Z_E, t_E)

def rho_E_minus_t_arr_fn(t_arr, V_E, theta_E, K_max, t_stim, Z_E=0, t_E=0):
    """
    for EA, prob density of t arr
    """
    return np.array([rho_E_minus_t_fn(t, V_E, theta_E, K_max, t_stim, Z_E, t_E) for t in t_arr])


def rho_E_t_arr_fn(t_arr, V_E, theta_E, K_max, t_stim, Z_E=0, t_E=0):
    """
    for EA, prob density of t arr
    """
    rho_E_minus = rho_E_minus_t_arr_fn(t_arr, V_E, theta_E, K_max, t_stim, Z_E, t_E)
    rho_E_plus = rho_E_minus_t_arr_fn(t_arr, -V_E, theta_E, K_max, t_stim, -Z_E, t_E)
    return  rho_E_minus + rho_E_plus



def rho_E_minus_t_fn(t, V_E, theta_E, K_max, t_stim, Z_E=0, t_E=0):
    v = V_E
    a = 2*theta_E
    w = 1/2
    if t <= t_stim:
        return 0
    else:
        t = t - t_stim
    if t > 0.25:
        non_sum_term = (np.pi/a**2)*np.exp(-v*a*w - (v**2 * t/2))
        k_vals = np.linspace(1, K_max, K_max)
        sum_sine_term = np.sin(k_vals*np.pi*w)
        sum_exp_term = np.exp(-(k_vals**2 * np.pi**2 * t)/(2*a**2))
        sum_result = np.sum(k_vals * sum_sine_term * sum_exp_term)
    else:
        non_sum_term = (1/a**2)*(a**3/np.sqrt(2*np.pi*t**3))*np.exp(-v*a*w - (v**2 * t)/2)
        K_max = int(K_max/2)
        k_vals = np.linspace(-K_max, K_max, 2*K_max + 1)
        sum_w_term = w + 2*k_vals
        sum_exp_term = np.exp(-(a**2 * (w + 2*k_vals)**2)/(2*t))
        sum_result = np.sum(sum_w_term*sum_exp_term)

    # if sum_result < 0:
    #     sum_result += 1e-3
    
    density =  non_sum_term * sum_result
    if density < 0:
        raise ValueError("Density cannot be negative")
    else:
        return density
    
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


def cum_E_t_arr_fn(t_arr, V_E, theta_E, K_max, t_stim):
    """
    For EA, calculate cummulative distrn of a time arr given V_E, theta_E for  K_max
    """
    cdf_arr = np.zeros((len(t_arr)))
    for idx, t in enumerate(t_arr):
        cdf_arr[idx], _ = quad(rho_E_t_fn, 0, t, args=(V_E, theta_E, K_max, t_stim))

    return cdf_arr

def cum_E_t_minus_arr_fn(t_arr, V_E, theta_E, K_max, t_stim):
    """
    For EA, calculate cummulative distrn of a time arr given V_E, theta_E for  K_max
    """
    cdf_arr = np.zeros((len(t_arr)))
    for idx, t in enumerate(t_arr):
        cdf_arr[idx], _ = quad(rho_E_minus_t_fn, 0, t, args=(V_E, theta_E, K_max, t_stim))

    return cdf_arr
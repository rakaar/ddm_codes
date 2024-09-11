import numpy as np
from scipy import integrate
from scipy.integrate import quad
from scipy.special import erf
from numba import jit


@jit
def simulate_psiam(V_A, theta_A, V_E, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor):
    AI = 0; DV = Z_E; t = 0; dt = 1e-6; dB = dt**0.5
    is_act = 0
    while True:
        if t*dt > t_stim + t_E_aff:
            DV += V_E*dt + np.random.normal(0, dB)
        
        if t*dt > t_A_aff:
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
        
        

    return choice, RT+t_motor, is_act



def rho_A_t_fn(t, V_A, theta_A, t_A_aff, t_motor):
    """
    For AI,prob density of t given V_A, theta_A
    """
    t -= t_A_aff + t_motor
    if t <= 0:
        return 0
    return (theta_A*1/np.sqrt(2*np.pi*(t)**3))*np.exp(-0.5 * (V_A**2) * (((t) - (theta_A/V_A))**2)/(t))



def rho_E_minus_small_t_NORM_fn(t, V_E, theta_E, K_max, t_stim, Z_E, t_E_aff, t_motor):
    """
    in normalized time, PDF of hitting the lower bound
    """
    v = V_E*theta_E
    w = (Z_E + theta_E)/(2*theta_E)
    a = 2
    t -= t_stim + t_E_aff + t_motor

    if t <= 0:
        return 0

    t /= theta_E**2

    
    non_sum_term = (1/a**2)*(a**3/np.sqrt(2*np.pi*t**3))*np.exp(-v*a*w - (v**2 * t)/2)
    K_max = int(K_max/2)
    k_vals = np.linspace(-K_max, K_max, 2*K_max + 1)
    sum_w_term = w + 2*k_vals
    sum_exp_term = np.exp(-(a**2 * (w + 2*k_vals)**2)/(2*t))
    sum_result = np.sum(sum_w_term*sum_exp_term)

    
    density =  non_sum_term * sum_result
    if density <= 0:
        density = 1e-16

    return density/theta_E**2


def P_small_t_btn_x1_x2(x1, x2, t, V_E, theta_E, Z, n_max, t_stim, t_E_aff, t_motor):
    """
    Integration of P_small(x,t) btn x1 and x2
    """
    v = V_E
    mu = v*theta_E
    z = 2 * (Z + theta_E)/(2*theta_E) # z is between 0 and 2
    t -= t_stim + t_E_aff + t_motor

    if t <= 0:
        return 0

    t /= (theta_E**2)

    result = 0
    
    sqrt_t = np.sqrt(t)
    
    for n in range(-n_max, n_max + 1):
        term1 = np.exp(4 * mu * n) * (
            Phi((x2 - (z + 4 * n + mu * t)) / sqrt_t) -
            Phi((x1 - (z + 4 * n + mu * t)) / sqrt_t)
        )
        
        term2 = np.exp(2 * mu * (2 * (1 - n) - z)) * (
            Phi((x2 - (-z + 4 * (1 - n) + mu * t)) / sqrt_t) -
            Phi((x1 - (-z + 4 * (1 - n) + mu * t)) / sqrt_t)
        )
        
        result += term1 - term2
    
    return result

def Phi(x):
    """
    Define the normal cumulative distribution function Φ(x) using erf
    """
    return 0.5 * (1 + erf(x / np.sqrt(2)))

def cum_A_t_fn(t, V_A, theta_A, t_A_aff, t_motor):
    """
    For AI, calculate cummulative distrn of a time t given V_A, theta_A
    """
    t -= t_A_aff + t_motor

    if t <= 0:
        return 0

    term1 = Phi(V_A * ((t) - (theta_A/V_A)) / np.sqrt(t))
    term2 = np.exp(2 * V_A * theta_A) * Phi(-V_A * ((t) + (theta_A / V_A)) / np.sqrt(t))
    
    return term1 + term2


def rho_E_t_fn(t, V_E, theta_E, K_max, t_stim, Z_E, t_E_aff, t_motor):
    """
    for EA, prob density of t given V_E, theta_E
    """
    return  rho_E_minus_small_t_NORM_fn(t, V_E, theta_E, K_max, t_stim, Z_E, t_E_aff, t_motor) + rho_E_minus_small_t_NORM_fn(t, -V_E, theta_E, K_max, t_stim, -Z_E, t_E_aff, t_motor)

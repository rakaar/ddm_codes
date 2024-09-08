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



def rho_E_minus_t_NORM_fn(t, V_E, theta_E, K_max, t_stim, Z_E, t_E=0):
    """
    in normalized time, PDF of hitting the lower bound
    """
    v = V_E*theta_E
    w = (Z_E + theta_E)/(2*theta_E)
    a = 2
    if t < t_stim:
        return 0
    else:
        t = t - t_stim

    t /= theta_E**2

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

    
    density =  non_sum_term * sum_result
    if density <= 0:
        density = 1e-16

    return density/theta_E**2

def rho_E_minus_t_fn(t, V_E, theta_E, K_max, t_stim, Z_E=0, t_E=0):
    v = V_E
    a = 2*theta_E
    w = (Z_E + theta_E)/(2*theta_E)
    
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

    
    density =  non_sum_term * sum_result
    if density <= 0:
        density = 1e-16

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
    term2 = np.exp(2 * V_A * theta_A) * Phi(-V_A * ((t - t_A) + (theta_A / V_A)) / np.sqrt(t - t_A))
    
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

def F(x, mu, n):
    """
    Integration of e^mu*x sin(n*pi*x/2)
    """
    term1 = np.exp(mu*x)/(mu**2 + (n*np.pi/2)**2)
    term2 = mu*np.sin(n*np.pi*x/2)  - n*(np.pi/2)*np.cos(n*np.pi*x/2)
    return term1*term2


def P_small_t_btn_1_2(t, V_E, theta_E, Z, n_max, t_stim):
    """
    Integration of P_small(x,t) with x from 1,2
    """
    v = V_E
    mu = v*theta_E
    z = 2 * (Z + theta_E)/(2*theta_E) # z is between 0 and 2

    if t <= t_stim:
        return 0
    else:
        t = t - t_stim

    t /= (theta_E**2)

    result = 0
    
    sqrt_t = np.sqrt(t)
    
    for n in range(-n_max, n_max + 1):
        term1 = np.exp(4 * mu * n) * (
            Phi((2 - (z + 4 * n + mu * t)) / sqrt_t) -
            Phi((1 - (z + 4 * n + mu * t)) / sqrt_t)
        )
        
        term2 = np.exp(2 * mu * (2 * (1 - n) - z)) * (
            Phi((2 - (-z + 4 * (1 - n) + mu * t)) / sqrt_t) -
            Phi((1 - (-z + 4 * (1 - n) + mu * t)) / sqrt_t)
        )
        
        result += term1 - term2
    
    return result


def P_large_t_btn_1_2(x1, x2, t, V_E, theta_E, Z, K_max, t_stim):
    """
    Integration of P_large(x,t) with x from 1,2
    """
    v = V_E
    mu = v*theta_E
    z = 2 * (Z + theta_E)/(2*theta_E) # z is between 0 and 2

    if t <= t_stim:
        return 0
    else:
        t = t - t_stim

    t /= (theta_E**2)

    result = 0
    
    # check if K_max is integer, else raise error
    if not isinstance(K_max, int):
        raise ValueError(f"K_max must be an integer.K_max={K_max}, type={type(K_max)}")
    
    for n in range(1, K_max + 1):
        delta_F = F(x2, mu, n) - F(x1, mu, n)
        term = delta_F * np.sin(n * np.pi * z / 2) * np.exp(-0.5 * (mu**2 + (n**2 * np.pi**2) / 4) * t)
        result += term
    
    result *= np.exp(-mu * z)
    
    return result


def S_E_fn(t, V_E, theta_E, Z, K_max):
    """
    Prob that EA survives till time 't'
    """
    v = V_E
    a = 2*theta_E
    mu = v*theta_E
    Z = a * (Z + theta_E)/(2*theta_E)

    term1 = (np.pi/4) * np.exp(-mu*Z - 0.5*(mu**2)*t)
    k_terms = np.arange(1, K_max + 1)
    
    sine_term = 2*k_terms*np.sin(k_terms * np.pi * Z /2)
    exp_term_1 = 2 * k_terms * np.exp(2*mu) * np.sin(k_terms * np.pi *(2-Z)/2)
    
    exp_term_num = np.exp(-(1/8)*(k_terms**2)*(np.pi**2)*t)
    exp_term_deno = 1/((mu**2) + ((k_terms**2)*(np.pi**2))/4)
    
    sum_term = np.sum((sine_term + exp_term_1) * exp_term_num * exp_term_deno)

    return term1 * sum_term


def P_x_large_t_fn(x, t, V_E, theta_E, Z, K_max):
    """
    Prob density that DV at x at time t for large times
    """
    v = V_E
    a = 2*theta_E
    mu = v*theta_E

    Z = a * (Z + theta_E)/(2*theta_E)

    term1 = np.exp(mu*x - mu*Z)
    k_terms = np.arange(1, K_max+1)
    sine_term_1 = np.sin(k_terms * np.pi * Z/2)
    sine_term_2 = np.sin(k_terms * np.pi * x/2)
    exp_term = np.exp(-0.5 * (mu**2 + 0.25*(k_terms**2)*np.pi**2) * t)
    
    sum_term = np.sum(sine_term_1 * sine_term_2 * exp_term)
    return term1 * sum_term

def P_x_small_t_fn(x, t, V_E, theta_E, Z, K_max):
    """
    Prob density that DV at x at time t for small times
    """
    v = V_E
    a = 2*theta_E
    mu = v*theta_E
    
    Z = a * (Z + theta_E)/(2*theta_E) # [-1,1] system to [0,2] system

    term1 = 1/(2*np.pi*t)**0.5    
    k_terms = np.linspace(-K_max, K_max, 2*K_max+1)
    exp_1 = np.exp(4*mu*k_terms - (((x - Z - 4*k_terms - mu*t)**2)/(2*t)))
    exp_2 = np.exp(2*mu*(2 - 2*k_terms - Z) - ((x + Z - 4 + 4*k_terms - mu*t)**2)/(2*t))
    exp_diff = exp_1 - exp_2

    return term1*np.sum(exp_diff)

def P_x_t_fn(x, t, V_E, theta_E, Z, K_max, t_stim):
    """
    Prob Density that DV is at x at time t
    """
    if t <= t_stim:
        return 0
    t = t - t_stim
    if t > 0.25:
        return P_x_large_t_fn(x, t, V_E, theta_E, Z, K_max)
    else:
        return P_x_small_t_fn(x, t, V_E, theta_E, Z, K_max)
    


def make_cdf(data):
    """
    Given data arr, return x axis, y axis for CDF
    """
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf

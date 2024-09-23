import numpy as np
from scipy import integrate
from scipy.integrate import quad
from scipy.special import erf, erfcx
from numba import jit

@jit
def simulate_psiam(V_A, theta_A, V_E, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, dt):
    AI = 0; DV = Z_E; t = 0; dB = dt**0.5
    is_act = 0
    while True:
        if t*dt > t_stim + t_E_aff:
            DV += V_E*dt + np.random.normal(0, dB)
        
        if t*dt > t_A_aff:
            AI += V_A*dt + np.random.normal(0, dB)
        
        t += 1
        
        if DV >= theta_E:
            choice = +1; RT = t*dt + t_motor
            break
        elif DV <= -theta_E:
            choice = -1; RT = t*dt + t_motor
            break
        
        if AI >= theta_A:
            is_act = 1
            AI_hit_time = t*dt
            if t*dt > t_stim:
                while t*dt <= (AI_hit_time + t_E_aff + t_motor):#  u can process evidence till stim plays
                    if t*dt > t_stim + t_E_aff: # Evid accum wil begin only after stim starts and afferent delay
                        DV += V_E*dt + np.random.normal(0, dB)
                        if DV >= theta_E:
                            DV = theta_E
                            break
                        elif DV <= -theta_E:
                            DV = -theta_E
                            break
                    t += 1
            
            break
        
        
    if is_act == 1:
        RT = AI_hit_time + t_motor
        if DV > 0:
            choice = 1
        elif DV < 0:
            choice = -1
        else: # if DV is 0 because stim has not yet been played, then choose right/left randomly
            randomly_choose_up = np.random.rand() >= 0.5
            if randomly_choose_up:
                choice = 1
            else:
                choice = -1       
    
    return choice, RT, is_act


def P_small_t_btn_x1_x2(x1, x2, t, V_E, theta_E, Z, n_max, t_stim, t_E_aff, t_motor):
    """
    Integration of P_small(x,t) btn x1 and x2
    """
    v = V_E
    mu = v*theta_E
    z = 2 * (Z + theta_E)/(2*theta_E) # z is between 0 and 2
    # t -= t_stim # t subtraction wil be done outside the func

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


def rho_A_t_fn(t, V_A, theta_A, t_A_aff, t_motor):
    """
    For AI,prob density of t given V_A, theta_A
    """
    # t -= t_A_aff + t_motor # t subtraction wil be done outside the func
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
    # t -= t_stim # t subtraction wil be done outside the func

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


def Phi(x):
    """
    Define the normal cumulative distribution function Φ(x) using erf
    """
    return 0.5 * (1 + erf(x / np.sqrt(2)))

def cum_A_t_fn(t, V_A, theta_A, t_A_aff, t_motor):
    """
    For AI, calculate cummulative distrn of a time t given V_A, theta_A
    """
    # t -= t_A_aff + t_motor # t subtraction wil be done outside the func

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


def phi(x):
    """Standard Gaussian function."""
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

def M(x):
    """Mills ratio."""
    return np.sqrt(np.pi / 2) * erfcx(x / np.sqrt(2))

def CDF_E_minus_small_t_NORM_fn(t, V_E, theta_E, K_max, t_stim, Z_E, t_E_aff, t_motor):
    """
    In normalized time, CDF of hitting the lower bound.
    """
    v = V_E * theta_E
    w = (Z_E + theta_E) / (2 * theta_E)
    a = 2
    
    # t subtraction will be done outside the function
    if t <= 0:
        return 0
    
    t /= theta_E**2
    result = np.exp(-v * a * w - (((v**2) * t) / 2))

    summation = 0
    for k in range(K_max + 1):
        if k % 2 == 0:  # even k
            r_k = k * a + a * w
        else:  # odd k
            r_k = k * a + a * (1 - w)
        
        term1 = phi((r_k) / np.sqrt(t))
        term2 = M((r_k - v * t) / np.sqrt(t)) + M((r_k + v * t) / np.sqrt(t))
        
        summation += ((-1)**k) * term1 * term2

    return (result*summation)


def all_RTs_fit_fn(t_pts, V_A, theta_A, V_E, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor):
    """
    PDF of all RTs array irrespective of choice
    """
    K_max = 10

    P_A = [rho_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A, t_A_aff, t_motor) for t in t_pts]# if AI hit
    C_E = [CDF_E_minus_small_t_NORM_fn(t-t_stim, -V_E, theta_E, K_max, t_stim, -Z_E, t_E_aff, t_motor) \
           + CDF_E_minus_small_t_NORM_fn(t-t_stim, V_E, theta_E, K_max, t_stim, Z_E, t_E_aff, t_motor) for t in t_pts]
    P_E_cum = np.zeros(len(t_pts))
    for i,t in enumerate(t_pts):
        t1 = t - t_motor - t_stim - t_E_aff
        t2 = t - t_stim
        if t1 < 0:
            t1 = 0
        P_E_cum[i] = CDF_E_minus_small_t_NORM_fn(t2, -V_E, theta_E, K_max, t_stim, -Z_E, t_E_aff, t_motor) \
                    + CDF_E_minus_small_t_NORM_fn(t2, V_E, theta_E, K_max, t_stim, Z_E, t_E_aff, t_motor) \
                    - CDF_E_minus_small_t_NORM_fn(t1, -V_E, theta_E, K_max, t_stim, -Z_E, t_E_aff, t_motor) \
                    - CDF_E_minus_small_t_NORM_fn(t1, V_E, theta_E, K_max, t_stim, Z_E, t_E_aff, t_motor)


    P_E = [rho_E_t_fn(t-t_E_aff-t_stim-t_motor, V_E, theta_E, K_max, t_stim, Z_E, t_E_aff, t_motor) for t in t_pts]
    C_A = [cum_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A, t_A_aff, t_motor) for t in t_pts]

    P_A = np.array(P_A); C_E = np.array(C_E); P_E = np.array(P_E); C_A = np.array(C_A)
    P_all = P_A*((1-C_E)+P_E_cum) + P_E*(1-C_A)

    return P_all

def up_RTs_fit_fn(t_pts, V_A, theta_A, V_E, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor):
    """
    PDF of up RTs array
    """
    K_max = 10

    P_A = [rho_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A, t_A_aff, t_motor) for t in t_pts]
    P_EA_btn_1_2 = [P_small_t_btn_x1_x2(1, 2, t-t_stim, V_E, theta_E, Z_E, K_max, t_stim, t_E_aff, t_motor) for t in t_pts]
    P_E_plus_cum = np.zeros(len(t_pts))
    for i,t in enumerate(t_pts):
        t1 = t - t_motor - t_stim - t_E_aff
        t2 = t - t_stim
        if t1 < 0:
            t1 = 0
        P_E_plus_cum[i] = CDF_E_minus_small_t_NORM_fn(t2, -V_E, theta_E, K_max, t_stim, -Z_E, t_E_aff, t_motor) - CDF_E_minus_small_t_NORM_fn(t1, -V_E, theta_E, K_max, t_stim, -Z_E, t_E_aff, t_motor)


    P_E_plus = [rho_E_minus_small_t_NORM_fn(t-t_stim-t_E_aff-t_motor, -V_E, theta_E, K_max, t_stim, -Z_E, t_E_aff, t_motor) for t in t_pts]
    C_A = [cum_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A, t_A_aff, t_motor) for t in t_pts]

    P_A = np.array(P_A); P_EA_btn_1_2 = np.array(P_EA_btn_1_2); P_E_plus = np.array(P_E_plus); C_A = np.array(C_A)
    P_correct_unnorm = (P_A*(P_EA_btn_1_2 + P_E_plus_cum) + P_E_plus*(1-C_A))
    return P_correct_unnorm


def down_RTs_fit_fn(t_pts, V_A, theta_A, V_E, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor):
    """
    PDF of down RTs array
    """
    K_max = 10
        
    P_A = [rho_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A, t_A_aff, t_motor) for t in t_pts]
    P_EA_btn_0_1 = [P_small_t_btn_x1_x2(0, 1, t-t_stim, V_E, theta_E, Z_E, K_max, t_stim, t_E_aff, t_motor) for t in t_pts]
    P_E_minus_cum = np.zeros(len(t_pts))
    for i,t in enumerate(t_pts):
        t1 = t - t_motor - t_stim - t_E_aff
        t2 = t - t_stim
        if t1 < 0:
            t1 = 0
        P_E_minus_cum[i] = CDF_E_minus_small_t_NORM_fn(t2, V_E, theta_E, K_max, t_stim, Z_E, t_E_aff, t_motor) - CDF_E_minus_small_t_NORM_fn(t1, V_E, theta_E, K_max, t_stim, Z_E, t_E_aff, t_motor)


    P_E_minus = [rho_E_minus_small_t_NORM_fn(t-t_stim-t_E_aff-t_motor, V_E, theta_E, K_max, t_stim, Z_E, t_E_aff, t_motor) for t in t_pts]
    C_A = [cum_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A, t_A_aff, t_motor) for t in t_pts]

    P_A = np.array(P_A); P_EA_btn_0_1 = np.array(P_EA_btn_0_1); P_E_minus = np.array(P_E_minus); C_A = np.array(C_A)
    P_wrong_unnorm = (P_A*(P_EA_btn_0_1+P_E_minus_cum) + P_E_minus*(1-C_A))
    return P_wrong_unnorm

def correct_RT_loglike_fn(t, V_A, theta_A, V_E, theta_E, Z_E, K_max, t_A_aff, t_E_aff, t_stim, t_motor):
    """
    log likelihood of correct RT
    """
    P_A = rho_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A, t_A_aff, t_motor)
    P_EA_btn_1_2 = P_small_t_btn_x1_x2(1, 2, t-t_stim, V_E, theta_E, Z_E, K_max, t_stim, t_E_aff, t_motor)
    t1 = t - t_motor - t_stim - t_E_aff
    t2 = t - t_stim
    if t1 < 0:
        t1 = 0
    # P_E_plus_cum = quad(rho_E_minus_small_t_NORM_fn, t1, t2, args=(-V_E, theta_E, K_max, t_stim, -Z_E, t_E_aff, t_motor))[0]
    P_E_plus_cum = CDF_E_minus_small_t_NORM_fn(t2, -V_E, theta_E, K_max, t_stim, -Z_E, t_E_aff, t_motor) \
                 - CDF_E_minus_small_t_NORM_fn(t1, -V_E, theta_E, K_max, t_stim, -Z_E, t_E_aff, t_motor)

    P_E_plus = rho_E_minus_small_t_NORM_fn(t-t_stim-t_E_aff-t_motor, -V_E, theta_E, K_max, t_stim, -Z_E, t_E_aff, t_motor)
    C_A = cum_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A, t_A_aff, t_motor)

    P_correct = (P_A*(P_EA_btn_1_2 + P_E_plus_cum) + P_E_plus*(1-C_A))

    if P_correct <= 0:
        P_correct = 1e-16

    return np.log(P_correct)


def wrong_RT_loglike_fn(t, V_A, theta_A, V_E, theta_E, Z_E, K_max, t_A_aff, t_E_aff, t_stim, t_motor):
    """
    log likelihood of wrong RT
    """
    P_A = rho_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A, t_A_aff, t_motor)
    P_EA_btn_0_1 = P_small_t_btn_x1_x2(0, 1, t-t_stim, V_E, theta_E, Z_E, K_max, t_stim, t_E_aff, t_motor)
    t1 = t - t_motor - t_stim - t_E_aff
    t2 = t - t_stim
    if t1 < 0:
        t1 = 0
    # P_E_minus_cum = quad(rho_E_minus_small_t_NORM_fn, t1, t2, args=(V_E, theta_E, K_max, t_stim, Z_E, t_E_aff, t_motor))[0]
    P_E_minus_cum = CDF_E_minus_small_t_NORM_fn(t2, V_E, theta_E, K_max, t_stim, Z_E, t_E_aff, t_motor) \
                  - CDF_E_minus_small_t_NORM_fn(t1, V_E, theta_E, K_max, t_stim, Z_E, t_E_aff, t_motor)


    P_E_minus = rho_E_minus_small_t_NORM_fn(t-t_stim-t_E_aff-t_motor, V_E, theta_E, K_max, t_stim, Z_E, t_E_aff, t_motor)
    C_A = cum_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A, t_A_aff, t_motor)

    P_wrong = (P_A*(P_EA_btn_0_1+P_E_minus_cum) + P_E_minus*(1-C_A))


    if P_wrong <= 0:
        P_wrong = 1e-16
    
    return np.log(P_wrong)


def abort_RT_loglike_fn(t, V_A, theta_A, V_E, theta_E, Z_E, K_max, t_A_aff, t_E_aff, t_stim, t_motor):
    """
    log likelihood of abort RT
    """
    P_A = rho_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A, t_A_aff, t_motor)
    P_abort = P_A
    if P_abort <= 0:
        P_abort = 1e-16

    return np.log(P_abort)



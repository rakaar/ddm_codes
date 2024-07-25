import numpy as np
import pymc as pm
import math


def rtd_density_a_NUTS(t, v, a, w, K_max=10):
    if t > 0.25:
        non_sum_term = (math.pi/a**2)*pm.math.exp(-v*a*w - (v**2 * t/2))
        k_vals = np.linspace(1, K_max, K_max)
        sum_sine_term = pm.math.sin(k_vals*np.pi*w)
        sum_exp_term = pm.math.exp(-(k_vals**2 * np.pi**2 * t)/(2*a**2))
        sum_result = pm.math.sum(k_vals * sum_sine_term * sum_exp_term)
    else:
        non_sum_term = (1/a**2)*(a**3/pm.math.sqrt(2*math.pi*t**3))*pm.math.exp(-v*a*w - (v**2 * t)/2)
        K_max = int(K_max/2)
        k_vals = np.linspace(-K_max, K_max, 2*K_max + 1)
        sum_w_term = w + 2*k_vals
        sum_exp_term = pm.math.exp(-(a**2 * (w + 2*k_vals)**2)/(2*t))
        sum_result = pm.math.sum(sum_w_term*sum_exp_term)

    # print('This is possible ? ',type(non_sum_term), type(sum_result))
    density =  non_sum_term * sum_result
    # if density <= 0:
    #     density = 1e-10
    return density


def loglike_fn_NUTS(v,a,w,data):
    choices = np.array(data['choices'])
    RTs = np.array(data['RTs'])

    choices_pos = np.where(choices == 1)[0]
    choices_neg = np.where(choices == -1)[0]

    RTs_pos = RTs[choices_pos]
    RTs_neg = RTs[choices_neg]

    prob_pos = [rtd_density_a_NUTS(t, -v, a, 1-w) for t in RTs_pos]
    prob_neg = [rtd_density_a_NUTS(t, v, a, w) for t in RTs_neg]
    # prob_pos = np.array(prob_pos)
    # prob_neg = np.array(prob_neg)

    # prob_pos[prob_pos <= 0] = 1e-10
    # prob_neg[prob_neg <= 0] = 1e-10

    log_pos = pm.math.log(prob_pos)
    log_neg = pm.math.log(prob_neg)
    
    
    sum_loglike = (pm.math.sum(log_pos) + pm.math.sum(log_neg))
    return sum_loglike



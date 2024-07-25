import pickle
import numpy as np
from numba import jit
from joblib import Parallel, delayed
from pybads import BADS

def bads_vs_N_2param(lb,ub,plb,pub,N_sample,v,a):
    N_iter = 100
    results = Parallel(n_jobs=-1)(delayed(run_bads_2p)(lb, ub, plb, pub) for _ in range(N_iter))
    return results

def bads_target_neg_loglike_2p(params):
    v, a = params
    w = 0.5
    with open('sample_rt.pkl', 'rb') as f:
        RTs = np.array(pickle.load(f))
    with open('sample_choice.pkl', 'rb') as f:
        choices = np.array(pickle.load(f))

    choices_pos = np.where(choices == 1)[0]
    choices_neg = np.where(choices == -1)[0]

    RTs_pos = RTs[choices_pos]
    RTs_neg = RTs[choices_neg]

    prob_pos = Parallel(n_jobs=-1)(delayed(rtd_density_a)(t, -v, a, 1-w) for t in RTs_pos)
    prob_neg = Parallel(n_jobs=-1)(delayed(rtd_density_a)(t, v, a, w) for t in RTs_neg)

    prob_pos = np.array(prob_pos)
    prob_neg = np.array(prob_neg)

    

    prob_pos[prob_pos <= 0] = 1e-10
    prob_neg[prob_neg <= 0] = 1e-10

    log_pos = np.log(prob_pos)
    log_neg = np.log(prob_neg)
    
    if np.isnan(log_pos).any() or np.isnan(log_neg).any():
        print('log_neg',log_neg)
        print('prob_neg = ', prob_neg)
        raise ValueError("NaN values found in log_pos or log_neg")

    obj = -(np.sum(log_pos) + np.sum(log_neg))
    # print(f'v={v},a={a},w={w},obj={obj}')
    return obj

def run_bads_2p(lb, ub, plb, pub):
    v0 = np.random.uniform(plb[0], pub[0])
    a0 = np.random.uniform(plb[1], pub[1])
    x0 = np.array([v0, a0]);

    options = {'display': 'off'}
    
    try:
        bads = BADS(bads_target_neg_loglike_2p, x0, lb, ub, plb, pub, options=options)
        optimize_result = bads.optimize()
        x_min = optimize_result['x']
        return x_min
    except Exception as e:
        print(f"Error during optimization: {e}, running again")
        run_bads_2p(lb, ub, plb, pub)


@jit(nopython=True)
def rtd_density_a(t, v, a, w, K_max=5):
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
        density += 1e-6
    return density
import numpy as np
from numba import jit, njit
from scipy import integrate
import pickle
from pybads import BADS
import re
import matplotlib.pyplot as plt
from joblib import Parallel, delayed



def rtd_mu_large_t(t,mu, K_max=50):
    non_sum_term = (np.pi/2) * np.cosh(mu) * np.exp(-(mu**2)*t/2)
    k_vals = np.linspace(0, K_max, K_max + 1)
    sum_neg_1_term = (-1)**k_vals
    sum_two_k_term = (2*k_vals) + 1
    sum_exp_term = np.exp(-(((2*k_vals + 1)**2)*(np.pi**2)*t )/(8))
    sum_term = np.sum(sum_neg_1_term*sum_two_k_term*sum_exp_term)
    return non_sum_term*sum_term

def rtd_mu_small_t(t, mu, K_max=50):
    non_sum_term = 2 * np.cosh(mu) * np.exp(-(mu**2)*t/2) * (1/np.sqrt(2*np.pi*(t**3)))
    k_vals = np.linspace(0, K_max, K_max + 1)
    sum_neg_1_term = (-1)**k_vals
    sum_two_k_term = (2*k_vals) + 1
    sum_exp_term = np.exp(-((2*k_vals + 1)**2)/(2*t))
    sum_term = np.sum(sum_neg_1_term*sum_two_k_term*sum_exp_term)
    return non_sum_term*sum_term


def rtd_mu(t,mu, smol_to_large_trans_fac):
    if t > smol_to_large_trans_fac:
        return rtd_mu_large_t(t, mu)
    else:
        return rtd_mu_small_t(t, mu)

def prob_rt_mu(t_arr, mu, smol_to_large_trans_fac):
    N_t = len(t_arr)
    prob_arr = np.zeros((N_t-1,1))
    for i in range(0, N_t-1):
        prob_arr[i] = integrate.quad(rtd_mu, t_arr[i], t_arr[i+1], args=(mu, smol_to_large_trans_fac))[0]
    return prob_arr

# def prob_rt_mu_posterior(t_arr, mu, smol_to_large_trans_fac):
#     N_t = len(t_arr)
#     prob_arr = np.zeros((N_t-1,1))
#     for i in range(0, N_t-1):

#         prob_arr[i] = integrate.quad(rtd_mu, t_arr[i]-0.0001, t_arr[i]+0.0001, args=(mu, smol_to_large_trans_fac))[0]
#     return prob_arr

def prob_rt_mu_posterior(t_arr, mu, smol_to_large_trans_fac):
    N_t = len(t_arr)
    def compute_prob(i):
        return integrate.quad(rtd_mu, t_arr[i]-0.0001, t_arr[i]+0.0001, args=(mu, smol_to_large_trans_fac))[0]
    
    prob_arr = Parallel(n_jobs=-1)(delayed(compute_prob)(i) for i in range(0, N_t-1))
    prob_arr = np.array(prob_arr).reshape(-1, 1)

    return prob_arr

def prob_rt_generic(t_arr, mu, func):
    N_t = len(t_arr)
    prob_arr = np.zeros((N_t-1,1))
    for i in range(0, N_t-1):
        prob_arr[i] = integrate.quad(func, t_arr[i], t_arr[i+1], args=(mu))[0]
    return prob_arr

def get_param_N_bads_tensor(N_sample_sizes, a, v, N_bads_run):
    param_N_bads = np.zeros((len(N_sample_sizes), N_bads_run, 3))
    for i,n in enumerate(N_sample_sizes):
        fname = f'vaw_bads_vals_v{v}_a{a}_n{n}.pkl'
        try:
            with open(fname, 'rb') as f:
                vaw_50 = pickle.load(f)
        except Exception as e:
            print(f"File {fname} is empty. Skipping.")
            continue
        param_N_bads[i] = vaw_50
    
    return param_N_bads

def run_bads_3param():
    # Large range
    lower_bounds = np.array([0.01, 5, 0.1]) 
    upper_bounds = np.array([10, 15, 0.9])

    plausible_lower_bounds = np.array([0.5, 7, 0.2])
    plausible_upper_bounds = np.array([7, 13, 0.8])

    # narrow range: v = 4, a = 2: v,a,w
    # lower_bounds = np.array([0.1, 0.1, 0.1]) 
    # upper_bounds = np.array([7, 5, 0.9])

    # plausible_lower_bounds = np.array([0.5, 1, 0.3])
    # plausible_upper_bounds = np.array([5, 4, 0.7])

    # narrow range: a = 10 v = 2
    # lower_bounds = np.array([0.5, 8, 0.2]) 
    # upper_bounds = np.array([5, 12, 0.8])

    # plausible_lower_bounds = np.array([1, 9, 0.3])
    # plausible_upper_bounds = np.array([4, 11, 0.7])

    v0 = np.random.uniform(plausible_lower_bounds[0], plausible_upper_bounds[0])
    a0 = np.random.uniform(plausible_lower_bounds[1], plausible_upper_bounds[1])
    w0 = np.random.uniform(plausible_lower_bounds[2], plausible_upper_bounds[2])
    x0 = np.array([v0, a0, w0]);

    options = {'display': 'off'}
    
    try:
        bads = BADS(bads_target_func, x0, lower_bounds, upper_bounds, plausible_lower_bounds, plausible_upper_bounds, options=options)
        optimize_result = bads.optimize()
        x_min = optimize_result['x']
        return x_min
    except Exception as e:
        print(f"Error during optimization: {e}")
        run_bads_3param()
    


def bads_target_func(params):
    v, a, w = params
    with open('sample_rt.pkl', 'rb') as f:
        RTs = pickle.load(f)
    probs = np.array([rtd_density_a(t, v, a, w) + rtd_density_a(t, -v, a, 1-w) for t in RTs])
    return -np.sum(np.log(probs))

def ex22_bads_vs_N_plots(N_sample_sizes, param_names, ground_truths, param_N_bads):
    N_sample_sizes_str = [str(size) for size in N_sample_sizes]
    indices = range(len(N_sample_sizes))  # Use indices for equally spaced x-ticks

    plt.figure(figsize=(15, 5))
    for i in range(3):
        N_bads_val = param_N_bads[:, :, i]
        plt.subplot(1, 3, i + 1)
        plt.errorbar(indices, np.mean(N_bads_val, axis=1), yerr=np.std(N_bads_val, axis=1) / np.sqrt(50), fmt='o-', capsize=5)
        plt.axhline(y=ground_truths[i], color='r', linestyle='-', label=f'Ground Truth: {ground_truths[i]}')
        plt.xlabel('Sample Size')
        plt.ylabel(f'Mean of {i+1}th Parameter')
        plt.title(f'Mean of {i+1}th Parameter vs Sample Size {param_names[i]}')
        plt.xticks(indices, N_sample_sizes_str)  # Setting x-tick labels to string representations of N_sample_sizes

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 5))
    for i in range(3):
        N_bads_val = param_N_bads[:, :, i]

        median_vals = np.median(N_bads_val, axis=1)
        mad_vals = np.median(np.abs(N_bads_val - np.median(N_bads_val, axis=1, keepdims=True)), axis=1)

        plt.subplot(1, 3, i + 1)
        plt.errorbar(indices, median_vals, yerr=mad_vals, fmt='o-', capsize=5)
        plt.axhline(y=ground_truths[i], color='r', linestyle='-', label=f'Ground Truth: {ground_truths[i]}')
        plt.xlabel('Sample Size')
        plt.ylabel(f'Median of {i+1}th Parameter')
        plt.title(f'Median of {i+1}th Parameter vs Sample Size {param_names[i]}')
        plt.xticks(indices, N_sample_sizes_str)  # Setting x-tick labels to string representations of N_sample_sizes

    plt.tight_layout()
    plt.show()

def ex22_bads_plots(filename):
    match = re.search(r'v(\d+)_a(\d+)', filename)
    v = int(match.group(1))
    a = int(match.group(2))

    with open(filename, 'rb') as f:
        vaw_bads_vals = pickle.load(f)



    v_vals_from_bads = [x[0] for x in vaw_bads_vals]
    a_vals_from_bads = [x[1] for x in vaw_bads_vals]
    w_vals_from_bads = [x[2] for x in vaw_bads_vals]

    plt.figure(figsize=(10,5))
    plt.subplot(1,3,1)
    plt.hist(v_vals_from_bads)
    plt.title(f'v: mean={np.mean(v_vals_from_bads):.2f},median={np.median(v_vals_from_bads):.2f},Truth = {v}', fontsize=8)
    plt.subplot(1,3,2)
    plt.hist(a_vals_from_bads)
    plt.title(f'a: mean={np.mean(a_vals_from_bads):.2f},median={np.median(a_vals_from_bads):.2f}, ground Truth = {a}', fontsize=8)
    plt.subplot(1,3,3)
    plt.hist(w_vals_from_bads)
    plt.title(f'w: mean={np.mean(w_vals_from_bads):.2f},median={np.median(w_vals_from_bads):.2f},Truth = 0.5',fontsize=8)
    plt.tight_layout()
    plt.show()



    plt.figure(figsize=(10,5))
    plt.subplot(1,3,1)
    plt.scatter(a_vals_from_bads, v_vals_from_bads)
    plt.title('a vs v')
    plt.xlabel('a')
    plt.ylabel('v')
    plt.subplot(1,3,2)
    plt.scatter(w_vals_from_bads, a_vals_from_bads)
    plt.title('w vs a')
    plt.xlabel('w')
    plt.ylabel('a')
    plt.subplot(1,3,3)
    plt.scatter(v_vals_from_bads, w_vals_from_bads)
    plt.title('v vs w')
    plt.xlabel('v')
    plt.ylabel('w')
    plt.tight_layout()
    plt.show()


@jit(nopython=True)
def rtd_density(t, mu, K_max=50):
    k_vals = np.linspace(0, K_max, K_max+1)
    sum_neg_1_term = (-1)**k_vals
    sum_two_k_term = (2*k_vals) + 1

    if t > 0.25:
        non_sum_term = np.pi/2 * np.cosh(mu) * np.exp(-(mu**2)*t/2)
        sum_exp_term = np.exp(-(((2*k_vals + 1)**2) * (np.pi**2) * t)/8)
    else:
        non_sum_term = 2*np.cosh(mu)*np.exp(-(mu**2)*t/2)*(1/np.sqrt(2 * np.pi * t**3))
        sum_exp_term = np.exp(-((2*k_vals + 1)**2)/(2*t))

    sum_term = np.sum(sum_neg_1_term*sum_two_k_term*sum_exp_term)

    return non_sum_term*sum_term
        

def prob_rt(t_arr, v):
    N_t = len(t_arr)
    prob_arr = np.zeros((N_t-1,1))
    for i in range(0, N_t-1):
        prob_arr[i] = integrate.quad(rtd_density, t_arr[i], t_arr[i+1], args=(v))[0]
    return prob_arr



def calculate_histogram(x_axis, y_axis):
    x_axis = np.sort(x_axis)
    histcounts, _ = np.histogram(y_axis, bins=x_axis)
    prob = histcounts/np.sum(histcounts)
    return prob

def parse_sim_results(results):
    choices =  [r[0] for r in results]
    rts = [r[1] for r in results]
    return choices, rts


@njit
def simulate_ddm(v, a, dt=1e-5):
    # assume starting pt = 0, sigma = 1, boundaries -a, +a
    DV = 0; t = 0; dB = dt**0.5 
    while True:
        DV += v*dt + np.random.normal(0, dB)
        t += 1

        if DV >= a/2:
            return +1, t*dt
        elif DV <= -a/2:
            return -1, t*dt
        

@jit(nopython=True)
def rtd_density_a(t, v, a, w, K_max=50):
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

    if sum_result < 0:
        sum_result += 1e-3
    
    density =  non_sum_term * sum_result
    if density < 0:
        print('+++++++++++++++++++++++++++++')
        print("t:", t, "v:", v, "a:", a, "w:", w, "sum_result:", sum_result, "non_sum_term:", non_sum_term, "\n")
        print(' sine term = ', sum_sine_term)

        print('+++++++++++++++++++++++++++++++')
        raise ValueError("Density cannot be negative")
    else:
        return density

def prob_rt_a(t_arr,v,a,w):
    N_t = len(t_arr)
    prob_arr = np.zeros((N_t-1,1))
    for i in range(0, N_t-1):
        prob_arr[i] = integrate.quad(rtd_density_a, t_arr[i], t_arr[i+1], args=(v,a,w))[0]
    return prob_arr

def prob_hit_low_bound(v,a,w):
    if v == 0:
        return 1 - w
    else:
        return (1 - np.exp(-2*v*a*(1-w)))/(np.exp(2*v*w*a) - np.exp(-2*v*a*(1-w)))

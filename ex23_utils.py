import pickle
import numpy as np
from scipy import stats
from numba import jit
from joblib import Parallel, delayed
from pyvbmc import  VBMC
import matplotlib.pyplot as plt

def sim_data_to_files_N(v,a,n):
    with open('all_sim_results.pkl', 'rb') as f:
        all_sim_results = pickle.load(f)
    
    keyname = f"a={str(a)},v={str(v)}"
    choices, RTs = parse_sim_results(all_sim_results[keyname])
    
    RTs = np.array(RTs)
    choices = np.array(choices)

    selected_indices = np.random.choice(len(RTs), size=n, replace=False)

    selected_RTs = RTs[selected_indices]
    selected_choices = choices[selected_indices]

    with open('sample_rt_pos.pkl', 'wb') as f:
        pickle.dump(selected_RTs, f)
    with open('sample_choice_pos.pkl', 'wb') as f:
        pickle.dump(selected_choices, f)


def sim_data_to_files(v,a):
    with open('all_sim_results.pkl', 'rb') as f:
        all_sim_results = pickle.load(f)
    
    keyname = f"a={str(a)},v={str(v)}"
    choices, RTs = parse_sim_results(all_sim_results[keyname])
    
    with open('sample_rt_pos.pkl', 'wb') as f:
        pickle.dump(RTs, f)
    with open('sample_choice_pos.pkl', 'wb') as f:
        pickle.dump(choices, f)

def parse_sim_results(results):
    choices =  [r[0] for r in results]
    rts = [r[1] for r in results]
    return choices, rts


def save_prior_bounds(prior_bounds):
    with open('prior_bounds.pkl', 'wb') as f:
        pickle.dump(prior_bounds, f)

def log_prior(params):
    v,a,w = params
    with open('prior_bounds.pkl', 'rb') as f:
        prior_bounds = pickle.load(f)

    # declare distributions
    v_prior = stats.uniform(loc=prior_bounds['v_low'], scale=prior_bounds['v_high'] - prior_bounds['v_low'])
    a_prior = stats.uniform(loc=prior_bounds['a_low'], scale=prior_bounds['a_high'] - prior_bounds['a_low'])
    w_prior = stats.uniform(loc=prior_bounds['w_low'], scale=prior_bounds['w_high'] - prior_bounds['w_low'])

    
    # pdf
    log_prior_v = v_prior.logpdf(v)
    log_prior_a = a_prior.logpdf(a)
    log_prior_w = w_prior.logpdf(w)

    sum_log_priors = log_prior_v + log_prior_a + log_prior_w

    # print(f"v_prior: {v_sample}, a: {a_sample}, w: {w_sample}, log_prior_v: {log_prior_v}, log_prior_a: {log_prior_a}, log_prior_w: {log_prior_w}, sum_log_priors: {sum_log_priors}")
    return sum_log_priors
    

@jit(nopython=True)
def rtd_density_a(t, v, a, w, K_max=10):
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
        density = 1e-10
    return density

def loglike_fn(params):
    v, a, w = params
    with open('sample_rt_pos.pkl', 'rb') as f:
        RTs = np.array(pickle.load(f))
    with open('sample_choice_pos.pkl', 'rb') as f:
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

    sum_loglike = (np.sum(log_pos) + np.sum(log_neg))
    # print(f'v={v},a={a},w={w},sum_loglike={sum_loglike}')
    return sum_loglike

def log_joint(params):
    loglike = loglike_fn(params)
    logprior = log_prior(params)
    prior_plus_loglike = loglike + logprior
    # print(f"like={loglike}, prior={logprior}, joint={prior_plus_loglike}")
    return prior_plus_loglike


def run_vbmc(x0, lb, ub, plb, pub):
    options = {'display': 'off'}
    vbmc = VBMC(log_joint, x0, lb, ub, plb, pub, options=options)
    vp, results = vbmc.optimize()
    return vp, results


def vbmc_plots(vp):
    v_a_w, _ = vp.sample(int(3e5))
    v_post = v_a_w[:,0];a_post = v_a_w[:,1];w_post = v_a_w[:,2]

    plt.figure(figsize=(15, 5))
    plt.subplot(1,3,1)
    plt.hist(v_post);plt.title(f'V mean={np.mean(v_post):.2f}');
    plt.subplot(1,3,2)
    plt.hist(a_post);plt.title(f'a mean = {np.mean(a_post):.2f}');
    plt.subplot(1,3,3)
    plt.hist(w_post);plt.title(f'w mean = {np.mean(w_post):.2f}');

    plt.figure(figsize=(20, 5))
    plt.subplot(1,3,1)
    plt.scatter(v_post, a_post);plt.xlabel('v');plt.ylabel('a');
    plt.subplot(1,3,2)
    plt.scatter(v_post, w_post);plt.xlabel('v');plt.ylabel('w');
    plt.subplot(1,3,3)
    plt.scatter(w_post, a_post);plt.xlabel('w');plt.ylabel('a');


import pickle
import numpy as np
from numba import jit
from joblib import Parallel, delayed
from pybads import BADS
import matplotlib.pyplot as plt


def sim_data_to_files(v,a):
    with open('all_sim_results.pkl', 'rb') as f:
        all_sim_results = pickle.load(f)
    
    keyname = f"a={str(a)},v={str(v)}"
    choices, RTs = parse_sim_results(all_sim_results[keyname])
    
    with open('sample_rt.pkl', 'wb') as f:
        pickle.dump(RTs, f)
    with open('sample_choice.pkl', 'wb') as f:
        pickle.dump(choices, f)

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

    with open('sample_rt.pkl', 'wb') as f:
        pickle.dump(selected_RTs, f)
    with open('sample_choice.pkl', 'wb') as f:
        pickle.dump(selected_choices, f)

def plot_bads_vs_N_median(bads_sim_results):
    sample_sizes = list(bads_sim_results.keys())
    param_names = ['v', 'a', 'w']

    # Calculate statistics
    medians = {param: [] for param in param_names}
    mad = {param: [] for param in param_names}

    for size in sample_sizes:
        data = bads_sim_results[size]
        for i, param in enumerate(param_names):
            medians[param].append(np.median(data[:, i]))
            mad[param].append(np.median(np.abs(data[:, i] - np.median(data[:, i]))))

    # Convert sample sizes to strings for x-axis labels
    sample_size_labels = [str(size) for size in sample_sizes]
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    for i, param in enumerate(param_names):
        ax = axes[i]
        ax.errorbar(sample_size_labels, medians[param], yerr=mad[param], fmt='-o', label='Median ± MAD')
        
        ax.set_xticks(range(len(sample_size_labels)))
        ax.set_xticklabels(sample_size_labels)
        ax.set_xlabel('Sample Size')
        ax.set_ylabel(param)
        ax.set_title(f'{param} Median Statistics')
        ax.legend()

    plt.tight_layout()
    plt.show()

def parse_sim_results(results):
    choices =  [r[0] for r in results]
    rts = [r[1] for r in results]
    return choices, rts

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

def bads_target_neg_loglike(params):
    v, a, w = params
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

def run_bads(lb, ub, plb, pub):
    v0 = np.random.uniform(plb[0], pub[0])
    a0 = np.random.uniform(plb[1], pub[1])
    w0 = np.random.uniform(plb[2], pub[2])
    x0 = np.array([v0, a0, w0]);

    options = {'display': 'off'}
    
    try:
        bads = BADS(bads_target_neg_loglike, x0, lb, ub, plb, pub, options=options)
        optimize_result = bads.optimize()
        x_min = optimize_result['x']
        return x_min
    except Exception as e:
        print(f"Error during optimization: {e}, running again")
        run_bads(lb, ub, plb, pub)


def bads_vs_N(lb,ub,plb,pub,N_sample,v,a):
    N_iter = 100
    results = Parallel(n_jobs=-1)(delayed(run_bads)(lb, ub, plb, pub) for _ in range(N_iter))
    # results_array = np.array(results)

    # save_results_array = {'results': results, 'a': a, 'v': v, 'N_iter': N_iter}
    # with open(f'bads_v{v}_a{a}_n{N_sample}.pkl', 'wb') as f:
    #     pickle.dump(save_results_array, f)
    return results

def run_bads_N_iter(lb,ub,plb,pub,N_iter,v,a):
    results = Parallel(n_jobs=-1)(delayed(run_bads)(lb, ub, plb, pub) for _ in range(N_iter))
    results_array = np.array(results)

    save_results_array = {'results': results_array, 'a': a, 'v': v, 'N_iter': N_iter}
    with open(f'bads_v{v}_a{a}.pkl', 'wb') as f:
        pickle.dump(save_results_array, f)

    v_s = results_array[:,0]; a_s = results_array[:,1]; w_s = results_array[:,2]
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.hist(v_s, bins=20)
    plt.title(f'v, mean = {np.mean(v_s):.2f}, median = {np.median(v_s):.2f}, truth = {v}')
    plt.subplot(1,3,2)
    plt.hist(a_s, bins=20)
    plt.title(f'a, mean = {np.mean(a_s):.2f}, median = {np.median(a_s):.2f}, truth = {a}')
    plt.subplot(1,3,3)
    plt.hist(w_s, bins=20)
    plt.title(f'w, mean = {np.mean(w_s):.2f}, median = {np.median(w_s):.2f}, truth = 0.5')
    plt.show()


    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.scatter(v_s, w_s)
    plt.xlabel('v')
    plt.ylabel('w')
    plt.subplot(1,3,2)
    plt.scatter(w_s, a_s)
    plt.xlabel('w')
    plt.ylabel('a')
    plt.subplot(1,3,3)
    plt.scatter(a_s, v_s)
    plt.xlabel('a')
    plt.ylabel('v')
    plt.show()



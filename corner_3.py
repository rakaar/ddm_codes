from dynesty import NestedSampler
from dynesty import plotting as dyplot
from ddm_utils import simulate_ddm, parse_sim_results
from joblib import Parallel, delayed
import pickle
from dynesty_utiils import loglike_fn, prior_transform, _quantile

import matplotlib.pyplot as plt
import numpy as np
from ex23_utils import run_vbmc, save_prior_bounds
import multiprocessing
import os
import pymc as pm
import corner
from nuts_pymc_utils import loglike_fn_NUTS

num_processes = multiprocessing.cpu_count()

# save dynesty bounds
priors_bounds = { 
    'v': [-5,5],
    'a': [1, 3],
    'w': [0.3, 0.7] 
}
with open('dynesty_priors.pkl', 'wb') as f:
    pickle.dump(priors_bounds, f)

# save VBMC bounds
prior_bounds = { 'v_low': -5, 'v_high': 5, 'a_low': 1, 'a_high': 3, 'w_low': 0.3, 'w_high': 0.7 }
save_prior_bounds(prior_bounds)

# search bounds for VBMC
lb = np.array([-5, 1, 0.3]); ub = np.array([5, 3, 0.7])
plb = np.array([-4.9,1.1,0.31]); pub = np.array([4.9,2.9,0.69])

# data sim
N_trials_arr = [100, 500, 1000, 5000]
# N_trials_arr = [100]


corner_span = [(-0.2, 0.75), (1.6, 2.4), (0.4, 0.6)]
param_labels = ['v', 'a', 'w']
true_v = 0.2; true_a = 2; true_w = 0.5
true_arr = [true_v, true_a, true_w]

save_path = "/home/rka/Pictures" 

for N_sim in N_trials_arr:
    print(f'################################ processing {N_sim} ################################')

    # save data
    sim_results = Parallel(n_jobs=-1)(delayed(simulate_ddm)(true_v, true_a) for _ in range(N_sim))
    choices, RTs = parse_sim_results(sim_results)
    with open('sample_rt.pkl', 'wb') as f:
        pickle.dump(RTs, f)
    with open('sample_choice.pkl', 'wb') as f:
        pickle.dump(choices, f)
    data_dict = {'choices': choices, 'RTs': RTs}

    # Nested sampling
    pool = multiprocessing.Pool(processes=num_processes)
    ndim = 3
    sampler = NestedSampler(loglike_fn, prior_transform, ndim, pool=pool, queue_size=num_processes)
    sampler.run_nested()
    pool.close()
    pool.join()
    nested_samp_results = sampler.results

    # NS Corner plots
    title_kwargs = {'fontsize': 12}  # Increase the title font size
    label_kwargs = {'fontsize': 12}  # Increase the label font size
    tick_labelsize = 14

    fig_1, axes = dyplot.cornerplot(nested_samp_results, color='blue', truths=np.array(true_arr),
                                truth_color='black', show_titles=True,
                                max_n_ticks=5, labels=param_labels,
                                label_kwargs=label_kwargs,
                                title_kwargs=title_kwargs, span=corner_span)

    # Adjust tick parameters
    for ax in axes.reshape(-1):  # Flatten the axes array and iterate through each subplot
        ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
    filename_ns = f"corner_nested_sample_{N_sim}.png"
    fig_1.savefig(f"{save_path}/{filename_ns}")
    plt.close(fig_1)


    # VBMC
    x0 = np.array([np.random.uniform(plb[0], pub[0]), np.random.uniform(plb[1], pub[1]), np.random.uniform(plb[2], pub[2])])
    vp, results = run_vbmc(x0, lb, ub, plb, pub)

    v_a_w, _ = vp.sample(int(1e5))
    v_samp = v_a_w[:,0]; a_samp = v_a_w[:,1]; w_samp = v_a_w[:,2]
    combined_samples_vb = np.transpose(np.vstack((v_samp, a_samp, w_samp)))
    figure_2 = corner.corner(combined_samples_vb, labels=param_labels, show_titles=True, quantiles=[0.025, 0.5, 0.975],range=corner_span, truths=true_arr)
    filename_vb = f"corner_vbmc_{N_sim}.png"  # You can change the filename as needed
    figure_2.savefig(f"{save_path}/{filename_vb}")
    plt.close(figure_2)  # Close the figure to free 
    
    # NUTS
    with pm.Model() as model:
        v_ = pm.Uniform('v', lower=-5, upper=5)
        a_ = pm.Uniform('a', lower=1, upper=3)
        w_ = pm.Uniform('w', lower=0.3, upper=0.7)

        likelihood = pm.Potential('likelihood', loglike_fn_NUTS(v_,a_,w_, data_dict))

        trace = pm.sample(25000, step = pm.NUTS())
        

    v_samples = trace.posterior['v'].values.flatten()
    a_samples = trace.posterior['a'].values.flatten()
    w_samples = trace.posterior['w'].values.flatten()

    combined_samples_nuts = np.transpose(np.vstack((v_samples, a_samples, w_samples)))



    figure_3 = corner.corner(combined_samples_nuts, labels=param_labels, show_titles=True, quantiles=[0.025, 0.5, 0.975],range=corner_span, truths=true_arr)
    filename_nuts = f"corner_nuts_{N_sim}.png"  # You can change the filename as needed
    figure_3.savefig(f"{save_path}/{filename_nuts}")
    plt.close(figure_3)  # Close the figure to free 

    
    
    

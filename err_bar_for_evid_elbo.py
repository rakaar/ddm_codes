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

evid_elbo_dict = {}
for N_sim in N_trials_arr:
    print(f'################################ processing {N_sim} ################################')
    evid_elbo_dict[N_sim] = {}
    # save data
    v = 0.2; a = 2
    sim_results = Parallel(n_jobs=-1)(delayed(simulate_ddm)(v, a) for _ in range(N_sim))


    choices, RTs = parse_sim_results(sim_results)
        
    with open('sample_rt.pkl', 'wb') as f:
        pickle.dump(RTs, f)
    with open('sample_choice.pkl', 'wb') as f:
        pickle.dump(choices, f)

    # Nested sampling 5x
    evid5 = []
    for _ in range(5):
        pool = multiprocessing.Pool(processes=num_processes)
        ndim = 3
        sampler = NestedSampler(loglike_fn, prior_transform, ndim, pool=pool, queue_size=num_processes)
        sampler.run_nested()

        pool.close()
        pool.join()

        evid5.append(sampler.results.logz[-1])

    evid_elbo_dict[N_sim]['evid'] = evid5

    # VBMC 5x
    elbo5 = []
    for _ in range(5):
        x0 = np.array([np.random.uniform(plb[0], pub[0]), np.random.uniform(plb[1], pub[1]), np.random.uniform(plb[2], pub[2])])
        vp, results = run_vbmc(x0, lb, ub, plb, pub)
        elbo5.append(results['elbo'])

    evid_elbo_dict[N_sim]['elbo'] = elbo5


with open('evid_elbo_dict.pkl', 'wb') as f:
    pickle.dump(evid_elbo_dict, f)
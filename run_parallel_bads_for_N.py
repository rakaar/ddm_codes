import pickle
from ddm_utils import parse_sim_results, run_bads_3param
import numpy as np
from joblib import Parallel, delayed

N_sample_sizes = [500, 1000, 5000, 10000, 25000, 35000, 50000]



with open('all_sim_results.pkl', 'rb') as f:
    all_sim_results = pickle.load(f)

a = 10; v = 2

for n in N_sample_sizes:
    print('Sample size = ',n)
    choices, RTs = parse_sim_results(all_sim_results[f'a={a},v={v}'])
    RTs = np.random.choice(RTs, size=n, replace=False)
    with open('sample_rt.pkl', 'wb') as f:
        pickle.dump(RTs, f)

    print('Starting BADS')
    N_bads_run = 100
    vaw_bads_vals = np.zeros((N_bads_run,3))
    vaw_bads_vals = Parallel(n_jobs=-1)(delayed(run_bads_3param)() for _ in range(N_bads_run))

    fname = f'vaw_bads_vals_v{v}_a{a}_n{n}.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(vaw_bads_vals, f)
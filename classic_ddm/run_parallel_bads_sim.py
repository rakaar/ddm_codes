import pickle
from ddm_utils import parse_sim_results, run_bads_3param
import numpy as np
from joblib import Parallel, delayed

av_combos = [(10,2)]
# av_combos = [(2,4)]

with open('all_sim_results.pkl', 'rb') as f:
    all_sim_results = pickle.load(f)

for iii in range(1):
    print('Combo = ', av_combos[iii])
    

    a = av_combos[iii][0]; v = av_combos[iii][1]
    choices, RTs = parse_sim_results(all_sim_results[f'a={a},v={v}'])

    with open('sample_rt.pkl', 'wb') as f:
        pickle.dump(RTs, f)

    print('Starting BADS')
    N_bads_run = 100
    vaw_bads_vals = np.zeros((N_bads_run,3))
    vaw_bads_vals = Parallel(n_jobs=-1)(delayed(run_bads_3param)() for _ in range(N_bads_run))

    fname = f'vaw_bads_vals_v{v}_a{a}.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(vaw_bads_vals, f)


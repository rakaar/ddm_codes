from numba import jit
import numpy as np
from joblib import Parallel, delayed
import pickle


N_sim = 0
user_input = input("Enter the value for N_sim (1 for 10K, 2 for 100k, 3 for 1 Million): ")
if user_input == '1':
    N_sim = 10000
elif user_input == '2':
    N_sim = 100000
elif user_input == '3':
    N_sim = 1000000
else:
    print("Invalid input. Using default value of N_sim = 10000")
    N_sim = 10000


@jit
def simulate_psiam(V_A, theta_A, V_E, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor):
    AI = 0; DV = Z_E; t = 0; dt = 1e-6; dB = dt**0.5
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
            if t%2 == 1:
                choice = 1
            else:
                choice = -1       
    
    return choice, RT, is_act



print('Starting Simulation')

V_A = 1; theta_A = 2; 
V_E = 1.2; theta_E = 2
Z_E = 0
t_stim = 0.5
t_A_aff = 20e-3
t_E_aff = 30e-3
t_motor = 50e-3


choices = np.zeros((N_sim, 1)); RTs = np.zeros((N_sim, 1)); is_act_resp = np.zeros((N_sim, 1))


results = Parallel(n_jobs=-1)(delayed(simulate_psiam)(V_A, theta_A, V_E, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor) for _ in range(N_sim))

choices, RTs, is_act_resp = zip(*results)
choices = np.array(choices).reshape(-1, 1)
RTs = np.array(RTs).reshape(-1, 1)
is_act_resp = np.array(is_act_resp).reshape(-1, 1)

print(f'Num of act resp = {is_act_resp.sum()}/{N_sim}')
print(f'Number of aborts = {(RTs < t_stim).sum()}')

psiam_data = {'choices': choices, 'RTs': RTs, 'is_act_resp': is_act_resp, 'V_A': V_A, 'theta_A': theta_A, 
              'V_E': V_E, 'theta_E': theta_E, 't_stim': t_stim, 'Z_E': Z_E,
              't_A_aff': t_A_aff, 't_E_aff': t_E_aff, 't_motor': t_motor
              }


with open('psiam_data_delay_paper_007.pkl', 'wb') as f: # _1 for t_stim = 0, _2 for non t_stim = 0.3
    pickle.dump(psiam_data, f)




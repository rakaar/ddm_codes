from typing import Dict, List, Union

import numpy as np
import sdeint
import torch

from ddm_models.base_simulator import Simulator

import matplotlib.pyplot as plt


class Master_DDM(Simulator):
    """Build the model used for simulating data.

    Attributes
    ----------
    inputs: List[str]
        Specifies inputs passed to the simulator functions.
        Corresponds to parameters and input values specified in the config simulation file.
    simulator_results: List[str]
        Specifies the structure of the simulator output.
        Corresponds to the observations specified in the config simulation file.

    Methods
    -------
    __init__(self, inputs: List[str], simulator_results: List[str], **kwargs) -> None
        Use the config file to initialize attributes.
    __call__(self, input_data: torch.Tensor) -> torch.Tensor
        Use the given batch of input_data to simulate data.
    generate_data(self, input_dict: Dict(torch.Tensor)) -> Union[torch.Tensor, np.ndarray]
        Use the given input_dict sample to simulate one simulation result.
    """

    def __init__(
        self
    ):
        pass

    def __call__(
        self, theta
    ):
        """
        Perform the evidence accumulation process and generate responses based on the given parameters.

        Args:
            theta (torch.Tensor): A tensor containing the model parameters for each trial.

        Returns:
            torch.Tensor: A tensor containing the response time, choice, and response type for each trial.
        """

        # EVIDENCE ACCUMULATION TRAJECTORIES #
        def f(x, t):
            return v-lr*x
        def f2(x, t):
            return v_go
        def G(x, t):
            return 1

        num_trials, _ = theta.size()

        # time at which stimulus is switched on
        t_0 = 0

        # sound (hardware) delay - time it takes for sound to arrive at ears
        t_hardware_delay = 0.010

        # time at which stimulus arrives at ears
        t_stim_onset = t_0 + t_hardware_delay

        # time resolution
        dt = 0.001

        # maximum time at which a trial is still valid
        t_close = 2.0

        #generator = np.random.default_rng(seed=None)
        try:
            theta = theta.numpy()
        except:
            pass

        output = np.empty(shape=(num_trials, 3))


        for trial in range(num_trials):

            theta_i = theta[trial,:]

            # unpack parameters
            Nr0, lam, t_s, t_m, b, v_go, b_go, abl, ild, tfix = theta_i

            # commitment sigmoid slope
            s = 1
            # bias
            w = 0.5
            # leak rate
            lr = 0
            # boundary decay rate
            d = 0

            # time at which subject enters port
            t_enter = -np.round(tfix,3) # round to level of precision we have
            #print("tenter", t_enter)

            # time at which evidence starts to be integrated
            t_integration = t_stim_onset + t_s

            # # pre-stimulus time points
            # tspan_pre_stim = np.arange(t_enter, t_stim_onset, dt)

            # # post-stimulus time points
            # tspan_post_stim = np.arange(t_stim_onset, t_close+dt, dt)

            # # all time points
            # tspan = np.concatenate((tspan_pre_stim, tspan_post_stim), axis=0)

            # all time points
            tspan = np.arange(t_enter,  t_close+dt, dt)

            # number of timesteps
            N = len(tspan)

            # time resolution
            h = (tspan[N-1] - tspan[0])/(N - 1)

            # auxiliary terms
            q_e = 1
            T0 = 1/Nr0
            ABL_fun = (2*q_e/T0)*10**(lam*abl/20)
            Chi = 40/np.log(10)

            # evidence accumulation drifts and sigmas
            drift_rate = ABL_fun*(1/Chi)*(lam*ild)
            sigma = np.sqrt(ABL_fun*q_e)

            # sigma scaled drifts and boundaries; boundary separations computation; relabelling rel. starting pts and ndts
            v = drift_rate/sigma
            bl = -b/sigma
            bu = b/sigma
            boundary_separation = np.abs(bu-bl)

            # COLLAPSING EVIDENCE BOUND #
            # exponential bound #
            thres = np.exp(-(tspan-t_integration)*d)
            lower_bound = thres*bl
            upper_bound = thres*bu

            # ACTION INITIATION BOUND #
            go_bound = b_go*np.ones(len(tspan))

            # allocate space for EA result, generate noise trajectory
            y_0 = (w-0.5) * boundary_separation # initial evidence (bias)
            y = np.zeros((N), dtype=y_0.dtype)
            y[np.where(tspan == t_integration)[0]] = y_0
            dW = np.random.normal(0.0, np.sqrt(h), N)

            # allocate space for AI result, generate noise trajectory
            y2_0 = 0  # initial "evidence" for action initiation
            y2 = np.zeros((N), dtype=y_0.dtype)
            y2[0] = y2_0
            dW2 = np.random.normal(0.0, np.sqrt(h), N)

            # initialise data
            response = np.nan
            rt = np.nan
            choice = np.nan

            # initalise necessary states
            act = False       # indicates whether the action initiation bound has been hit
            proceed_ea = True # indicates whether the evidence accumulation is still active
            proceed_ai = True # indicates whether the action initiation process is still active
            #pre_stim = True   # indicated whether the current time preceeds stimulus onset
            #action_clock = 0  # counts timesteps after action initiation bound has been hit

            # # initialise key time points
            # t_leave = np.nan
            # t_left = np.nan
            # t_decided = np.nan

            for n in range(0, N-1): # loop over time steps
                
                tn = tspan[n]

                # EA integration
                yn = y[n]
                y[n+1] = yn
                dWn = dW[n]
                if tn >= t_integration and proceed_ea:
                    #pre_stim = False
                    #post_stim = True
                    y[n+1] += f(yn, tn)*h + G(yn, tn)*dWn

                # AI integration
                y2n = y2[n]
                y2[n+1] = y2n
                dW2n = dW2[n]
                if tn >= t_enter and proceed_ai: # (tn >= t_enter always true because there is currently no t_s_ai)
                    y2[n+1] += f2(y2n, tn)*h + G(y2n, tn)*dW2n

                # Proactive responses
                if y2[n+1] >= go_bound[n] and proceed_ai:
                    act = True
                    t_act = tn
                    # print("act")
                    # print("tn, tm", tn, t_m)
                    if tn < -t_m+t_hardware_delay: # abort
                        rt, choice = tn+t_m, np.random.choice(a=[0, 1], p=[0.5, 0.5])
                        response = 0
                        break
                    else: # contaminated response
                        rt = tn+t_m
                        response = 1
                        proceed_ai = False

                # Contaminated (proactive) responses
                if act and tn-t_act >= t_s+t_m:
                    p = 1/(1 + np.exp(-s*yn))              ### SIGMOID FN SLOPE PARAMETER TO BE ADDED TO MODEL PARAMETERS ###
                    cs = [0, 1]
                    ps = [1-p, p]
                    choice = np.random.choice(a=cs, p=ps)
                    break

                # Reactive responses
                elif y[n+1] <= lower_bound[n] and proceed_ea:
                    proceed_ea = False
                    if not act:
                        rt, choice = tn+t_m, 0
                        response = 2
                        break
                elif y[n+1] >= upper_bound[n] and proceed_ea:
                    proceed_ea = False
                    if not act:
                        rt, choice = tn+t_m, 1
                        response = 2
                        break

            #print("tn, rt, c", tn, rt, choice)

            # No response
            if (rt > t_close) or (tn > t_close - 2*dt):
                rt, choice = np.random.gamma(1.0) + t_close, np.random.choice(a=[0, 1], p=[0.5, 0.5])
                response = 3

            # if rt == np.nan:
            #     print("nan rt, last time was ", tn)

                # fig, axs = plt.subplots(1,2)
                # axs[0].plot(tspan, y, c="tab:blue")
                # if response == 0:
                #     marker_num = "abort" #"3"
                # elif choice == 0:
                #     marker_num = "left"  #"1"
                # elif choice == 1:
                #     marker_num = "right"  #"2"
                # elif response == 3:
                #     marker_num = "no resp"  #"4"

                # if reject == 1:
                #     reject_status = "try again"
                # else: 
                #     reject_status = "next theta"
                # #axs[0].plot(tspan[::10], np.zeros(len(tspan[::10])), c="r", marker=marker_num, ms=5)
                # axs[0].annotate(marker_num + ", " + reject_status, xy=(-12, -12), xycoords='axes points',
                #     size=14, ha='right', va='top',
                #     bbox=dict(boxstyle='round', fc='w'))
                # axs[1].plot(tspan, y2, c="tab:orange")
                # axs[0].plot(tspan, lower_bound, c="k")
                # axs[0].plot(tspan, upper_bound, c="k")
                # axs[1].plot(tspan, go_bound, c="k")
                # plt.show()

                # print("y", y)
                # print("ub", upper_bound)
                # print("lb", lower_bound)
                # print("y2", y2)
                # print("go bound", go_bound)

            output[trial,0] = rt
            output[trial,1] = choice
            output[trial,2] = response
        
        #print(rejection_counter)

        return torch.from_numpy(output)
    

    def simulate_trajectories(
        self, theta
    ):
        """
        Simulates trajectories of evidence accumulation for a given set of parameters.

        Args:
            theta (Tensor): A tensor containing the model parameters for each trial. 
                            Shape: (num_trials, num_parameters)

        Returns:
            dict: A dictionary containing the simulation results.
                - data (Tensor): A tensor containing the response time, choice, and response type for each trial.
                                 Shape: (num_trials, 3)
                - parameters (dict): A dictionary containing the model parameters used for simulation.
                                     - lb (list): A list of lower bounds for each time step.
                                     - ub (list): A list of upper bounds for each time step.
                                     - bgo (list): A list of go bounds for each time step.
                                     - tfix (list): A list of fixation periods for each trial.
                                     - tspan (list): A list of time steps for each trial.
                                     - ndt (list): A list of non-decision times for each trial.
                - trajectories (dict): A dictionary containing the evidence accumulation trajectories for each trial.
                                       - ea (list): A list of trajectories for the evidence accumulation process.
                                       - ai (list): A list of trajectories for the action initiation process.
        """

        # EVIDENCE ACCUMULATION TRAJECTORIES #
        def f(x, t):
            return v-lr*x
        def f2(x, t):
            return v_go
        def G(x, t):
            return 1

        num_trials, _ = theta.size()

        # the reaction time starts at 0
        rt0 = 0
        # define time interval and number of intermediate steps
        #nunit = 100
        tmin = 0.01
        dt = 0.001
        tmax = 1.0
        tspan_ea = np.arange(rt0, tmax+dt, dt)#int(tmax*nunit)+1)

        #generator = np.random.default_rng(seed=None)
        theta = theta.numpy()

        output_data = np.empty(shape=(num_trials, 3))

        trajectories = {}
        trajectories["ai"] = []
        trajectories["ea"] = []

        output_parameters = {}
        output_parameters["lb"] = []
        output_parameters["ub"] = []
        output_parameters["bgo"] = []
        output_parameters["tfix"] = []
        output_parameters["tspan"] = []
        #output_parameters["ndt"] = []
        output_parameters["t_s"] = []
        output_parameters["t_m"] = []

        rejection_counter = 0

        for trial in range(num_trials):

            # initialise rejection (abort) counter
            reject = 1

            theta_i = theta[trial,:]

            #print(theta_i)

            # unpack parameters
            N_neur, r0, lam, alpha, t_s_ea, t_m_ea, t_m_ai, w, lr, b, d, s, v_go, b_go, abl, ild, tfix = theta_i

            # fixation period
            tspan_fixation = np.arange(-tfix, rt0, dt)
            tspan = np.concatenate((tspan_fixation, tspan_ea), axis=0)
            N = len(tspan) # number of timesteps
            h = (tspan[N-1] - tspan[0])/(N - 1)
            # # Find the index where tspan >= t_s for the first time
            ea_idx = np.argmax(tspan >= t_s_ea)
            #print("tstart", tspan[ea_idx])

            # auxiliary terms
            q_e = 1
            #T0 = 1/g
            ABL_fun_power = r0*10**(lam*abl/20)
            # print("ABL_fun_power", ABL_fun_power)
            # print("lam", lam)
            # print("abl", abl)
            # print("r0", r0)
            # print("10**(lam*abl/20)", 10**(lam*abl/20))
            ABL_fun_factor = 2*N_neur*q_e
            ABL_fun_mu = ABL_fun_factor*ABL_fun_power
            ABL_fun_sigma = ABL_fun_factor*q_e*ABL_fun_power**alpha
            Chi = 40/np.log(10)

            # evidence accumulation drifts, sigmas and ndt
            drift_rate = ABL_fun_mu*(1/Chi)*(lam*ild)
            sigma = np.sqrt(ABL_fun_sigma)
            #ndt = t_s+t_m

            # sigma scaled drifts and boundaries; boundary separations computation; relabelling rel. starting pts and ndts
            v = drift_rate/sigma
            bl = -b/sigma
            bu = b/sigma
            boundary_separation = np.abs(bu-bl)

            # print("b", b)
            # print("bu", bu)
            # print("bl", bl)
            # print("boundary separation", boundary_separation)

            # COLLAPSING EVIDENCE BOUND #
            # exponential #
            thres = np.exp(-tspan*d)
            lower_bound = thres*bl
            upper_bound = thres*bu

            # ACTION INITIATION BOUND #
            go_bound = b_go*np.ones(len(tspan))

            #print("theta", theta_i)
            y0 = (w-0.5) * boundary_separation
            y2_0 = 0

            # allocate space for result
            y = np.zeros((N), dtype=y0.dtype)
            y[ea_idx] = y0 # initial evidence for EA
            y2 = np.zeros((N), dtype=y0.dtype)
            dW = np.random.normal(0.0, np.sqrt(h), N)
            dW2 = np.random.normal(0.0, np.sqrt(h), N)
            response = np.nan
            rt = np.nan
            choice = np.nan

            #y[0] = y0 # initial evidence for EA
            y2[0] = y2_0 # initial evidence for AI

            for n in range(0, N-1): # loop over time steps
                tn = tspan[n]

                # EA
                yn = y[n]
                dWn = dW[n]
                if tn >= t_s_ea:
                    # print("ping")
                    # print("ping yn", yn)
                    y[n+1] = yn + f(yn, tn)*h + G(yn, tn)*dWn
                    # print("ping yn+1", y[n+1])
                # else:
                #     y[n+1] = yn

                # AI
                y2n = y2[n]
                dW2n = dW2[n]
                y2[n+1] = y2n + f2(y2n, tn)*h + G(y2n, tn)*dW2n

                # Aborts and Proactive responses
                if y2[n+1] >= go_bound[n]:
                    if tn < -t_m_ai + tmin:  # abort
                        rt, choice = tn+t_m_ai, 0
                        response = 0
                        #print("response 0")
                        reject = 1
                        break
                    else:  # contaminated response             ### PROACTIVE RESP rt AND choice TO BE REVIEWED
                        rt = tn+t_m_ai
                        p = 1/(1 + np.exp(-s*yn))              ### SIGMOID FN SLOPE PARAMETER TO BE ADDED TO MODEL PARAMETERS ### 
                        # print("p",p)
                        # print("s",s)
                        # print("yn",yn)
                        
                        cs = [0, 1, 2, 3]
                        ps = [0, 1-p, p, 0]
                        choice = np.random.choice(a=cs, p=ps)
                        #print(choice)
                        response = 1
                        #print("response 1")
                        reject = 0
                        break
                # Reactive responses
                elif y[n+1] <= lower_bound[n]:
                    rt, choice = tn+t_m_ea, 1
                    # if rt < 0:
                    #     print("rt < 0")
                    #     print("tn", tn)
                    #     print("t_m", t_m)
                    #     print("t_s", t_s)
                    #     print("rt", rt)
                    #     print("choice", choice)
                    #     print("response", response)
                    #     print("lower_boundn", lower_bound[n])
                    #     print("yn", y[n])
                    #     print("yn+1", y[n+1])
                    #     print("y", y)
                    #     print("lower_bound", lower_bound)
                    #     print("theta", theta_i)
                    #     print("sigma", sigma)
                    response = 2
                    #print("response 2left")
                    reject = 0
                    break
                elif y[n+1] >= upper_bound[n]:
                    rt, choice = tn+t_m_ea, 2
                    response = 2
                    #print("response 2right")
                    reject = 0
                    break

                # No response
                elif tn + t_m_ea > tmax:
                    rt, choice = np.random.gamma(1.0) + tmax, 3
                    response = 3
                    reject = 1
                    break

            # # No response
            # if tn == tspan[N-2]:
            #     rt, choice = t_m + np.random.gamma(1.0) + tmax, 3
            #     response = 3
            #     reject = 1

            # if rt == np.nan:
            #     print("nan rt, last time was ", tn)

                # fig, axs = plt.subplots(1,2)
                # axs[0].plot(tspan, y, c="tab:blue")
                # if response == 0:
                #     marker_num = "abort" #"3"
                # elif choice == 0:
                #     marker_num = "left"  #"1"
                # elif choice == 1:
                #     marker_num = "right"  #"2"
                # elif response == 3:
                #     marker_num = "no resp"  #"4"

                # if reject == 1:
                #     reject_status = "try again"
                # else: 
                #     reject_status = "next theta"
                # #axs[0].plot(tspan[::10], np.zeros(len(tspan[::10])), c="r", marker=marker_num, ms=5)
                # axs[0].annotate(marker_num + ", " + reject_status, xy=(-12, -12), xycoords='axes points',
                #     size=14, ha='right', va='top',
                #     bbox=dict(boxstyle='round', fc='w'))
                # axs[1].plot(tspan, y2, c="tab:orange")
                # axs[0].plot(tspan, lower_bound, c="k")
                # axs[0].plot(tspan, upper_bound, c="k")
                # axs[1].plot(tspan, go_bound, c="k")
                # plt.show()

                # print("y", y)
                # print("ub", upper_bound)
                # print("lb", lower_bound)
                # print("y2", y2)
                # print("go bound", go_bound)

            output_data[trial,0] = rt
            output_data[trial,1] = choice
            output_data[trial,2] = response

            output_parameters["lb"].append(lower_bound)
            output_parameters["ub"].append(upper_bound)
            output_parameters["tfix"].append(tfix)
            output_parameters["tspan"].append(tspan)
            output_parameters["bgo"].append(go_bound)
            #output_parameters["ndt"].append(ndt)
            output_parameters["t_s"].append(t_s)
            output_parameters["t_m"].append(t_m)

            trajectories["ea"].append(y)
            trajectories["ai"].append(y2)

        output = {}
        output["data"] = torch.from_numpy(output_data)
        output["parameters"] = output_parameters
        output["trajectories"] = trajectories
        
        #print(rejection_counter)

        return output
    

    def simulate_trajectories_debugging(
        self, theta
    ):
        """
        Simulates trajectories for debugging purposes.

        Parameters:
        - theta (Tensor): A tensor containing the model parameters for each trial.

        Returns:
        - output (dict): A dictionary containing the simulation results.
            - data (Tensor): A tensor containing the response time, choice, and response type for each trial.
            - parameters (dict): A dictionary containing the lower bound, upper bound, fixation time, time span, go bound, and non-decision time for each trial.
            - trajectories (dict): A dictionary containing the evidence accumulation trajectories for EA and AI for each trial.
        """

        # EVIDENCE ACCUMULATION TRAJECTORIES #
        def f(x, t):
            # print("scaled drift, v ", v)
            # print("leak rate, lr ", lr)
            # print("lr*x ", lr*x)
            return v-lr*x
        def f2(x, t):
            return v_go
        def G(x, t):
            return 1

        num_trials, _ = theta.size()

        # the reaction time starts at 0
        rt0 = 0
        # define time interval and number of intermediate steps
        #nunit = 100
        tmin = 0.01
        dt = 0.001
        tmax = 1.0
        tspan_ea = np.arange(rt0, tmax+dt, dt)#int(tmax*nunit)+1)

        #generator = np.random.default_rng(seed=None)
        theta = theta.numpy()

        output_data = np.empty(shape=(num_trials, 3))

        trajectories = {}
        trajectories["ai"] = []
        trajectories["ea"] = []

        output_parameters = {}
        output_parameters["lb"] = []
        output_parameters["ub"] = []
        output_parameters["bgo"] = []
        output_parameters["tfix"] = []
        output_parameters["tspan"] = []
        output_parameters["ndt"] = []

        rejection_counter = 0

        for trial in range(num_trials):

            # initialise rejection (abort) counter
            reject = 1

            theta_i = theta[trial,:]

            # unpack parameters
            g, e, b, w, d, ndt, lr, v_go, b_go, abl, ild, tfix = theta_i

            # fixation period
            tspan_fixation = np.arange(-tfix, rt0, dt)
            tspan = np.concatenate((tspan_fixation, tspan_ea), axis=0)
            N = len(tspan) # number of timesteps
            h = (tspan[N-1] - tspan[0])/(N - 1)

            # auxiliary terms
            q_e = 1
            T0 = 1/g
            ABL_fun = (2*q_e/T0)*10**(e*abl/20)
            Chi = 40/np.log(10)

            # evidence accumulation drifts and sigmas
            drift_rate = ABL_fun*(1/Chi)*(e*ild)
            sigma = np.sqrt(ABL_fun*q_e)

            # print("drift rate ", drift_rate)
            # print("sigma ", sigma)

            # sigma scaled drifts and boundaries; boundary separations computation; relabelling rel. starting pts and ndts
            v = drift_rate/sigma
            bl = -b/sigma
            bu = b/sigma
            boundary_separation = np.abs(bu-bl)

            # print("b ", b)
            # print("bu ", bu)

            # COLLAPSING EVIDENCE BOUND #
            # exponential #
            thres = np.exp(-tspan*d)
            lower_bound = thres*bl
            upper_bound = thres*bu

            # ACTION INITIATION BOUND #
            go_bound = b_go*np.ones(len(tspan))

            y0 = (w-0.5) * boundary_separation
            y0_2 = 0

            #print("theta", theta_i)

            # allocate space for result
            y = np.zeros((N), dtype=y0.dtype)
            y2 = np.zeros((N), dtype=y0.dtype)
            dW = np.random.normal(0.0, np.sqrt(h), N)
            dW2 = np.random.normal(0.0, np.sqrt(h), N)
            response = np.nan
            rt = np.nan
            choice = np.nan

            y[0] = y0 # initial evidence for EA
            y2[0] = y0_2 # initial evidence for AI
            for n in range(0, N-1): # loop over time steps
                tn = tspan[n]

                # EA
                yn = y[n]
                dWn = dW[n]
                if tn >= 0:
                    y[n+1] = yn + f(yn, tn)*h + G(yn, tn)*dWn
                else:
                    y[n+1] = yn

                # AI
                y2n = y2[n]
                dW2n = dW2[n]
                y2[n+1] = y2n + f2(y2n, tn)*h + G(y2n, tn)*dW2n

                # Aborts and Proactive responses
                if y2[n+1] >= go_bound[n]:
                    if tn < -ndt + tmin:  # abort
                        rt, choice = tn+ndt, 0
                        response = 0
                        reject = 1
                        break
                    else:  # contaminated response             ### PROACTIVE RESP rt AND choice TO BE REVIEWED
                        rt = tn+ndt
                        p = 1/(1 + np.exp(-yn))              ### SIGMOID FN SLOPE PARAMETER TO BE ADDED TO MODEL PARAMETERS ### 
                        cs = [0, 1, 2, 3]
                        ps = [0, 1-p, p, 0]
                        choice = np.random.choice(a=cs, p=ps)
                        print("p",p)
                        print("choice",choice)
                        response = 1
                        reject = 0
                        break
                # Reactive responses
                elif y[n+1] <= lower_bound[n]:
                    rt, choice = tn+ndt, 1
                    response = 2
                    reject = 0
                    break
                elif y[n+1] >= upper_bound[n]:
                    rt, choice = tn+ndt, 2
                    response = 2
                    reject = 0
                    break

            # No response
            if tn == tspan[N-2]:
                rt, choice = np.random.gamma(1.0) + tmax, 3
                response = 3
                reject = 1

            # if rt == np.nan:
            #     print("nan rt, last time was ", tn)

                # fig, axs = plt.subplots(1,2)
                # axs[0].plot(tspan, y, c="tab:blue")
                # if response == 0:
                #     marker_num = "abort" #"3"
                # elif choice == 0:
                #     marker_num = "left"  #"1"
                # elif choice == 1:
                #     marker_num = "right"  #"2"
                # elif response == 3:
                #     marker_num = "no resp"  #"4"

                # if reject == 1:
                #     reject_status = "try again"
                # else: 
                #     reject_status = "next theta"
                # #axs[0].plot(tspan[::10], np.zeros(len(tspan[::10])), c="r", marker=marker_num, ms=5)
                # axs[0].annotate(marker_num + ", " + reject_status, xy=(-12, -12), xycoords='axes points',
                #     size=14, ha='right', va='top',
                #     bbox=dict(boxstyle='round', fc='w'))
                # axs[1].plot(tspan, y2, c="tab:orange")
                # axs[0].plot(tspan, lower_bound, c="k")
                # axs[0].plot(tspan, upper_bound, c="k")
                # axs[1].plot(tspan, go_bound, c="k")
                # plt.show()

                # print("y", y)
                # print("ub", upper_bound)
                # print("lb", lower_bound)
                # print("y2", y2)
                # print("go bound", go_bound)

            output_data[trial,0] = rt
            output_data[trial,1] = choice
            output_data[trial,2] = response

            output_parameters["lb"].append(lower_bound)
            output_parameters["ub"].append(upper_bound)
            output_parameters["tfix"].append(tfix)
            output_parameters["tspan"].append(tspan)
            output_parameters["bgo"].append(go_bound)
            output_parameters["ndt"].append(ndt)

            trajectories["ea"].append(y)
            trajectories["ai"].append(y2)

        output = {}
        output["data"] = torch.from_numpy(output_data)
        output["parameters"] = output_parameters
        output["trajectories"] = trajectories
        
        #print(rejection_counter)

        return output



    def generate_iid_data(
        self, input_dict: Dict
    ) -> Union[torch.Tensor, np.ndarray]:
        """Use the given input_dict sample to simulate iid simulation results.

        Parameters
        ----------
        input_dict
            The current sample of parameters and experimental conditions to be used for simulating data.
            The input names correspond to self.inputs. Access the inputs e.g. via `input_dict['drift]`.

        Returns
        -------
        sample
            Contains one simulation result in the order specified by self.simulator_results.
            If no valid results has been computed, return a tensor or array containing `NaN` or `±∞`.
        """
        #####

        batch_size = input_dict["compressive_power"].size()[0]

        x = torch.empty(size=(batch_size,2))

        for nth_obs in range(batch_size):

            compressive_power = input_dict["compressive_power"][nth_obs]
            firing_rate_coefficient = input_dict["firing_rate_coefficient"][nth_obs]
            T0 = 1/firing_rate_coefficient

            boundary_separation = input_dict["boundary_separation"][nth_obs]
            quantum_evidence = 1
            #threshold = boundary_separation/quantum_evidence

            ABL = input_dict["ABL"][nth_obs]
            ILD = input_dict["ILD"][nth_obs]

            ABL_fun = (2*quantum_evidence/T0)*10**(compressive_power*ABL/20)
            Chi = 40/np.log(10)
            #Gamma = compressive_power*threshold/Chi

            drift_rate = ABL_fun*(1/Chi)*(compressive_power*ILD)
            sigma = np.sqrt(ABL_fun*quantum_evidence)

            relative_starting_point = 0.5  # we assume no bias
            non_decision_time = input_dict["non_decision_time"][nth_obs]

            # the reaction time starts at 0
            rt = 0
            # define time interval and number of intermediate steps
            tmax = 7.0
            tspan = np.linspace(0.0, tmax, 1001)

            def f(x, t):
                return float(drift_rate)

            def g(x, t):
                return float(sigma)

            starting_point = float(relative_starting_point * boundary_separation)

            traj = sdeint.itoSRI2(f, g, starting_point, tspan).flatten()

            # check which boundary has been crossed (first)
            lower_bound = 0
            upper_bound = boundary_separation.item()
            pass_lower_bound = np.where(traj < lower_bound)[0]
            pass_upper_bound = np.where(traj > upper_bound)[0]

            if pass_lower_bound.size > 0 and pass_upper_bound.size > 0:
                if pass_lower_bound[0] < pass_upper_bound[0]:
                    rt, choice = tspan[pass_lower_bound[0]] + non_decision_time, 0
                else:
                    rt, choice = tspan[pass_upper_bound[0]] + non_decision_time, 1
            elif pass_lower_bound.size > 0:
                rt, choice = tspan[pass_lower_bound[0]] + non_decision_time, 0
            elif pass_upper_bound.size > 0:
                rt, choice = tspan[pass_upper_bound[0]] + non_decision_time, 1
            # if no boundary has been crossed, return nan
            else:
                rt, choice = torch.nan, torch.nan

            x[nth_obs,0] = rt
            x[nth_obs,1] = choice

        return x


### IMPORTS ###
import torch
from utils.transform_distributions import _define_affine_transdist, _define_loguniform_transdist
from torch.distributions import Uniform, Categorical, Exponential
from sbi.utils import MultipleIndependent



### DISTRIBUTION PARAMETERS ###
N_ILD_conds = 2*16+1  # total number of ILD conditions (16 negative, 16 positive and one equal to 0)
p_ILD_cond = float(1/(N_ILD_conds))
p_ILD_cond_vector = torch.tensor([p_ILD_cond]*N_ILD_conds).reshape(1,-1)

N_ABL_conds = 3  # total numner of ABL conditions
p_ABL_cond = float(1/N_ABL_conds)
p_ABL_cond_vector = torch.tensor([p_ABL_cond]*N_ABL_conds).reshape(1,-1)

m_TFX = 0.500
r_TFX = 1/m_TFX
t_TFX = 0.100



### PRIOR SUPPORT ###
# This HAS TO agree with the lb and ub of the uniform distributions below
prior_support = [(100.0, 5000.0),
                 (0.05, 0.2),
                 (0.005, 0.150),
                 (0.005, 0.100),
                 (30.0, 100.0),
                 (0.0, 5.0),
                 (1.0, 5.0)]

# Define labels for each parameter
parameter_labels = [r"$Nr_0$",
                    r"$\lambda$",
                    r"$t_{s}^{ea}$",
                    r"$t_{m}^{ea}$",
                    r"$\theta_{ea}$",
                    r"$v_{ai}$",
                    r"$\theta_{ai}$"]


### MODEL PARAMETERS ###

# Channel parameters
#NNC_Distribution = _define_loguniform_transdist(lb=torch.tensor([100]), ub=torch.tensor([5000]))              # number of neurons in each channel
NNC_Distribution = Uniform(torch.tensor([100.0]), torch.tensor([5000.0]))                                      # number of neurons in each channel
#CPO_Distribution = _define_loguniform_transdist(lb=torch.tensor([0.05]), ub=torch.tensor([0.2]))              # compression power
CPO_Distribution = Uniform(torch.tensor([0.05]), torch.tensor([0.2]))  # compression power

# Delay parameters
SED_Distribution = Uniform(torch.tensor([0.005]), torch.tensor([0.150]))            # sensory delay (EA)
MOD_Distribution = Uniform(torch.tensor([0.005]), torch.tensor([0.100]))            # motor delay

# Control parameters
#DBO_Distribution = _define_loguniform_transdist(lb=torch.tensor([30]), ub=torch.tensor([100]))    # decision boundary
DBO_Distribution = Uniform(torch.tensor([30.0]), torch.tensor([100.0]))    # decision boundary

# Action parameters
AID_Distribution = Uniform(torch.tensor([0.0]), torch.tensor([5.0]))                # action initiation drift rate
AIB_Distribution = Uniform(torch.tensor([1.0]), torch.tensor([5.0]))                # action initiation threshold

# Experimental parameters
ABL_Distribution = _define_affine_transdist(loc=20*torch.ones(1), scale=20*torch.ones(1), inc_dist=Categorical(probs=p_ABL_cond_vector), ndim=1)    # ABL (experimental condition)
ILD_Distribution = _define_affine_transdist(loc=-16*torch.ones(1), scale=1*torch.ones(1), inc_dist=Categorical(probs=p_ILD_cond_vector), ndim=1)    # ILD (experimental condition)
TFX_Distribution = _define_affine_transdist(loc=t_TFX*torch.ones(1), scale=1*torch.ones(1), inc_dist=Exponential(torch.tensor([r_TFX])), ndim=1)    # fixation time (experimental condition)



### PROPOSAL ###
proposal = MultipleIndependent(
    [
        ### Channel parameters
        NNC_Distribution,     # number of neurons in each channel
        CPO_Distribution,     # compression power

        ### Delay parameters
        SED_Distribution,     # EA sensory delay
        MOD_Distribution,     # motor delay

        ### Control parameters
        DBO_Distribution,     # decision boundary

        ### Action parameters
        AID_Distribution,     # action initiation drift rate
        AIB_Distribution,     # action initiation threshold

        ### Experimental parameters
        ABL_Distribution,     # ABL (experimental condition)
        ILD_Distribution,     # ILD (experimental condition)
        TFX_Distribution,     # fixation time (experimental condition)
    ],
    validate_args=False,
)


### PRIOR ###
prior = MultipleIndependent(
    [
        ### Channel parameters
        NNC_Distribution,     # number of neurons in each channel
        CPO_Distribution,     # compression power

        ### Delay parameters
        SED_Distribution,     # EA sensory delay
        MOD_Distribution,     # motor delay

        ### Control parameters
        DBO_Distribution,     # decision boundary

        ### Action parameters
        AID_Distribution,     # action initiation drift rate
        AIB_Distribution,     # action initiation threshold
    ],
    validate_args=False,
)
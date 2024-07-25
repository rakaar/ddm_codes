import numpy as np
from numba import jit
import pickle
from joblib import Parallel, delayed

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

def loglike_fn(params):
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

    loglike = np.sum(log_pos) + np.sum(log_neg)
    
    return loglike

def transform_random_number(u, a, b):
    return (b-a)*u + a 


def prior_transform(u):
    # u ~ uniform [0,1]
    priors = np.zeros_like(u)
    with open('dynesty_priors.pkl', 'rb') as f:
        priors_bounds = pickle.load(f)
    
    priors[0] =  transform_random_number(u[0], priors_bounds['v'][0], priors_bounds['v'][1])
    priors[1] = transform_random_number(u[1], priors_bounds['a'][0], priors_bounds['a'][1])
    priors[2] = transform_random_number(u[2], priors_bounds['w'][0], priors_bounds['w'][1])

    return priors



def _quantile(x, q=[0.5 - 0.5 * 0.999999426697, 0.5 + 0.5 * 0.999999426697], weights=None):
    """
    Compute (weighted) quantiles from an input set of samples.

    Parameters
    ----------
    x : `~numpy.ndarray` with shape (nsamps,)
        Input samples.

    q : `~numpy.ndarray` with shape (nquantiles,)
       The list of quantiles to compute from `[0., 1.]`.

    weights : `~numpy.ndarray` with shape (nsamps,), optional
        The associated weight from each sample.

    Returns
    -------
    quantiles : `~numpy.ndarray` with shape (nquantiles,)
        The weighted sample quantiles computed at `q`.

    """

    # Initial check.
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    # Quantile check.
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0. and 1.")

    if weights is None:
        # If no weights provided, this simply calls `np.percentile`.
        return np.percentile(x, list(100.0 * q))
    else:
        # If weights are provided, compute the weighted quantiles.
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x).")
        idx = np.argsort(x)  # sort samples
        sw = weights[idx]  # sort weights
        cdf = np.cumsum(sw)[:-1]  # compute CDF
        cdf /= cdf[-1]  # normalize CDF
        cdf = np.append(0, cdf)  # ensure proper span
        quantiles = np.interp(q, cdf, x[idx]).tolist()
        return quantiles
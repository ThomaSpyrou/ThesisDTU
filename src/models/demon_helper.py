import numpy as np
from scipy.stats import t, norm
import sys
sys.path.append('../')
from utils.utilities import *


G = []   
mu_hat = 0.5    
batch_size = 10    
risk_tol = 0.1    
query_period = 0.5    
drift_bound = 0.1    
linear_prior = 0.5    
N = 0   
query = True    

def g(S, S_prime):
    # anomaly score
    return np.abs(np.mean(S_prime) - np.mean(S))


def compute_wt(wt_prev, Gt, delta_mu_t):
    return wt_prev + Gt * delta_mu_t

def std_err_forecast(S, wt, tau, batch_size):
    S = np.array(S)
    X = np.column_stack((np.ones(len(S)), np.arange(len(S))))
    y = S - wt[0] - wt[1] * np.arange(len(S))
    RSS = np.sum(y**2)
    sigma_hat = np.sqrt(RSS / (len(S) - 2))
    se = sigma_hat * np.sqrt(1 + (tau + (batch_size+1)/2) * np.sum(X**2))

    return se

def compute_plbl(S, wt, tau, batch_size, delta, rho, risk_tol):
    # probability of label given the detection threshold
    se_t = std_err_forecast(S, wt, tau, batch_size)
    nu_t = delta * (tau + (batch_size+1)/2)
    t_t = max(abs(mu_hat - rho) - risk_tol, 0) / delta
    z_t = (risk_tol + t_t - batch_size * delta) / se_t
    plbl = 1 - 2 * norm.cdf(z_t)

    return max(min(plbl, 1 - risk_tol), risk_tol)

def compute_pdet(N, S, wt, tau, batch_size, delta, rho, risk_tol):
    # probability of detection given the label and the drift bound
    se_t = std_err_forecast(S, wt, tau, batch_size)
    nu_t = delta * (tau + (batch_size+1)/2)
    t_t = max(abs(mu_hat - rho) - risk_tol, 0) / delta
    z_t = (risk_tol + t_t - batch_size * delta) / se_t
    p_t = 2 - 2 * t.cdf(z_t, N-2)
    pdet = 1 - t.cdf((nu_t - t_t + risk_tol) / se_t, N-2)

    return pdet

def compute_pt(S, wt, tau, batch_size, delta, rho, risk_tol, linear_prior):
    #  probability of querying given the label and the detection threshold
    plbl = compute_plbl(S, wt, tau, batch_size, delta, rho, risk_tol)
    pdet = compute_pdet(N, S, wt, tau, batch_size, delta)

    return plbl, pdet


if __name__ == '__main__':
    
    from scipy.stats import ks_2samp

    # Define the transforms to apply to the datders for the two datasets
    train_loader, test_loader = load_data() 

    # Get a batch of images from each dataset
    batch1, _ = next(iter(train_loader))
    batch2, _ = next(iter(test_loader))

    # Flatten the images to 1D arrays
    batch1 = batch1.view(batch1.shape[0], -1)
    batch2 = batch2.view(batch2.shape[0], -1)

    # Convert the torch.Tensor objects to lists
    batch1 = [item for sublist in batch1.tolist() for item in sublist]
    batch2 = [item for sublist in batch2.tolist() for item in sublist]

   
    # Calculate the KS statistic and p-value
    ks_statistic, p_value = ks_2samp(batch1, batch2)
    # Print the results
    print('KS statistic: {:.4f}'.format(ks_statistic))
    print('p-value: {:.4f}'.format(p_value))



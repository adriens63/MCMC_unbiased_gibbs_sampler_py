from os import stat
import numpy as np
from scipy import stats
from typing import Array
from math import gamma

from constants import *






# ********************* seed ******************
SEED = 101
np.random.seed(SEED)






# ********************* gibbs sampler *****************

def sum_over_theta(theta: Array[np.float32]):
    K = theta.shape[0]
    sum_of_squares = np.sum(theta ** 2)
    sum_elements = np.sum(theta)
    return  (sum_of_squares - sum_elements**2 / K) / 2. , sum_elements / K




def gibbs_sampler(theta: Array[np.float32]):
    K = theta.shape[0]
    sum_gamma, mean_mu = sum_over_theta(theta)

    # simulation of A
    A = stats.invgamma(a + (K - 1.) / 2., scale = b + sum_gamma, size = 1)

    # simulation of mu
    mu = stats.norm.rvs(loc = mean_mu, scale = A / K, size = 1)

    # computation of the constants
    inv_v_plus_a = 1. / V + A

    # simulation of theta 
    for i in range(K):
        theta[i] = stats.norm.rvs(loc = inv_v_plus_a * (mu * V + z[i]), scale = inv_v_plus_a * A * V, size = 1)
    
    return A, mu, theta






# **************** coupling *************

def maximal_coupling(theta_x_t, theta_y_t_1, sampler_p, p, sampler_q, q):
    """
    sampler_p: function that sample from the law that has distribution p
    p: first distribution, val p : float -> float
    sampler_q: function that sample from the law that has distribution q
    q: second distribution, val p : float -> float

    in our case, sampler_p and sampler_q will be the gibbs_sampler, that yields the random vector (A, mu, theta_i)
    both p and q will be the distribution of the random vector (A, mu, theta_i) 
    """
    X = sampler_p(theta_x_t)
    W = np.random.uniform(low = 0., high = p(X))

    if W <= q(X):
        return X,X
    else:
        Y_star = sampler_q(theta_y_t_1)
        W_star = np.random.uniform(low = 0., high = q(Y_star))

        def aux_step_2(y_star, w_star):
            if w_star <= p(y_star):
                y_star = sampler_q(theta_y_t_1)
                w_star = np.random.uniform(low = 0., high = q(y_star))
                return aux_step_2(y_star, w_star)
            else:
                return X, y_star

        aux_step_2(y_star = Y_star, w_star = W_star)




def p(alpha: float, m:float, t: Array[float]):
    """
    distribution of the random vector (A, mu, theta_i) evaluated at (alpha, m, t_i), computed with the bayes theorem applied to 
    X = (A, mu), THETA = theta_i :
    """
    K = t.shape[0]
    return np.exp(- (b + np.sum((t - m) ** 2) / 2) / alpha) * (1. / (2. * np.pi * alpha) ) ** (K / 2.) * b ** a * alpha ** (- a - 1.) / gamma(a)
    



def coupled_gibbs_sampler(x_t: Array[np.float32], y_t_1: Array[np.float32]):
    """
    One step of the coupled gibbs sampler
    """
    return maximal_coupling(theta_x_t = x_t[2], theta_y_t_1 = y_t_1[2], sampler_p = gibbs_sampler, p = p, sampler_q = gibbs_sampler, q = p)




def unbiased_mcmc_coupled_gibbs_sampler(burnin: int, m: int, theta_0: Array[float], h):
    """
    burnin is 'k' in the paper
    this function returns the mean H_k:m (X,Y), where X is (A, mu, theta_i)
    """
    
    x_0 = y_0 = theta_0
    x_1 = gibbs_sampler(theta_0) # we do not need to pass x_O, theta_0 is sufficient in the case of this gibbs sampler

    x_y = [(x_1, y_0)]
    H = []
    tau = 1

    while tau < m or x_y[-1][0] != x_y[-1][1]:
        x_y.append(coupled_gibbs_sampler(*x_y[-1]))
        tau += 1

    for l in range(burnin, m+1):
        H.append(h(x_y[l][0]) + np.sum([h(x) - h(y) for x, y in x_y[l+1:tau]]))

    return np.mean(H)



    
    

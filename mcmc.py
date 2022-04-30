from os import stat
import numpy as np
from scipy import stats
from numpy.typing import NDArray
from math import gamma

from constants import *






# ********************* seed ******************
SEED = 101
np.random.seed(SEED)




# ******************** tools ****************
tuple_add = lambda a, b: tuple(i + j for i, j in zip(a, b))
tuple_diff = lambda a, b: tuple(i - j for i, j in zip(a, b))







# ********************* gibbs sampler *****************

def sum_over_theta(theta: NDArray[np.float32]):
    K = theta.shape[0]

    print('....Computing sums for GS')
    sum_of_squares = np.sum(theta ** 2)
    print('sum_of_squares: ', sum_of_squares)

    sum_elements = np.sum(theta)
    print('sum_element: ', sum_elements)
    print('done;')
    print()
    return  (sum_of_squares - sum_elements**2 / K) / 2. , sum_elements / K




def gibbs_sampler(theta: NDArray[np.float32]):
    K = theta.shape[0]
    sum_gamma, mean_mu = sum_over_theta(theta)
    print('sum_gamma, mean_mu: ', sum_gamma, mean_mu)
    print()

    # simulation of A
    print('....Simulating A')
    print('scale:', b + sum_gamma)
    A = stats.invgamma.rvs(a + (K - 1.) / 2., scale = b + sum_gamma, size = 1)
    print('done;')

    # simulation of mu
    print('....Simulating mu')
    print('mean_mu:', mean_mu)
    mu = stats.norm.rvs(loc = mean_mu, scale = A / K, size = 1)
    print('done;')

    # computation of the constants
    inv_v_plus_a = 1. / (V + A)

    # simulation of theta 
    new_theta = np.zeros(shape = [K])
    print('....Simulating theta')
    for i in range(K):
        mean = inv_v_plus_a * (mu * V + z[i] * A)
        std = inv_v_plus_a * A * V
        print(f'mean: {mean[0]}')
        print(f'std: {std[0]}')
        new_theta[i] = stats.norm.rvs(loc = inv_v_plus_a * (mu * V + z[i] * A), scale = inv_v_plus_a * A * V, size = 1)
    print('done;')
    print()
    return A, mu, new_theta






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
    print('X', X)
    print()

    print('p(*X)', p(*X))
    print()

    W = np.random.uniform(low = 0., high = p(*X))

    if W <= q(*X):
        return X,X
    else:
        Y_star = sampler_q(theta_y_t_1)
        W_star = np.random.uniform(low = 0., high = q(*Y_star))

        def aux_step_2(y_star, w_star):
            if w_star <= p(*y_star):
                y_star = sampler_q(theta_y_t_1)
                w_star = np.random.uniform(low = 0., high = q(*y_star))
                return aux_step_2(y_star, w_star)
            else:
                return X, y_star

        aux_step_2(y_star = Y_star, w_star = W_star)




def p(alpha: float, m:float, t: NDArray[np.float32]):
    """
    distribution of the random vector (A, mu, theta_i) evaluated at (alpha, m, t_i), computed with the bayes theorem applied to 
    X = (A, mu), THETA = theta_i :
    """
    K = t.shape[0]
    return np.exp(- (b + np.sum((t - m) ** 2) / 2) / alpha) * (1. / (2. * np.pi * alpha) ) ** (K / 2.) * b ** a * alpha ** (- a - 1.) / gamma(a)
    



def coupled_gibbs_sampler(x_t: NDArray[np.float32], y_t_1: NDArray[np.float32]):
    """
    One step of the coupled gibbs sampler
    """
    return maximal_coupling(theta_x_t = x_t[2], theta_y_t_1 = y_t_1[2], sampler_p = gibbs_sampler, p = p, sampler_q = gibbs_sampler, q = p)




def unbiased_mcmc_coupled_gibbs_sampler(burnin: int, m: int, theta_0: NDArray[np.float32], h):
    """
    burnin is 'k' in the paper
    this function returns the mean H_k:m (X,Y), where X is (A, mu, theta_i)
    """
    
    x_0 = theta_0
    y_0 = (1., 1., theta_0.copy()) # The 'ones' will not be usefull, they will not interfere since we only care about y_0[2] 
    x_1 = gibbs_sampler(theta_0) # we do not need to pass x_O, theta_0 is sufficient in the case of this gibbs sampler

    x_y = [(x_1, y_0)] 

    H = []
    tau = 1

    while tau < m or x_y[-1][0] != x_y[-1][1]:
        print('....Starting loop')
        print('tau :', tau)
        print()
        print()

        print('x_y: ', x_y)
        print()
        print()

        x_y.append(coupled_gibbs_sampler(*x_y[-1]))
        tau += 1
    print('done;')


    for l in range(burnin, m):
        print('1', h(x_y[l][0]))
        print('2', np.sum([h(x) - h(y) for x, y in x_y[l+1:tau]]))
        print('l', l)
        print('m', m)
        print()
        H.append(h(x_y[l][0]) + np.sum([h(x) - h(y) for x, y in x_y[l+1:tau]]))

    print('H:', H)
    print()

    return np.mean(H, axis = -1)



    
    

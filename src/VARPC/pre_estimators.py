from __future__ import division, print_function
import os
import sys
from contextlib import redirect_stdout
import pandas as pd
from tqdm import notebook
import numpy as np
from scipy import sparse as sp
from sklearn.metrics import mean_squared_error
from IPython.display import Math, display

def estimate_Ahat(X,
                  Y,
                  eta=10e-0,
                  max_iter=100000,
                  sample_size='large',
                  lam=1e-5,
                  convergence_tolerance=1e-5):
    """Estimates pre-estimator for VAR for weighted lasso.

    Args:
        X: past-time samples as features, a float numpy array of shape (T, N, p).
        Y: target responses, a float numpy array of shape (T, N).
        eta: learning rate, float scalar.
        max_iter: maximum number of iterations, int scalar.
        sample_size: number of firms in the financial networks dictates sample_size, str.
            - 'small', it may be reasonable to select small when num_time_samples < num_firms.
            - 'large'
        lam: penalty for A, float scalar.
        convergence_tolerance: stopping criteria, float scalar.

    Returns:
        A: pre-estimatore for A, a float numpy array of shape (N, N, p).
    """
    T, N, p = X.shape
    obj = np.inf
    A = np.zeros((N, N, p), dtype=float)
    f = np.zeros_like(Y)
    print('Starting optimization for pre-estimator for A...')
    print('{0:5s}   {1:9s}   {2:9s}'.format('Iter',  'objective',  'relative_objective'))
    for iteration in range(max_iter):

        obj_prev = obj

        # Find gradients of obj with respect to A
        res = Y - f
        grad_A = -(2 / (N * T)) * np.tensordot(res.T, X, axes=([1], [0]))
        if sample_size=='small':
            grad_A +=  2 * lam * A

        # Update A
        A -= eta * grad_A
        f = X.reshape(T, -1)@A.reshape(N, -1).T

        obj = mean_squared_error(Y, f)
        if sample_size == 'small':
            obj += lam * np.sum(np.square(A))

        obj_rel = np.absolute((obj - obj_prev)/obj)
        if iteration%100==0:
            print('{0:5d}   {1: 3.6f}   {2: 3.6f}'.format(iteration, obj, obj_rel))

        if obj_rel <= convergence_tolerance:
            print('Solution converged in {} iterations with final objective={:.5f} and relative objective={:.8f}<={}'.format(iteration, obj, obj_rel, convergence_tolerance))
            break

    return A

def estimate_what(X,
                  Y,
                  G,
                  eta=1e-0,
                  max_iter=100000,
                  convergence_tolerance=1e-5):
    """Estimates weights for single-layered linear Graph Convolution adapted for time-series forecasting:
    with shapes:
      G: (num nodes, num nodes)
      X: (num nodes, num samples, num past-time samples)
      w: (num past-time samples, )

    Args:
      X: input, float numpy array of shape (num samples, num nodes, num past-time samples).
      Y: output, float numpy array of shape (num samples, num nodes)
      graph: adjacency matric, float numpy array of shape (num nodes, num nodes).
      eta: learning rate, float scalar.
      max_iter: maximum number of iterations, int scalar.
      convergence_tolerance: stopping criteria, float scalar.

    Returns:
      w: estimated parameters, float numpy array of shape (num past-time samples, ).
    """
    T, N, p = X.shape
    obj = np.inf
    w = np.zeros((p, ), dtype=float)
    f = np.zeros(Y.shape)
    X = np.transpose(X, axes=[1, 0, 2]) # N, T, p
    GX = np.tensordot(G, X, axes=([1], [0])) # gives N, T, p

    for iteration in range(max_iter):

        obj_prev = obj

        # Find gradients of obj with respect to w
        grad_w = np.zeros_like(w)
        res = Y - f
        grad_w = -(2 / (N * T)) * np.tensordot(
            res.transpose(),
            GX,
            axes=([0,1], [0,1])
        )

        # Update w
        w -= eta * grad_w
        f = np.tensordot(GX, w, axes=([2], [0])).transpose()

        obj = mean_squared_error(Y, f)
        obj_rel = np.absolute((obj - obj_prev)/obj)
        if iteration%100==0:
            print('{0:5d}   {1: 3.6f}   {2: 3.6f}'.format(iteration, obj, obj_rel))

        if obj_rel <= convergence_tolerance:
            print('Solution converged in {} iterations with final objective={:.5f} and relative objective={:.8f}<={}'.format(iteration, obj, obj_rel, convergence_tolerance))
            break

    return w


def estimate_rhohat(Y,
                    f,
                    sample_size='large'):
    """Estimates pre-estimator for partial correlation.

    Args:
        Y: target responses, a float numpy array of shape (T, N).
        f, forecast responses,  a float numpy array of shape (T, N).
        sample_size: number of firms in the financial networks dictates sample_size, str.
            - 'small', it may be reasonable to select small when num_time_samples < num_firms.
            - 'large'
    Returns:
        rho: pre-estimatore for rho, a float numpy array of shape (N, N).
    """
    eps = Y - f
    if sample_size == 'small':
        covariance = shrinkage_estimator(Y=eps)
    else:
        covariance = np.cov(eps.transpose())
    precision = np.linalg.inv(covariance)
    rho = np.zeros_like(precision)
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            rho[i,j] = -precision[i, j] / ((precision[i, i] * precision[j, j])**0.5)

    return rho

def estimate_c(Y):
    """Estimates pre-estimator for c with zero prediction.

    Args:
        Y: target responses, a float numpy array of shape (T, N).

    Returns:
        c: inverse of conditional variances, a float numpy array of shape (N, ).
    """
    c = 1 / np.var(Y, axis=0)
    return c

def generate_pre_estimators(X, Y, regularizer_rho=None, f_eta=5, f_lam=None, sample_size='large', model='varpc', G=None):
    """Generates pre-estimators for VAR (A), partial correlation (rho), and inverse of conditional variance (c).

    Args:
        X: past-time samples as features, a float numpy array of shape (T, N, p).
        Y: target responses, a float numpy array of shape (T, N).
        regularizer_rho:
        sample_size: number of firms in the financial networks dictates sample_size, str.
            - 'small', it may be reasonable to select small when num_time_samples < num_firms.
            - 'large'
        model: which model to use to find pre-estimators, str.
            - 'varpc'
            - 'gcnpc'
        G: weighted graph matrix, a float numpy array of shape (N, N).
          ignored when model='varpc'.
    """
    T, N, p = X.shape
    c = estimate_c(Y)
    preestimators = [c]
    if model=='varpc':
        A_hat = estimate_Ahat(X=X, Y=Y, eta=f_eta, lam=f_lam, sample_size=sample_size)
        preestimators.append(A_hat)
    elif model=='gcnpc':
        w_hat = estimate_what(X=X, Y=Y, G=G, eta=f_eta)
        preestimators.append(w_hat)
    else:
        raise ValueError('model: {} is not supported for pre-estimators.'.format(model_type))

    if regularizer_rho == 'Lasso' or regularizer_rho == 'L0L2':
        rho_hat = np.ones((N, N),dtype=float)
    elif regularizer_rho == 'AdaptiveLasso':
        if model=='varpc':
            f = X.reshape(T, -1)@A_hat.reshape(N, -1).T
        elif model=='gcnpc':
            X = np.transpose(X, axes=[1, 0, 2]) # N, T, p
            GX = np.tensordot(G, X, axes=([1], [0])) # gives N, T, p
            f = np.tensordot(GX, w_hat, axes=([2], [0])).transpose()
        else:
            raise ValueError('model_type: {} is not supported'.format(model_type))
        rho_hat = estimate_rhohat(Y, f, sample_size=sample_size)
    else:
        rho_hat = None

    preestimators.append(rho_hat)

    return tuple(preestimators)

def get_mask_partialcorrelation(G=None, masking=None, masking_level=None, **kwargs):
    """Generates masking for partial correlation parameters.

    Args:
        G: weighted graph matrix, a float numpy array of shape (N, N).
        masking: type of knowledge graph masking for regularization, str or None.
          - None, No knowledge graph used
          - 'hard', binary mask.
          - 'soft', weighted mask.
        masking_level: number of peers to retain in masking for each firm, scalar int.

    Returns:
        mask_rho: mask for partial correlation, a float numpy array of shape (N, N).
    """
    if masking is None:
        mask_rho = np.ones((G.shape[1], G.shape[1]), dtype=float)
    elif masking == 'hard':
        mask_rho = generate_penalty_coef(G, method="hard", sparse=masking_level)
    elif masking == 'soft':
        mask_rho = generate_penalty_coef(G, method="soft", sparse=masking_level, alpha=kwargs['alpha'])

    return mask_rho

def generate_penalty_coef(co, method="soft", **kwargs):
    """Generates hard and soft masking matrices based on knowledge graph for partial correlation regularization.

    Args:
        co: weighted graph matrix, a float numpy array of shape (N, N).
        method: masking strategy, str.
          - 'hard', a binary matrix of top K peers per firm.
          - 'soft', a weighted (g_max/g_ij) matrix of top K peers per firm.

    Returns:
        M: Masking matrix, a float numpy array of shape (N, N).
    """
    p = co.shape[1]
    iu1 = np.triu_indices(p,1)
    co1 = co[iu1]
    if method == "exp":
        alpha = kwargs['alpha']
        a = np.log(alpha)/(co1.max())
        coef = 1/alpha*np.exp(a*co)
        np.fill_diagonal(coef,0)
        return coef
    elif method == "hard":
        sparse = min(kwargs['sparse'],p)
        M1 = np.argsort(co,axis=1)[:,-sparse:]
        M = np.zeros((p,p))
        for i in range(p):
            tmp = set(M1[i,:])
            for j in range(p):
                if j == i:
                    continue
                elif j in tmp and co[i,j] != 0:
                    M[i,j] = 1
                    M[j,i] = 1
        M[M==0] = np.inf
        np.fill_diagonal(M, np.inf)
        return M
    elif method == "soft":
        cmax = co1.max()
        sparse = min(kwargs['sparse'],p)
        alpha = kwargs['alpha']
        M1 = np.argsort(co,axis=1)[:,-sparse:]
        M = np.zeros((p,p))
        for i in range(p):
            tmp = set(M1[i,:])
            for j in range(p):
                if j == i:
                    continue
                elif j in tmp:
                    M[i,j] = cmax/co[i,j]
                    M[j,i] = cmax/co[i,j]
        M[M==0] = M[M!=0].max()/alpha
        np.fill_diagonal(M, np.inf)
        return M

def shrinkage_estimator(Y, to='identity'):
    """Covariance Shrinkage estimator.

    Args:
        Y: multivariate time-series, a float numpy array of shape (T, N).
        to: type of shrinkage, str.
           - 'identity', shrink to identity.
    """
    n,p = Y.shape
    Y -=np.mean(Y,axis=0)
    S = Y.T@Y/n
    if to =="identity":
        m = np.mean(np.diag(S))
        d2 = np.linalg.norm(S - m *np.eye(p))**2/n
        b2 = np.sum([np.linalg.norm(Y[[i],:].T@Y[[i],:] - S)**2 for i in range(n)])/n**3
        b2 = np.minimum(b2,d2)
        return b2/d2*m*np.eye(p)+(d2-b2)/d2*S

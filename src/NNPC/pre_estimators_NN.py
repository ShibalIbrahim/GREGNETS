from __future__ import division, print_function
import os
import sys
from contextlib import redirect_stdout
import pandas as pd
from tqdm import tnrange, tqdm_notebook
import numpy as np
from scipy import sparse as sp
from sklearn.metrics import mean_squared_error
from IPython.display import Math


def Estimate_rhohat(Y = None, 
                    Ypred = None, 
                    n = 'large'):
    res = Y - Ypred
    if n == 'small':
        covariance = shrinkage_estimator(Y = res)
    else:
        covariance = np.cov(res.transpose())
    precision = np.linalg.inv(covariance)
    rho= np.zeros(precision.shape,dtype=float)
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            rho[i,j] = -precision[i,j]/((precision[i,i]*precision[j,j])**0.5) 
            
    return rho

def Estimate_c(X=None):
    
    c = 1/np.var(X,axis=0)
    
    return c

def get_pre_estimators(X=None, Y=None, model=None, pen=None, n='large'):
    
    c = Estimate_c(X=Y)
    
    if pen == 'Lasso' or pen == 'L0':
        rho_hat = np.ones((Y.shape[1], Y.shape[1]), dtype=float)
    elif pen == 'AdaptiveLasso':
        rho_hat = Estimate_rhohat(Y=Y, Ypred=model.predict(X, verbose=False), n=n)
    else:
        rho_hat = None
        
    return c, model, rho_hat

def get_M(cf = None, masking = None, **kwargs):
    
    if masking is None:
        M = np.ones((cf.shape[1],cf.shape[1]),dtype=float)
        np.fill_diagonal(M, np.inf)
    elif masking == 'hard':
        M = generate_penalty_coef(cf, method = "hard", sparse = kwargs['sparse'])
    elif masking == 'soft':
        M = generate_penalty_coef(cf, method = "soft", sparse = kwargs['sparse'], alpha = kwargs['alpha'])

    return M

# Wenyu's masking function for regularization
def generate_penalty_coef(co, method = "soft",**kwargs):
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
                elif j in tmp:
                    M[i,j] = 1 if co[i,j]!=0 else 0
                    M[j,i] = 1 if co[i,j]!=0 else 0
        M[M==0] = np.inf
        np.fill_diagonal(M,np.inf)
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
                    M[i,j] = cmax/co[i,j] if co[i,j]!=0 else 0
                    M[j,i] = cmax/co[i,j] if co[i,j]!=0 else 0
        M[M==0] = M[M!=0].max()/alpha
        np.fill_diagonal(M, np.inf)
        return M

# Wenyu's shrinkage estimator (from paper cited in LSE)
def shrinkage_estimator(Y = None, to='identity'):
    n,p = Y.shape
    Y -=np.mean(Y,axis=0)
    S = Y.T@Y/n
    if to =="identity":
        m = np.mean(np.diag(S))
        d2 = np.linalg.norm(S - m *np.eye(p))**2/n
        b2 = np.sum([np.linalg.norm(Y[[i],:].T@Y[[i],:] - S)**2 for i in range(n)])/n**3
        b2 = np.minimum(b2,d2)
        return b2/d2*m*np.eye(p)+(d2-b2)/d2*S
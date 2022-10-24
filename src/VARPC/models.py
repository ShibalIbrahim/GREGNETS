from __future__ import division, print_function
import os
import sys
from contextlib import redirect_stdout
import pandas as pd
from tqdm import notebook
import numpy as np
from scipy import sparse as sp
from sklearn.metrics import mean_squared_error
from copy import deepcopy

import utils_gradient
import pre_estimators
import utilities


class VARPC(object):
    """VAR with Partial Correlation (VARPC) with KG-masked regularizers Estimator.

    Attributes:
        G: weighted graph matrix, a float numpy array of shape (N, N).
        regularizer_rho: type of regularization for rho,
          - 'L0L2'
          - 'Lasso'
          - 'AdaptiveLasso'
        masking: type of knowledge graph masking for regularization, str or None.
          - None, No knowledge graph used
          - 'hard', binary mask.
          - 'soft', weighted mask.
        masking_level: number of peers to retain in masking for each firm, scalar int.

        lambda_A: penalty for A, float scalar.
        lambda_rho: penalty for rho, float tuple of length 1 or 2.
          - length 1 tuple when regularizer_rho is 'Lasso' or 'AdaptiveLasso' regularizer_rho.
          - length 2 tuple when regularizer_rho is 'L0L2'.
        eta: learning rate, float scalar.

        X: past-time samples as features, a float numpy array of shape (T, N, p).
        Y: target responses, a float numpy array of shape (T, N).

        A_hat: pre-estimator of VAR component for weighted lasso, a float numpy array of shape (N, N, p).
        rho_hat: pre-estimator of partial correlation for adaptive lasso, a float numpy array of shape (N, N).
        M: weighted masking, a float numpy array of shape (N, N).

        A: parameters of VAR component, a float numpy array of shape (N, N, p).
        c: inverse of conditional variances, a float numpy array of shape (N, ).
        rho: partial correlation parameters (symmetric), a float numpy array of shape (N, N).

    """
    def __init__(self, G, regularizer_rho, masking, masking_level, regularizer_A='AdaptiveLasso', path=None):
        self.G = G
        self.regularizer_rho = regularizer_rho
        self.masking = masking
        self.masking_level = masking_level
        self.regularizer_A = regularizer_A
        if path is not None:
            self.path = path

    def compute_preestimators(self, X, Y, A_eta=10, A_lam=None, sample_size='small'):
        """Computes pre-estimators based on two-stage approach for adaptive regularizer scenarios for joint learning.

        Args:
            sample_size: number of firms in the financial networks dictates sample_size, str.
              - 'small', it may be reasonable to select small when num_time_samples < num_firms.
              - 'large'
        """
        (self.c_hat, self.A_hat, self.rho_hat) = pre_estimators.generate_pre_estimators(X, Y, f_eta=A_eta, f_lam=A_lam, regularizer_rho=self.regularizer_rho, sample_size=sample_size)
        if self.regularizer_A == 'Lasso':
            self.A_hat = np.ones_like(self.A_hat)
        self.mask_rho = pre_estimators.get_mask_partialcorrelation(G=self.G, masking=self.masking, masking_level=self.masking_level, alpha=0)
        self.c = deepcopy(self.c_hat)

    def proximal_gradient_descent(self,
                                  max_iter=10000,
                                  convergence_tolerance=1e-4):
        """Proximal Gradient Descent optimization algorithm.
        """
        logging_file = os.path.join(self.path, "optimization.txt")
        obj = np.inf
        if self.warm_starts==False:
            if self.regularizer_A == 'AdaptiveLasso':
                self.A = deepcopy(self.A_hat)
            elif self.regularizer_A == 'Lasso':
                self.A = np.zeros_like(self.A_hat)
            else:
                raise ValueError("regularizer_A :{} is not supported".format(self.regularizer_A))
            self.rho = np.zeros_like(self.rho_hat)
            self.c = deepcopy(self.c_hat)
        with open(logging_file, 'a') as f:
            with redirect_stdout(f):
                print('Starting optimization for lambda_A:', self.lambda_A, 'lambda_rho:', *(tuple(self.lambda_rho),))
                print('{0:5s}   {1:9s}   {2:9s}'.format('Iter',  'objective',  'relative_objective'))

        print('Starting optimization for lambda_A:', self.lambda_A, 'lambda_rho:', *(tuple(self.lambda_rho),))
        print('{0:5s}   {1:9s}   {2:9s}'.format('Iter',  'objective',  'relative_objective'))
        A_initial = deepcopy(self.A)
        rho_initial = deepcopy(self.rho)
        c_initial = deepcopy(self.c)
        count = 0
        for iteration in range(max_iter):
#             obj_prev = obj
            obj_prev, obj = self.proximal_update()
            if obj > obj_prev:
                obj_rel = np.absolute((obj - obj_prev) / obj)
                with open(logging_file, 'a') as f:
                    with redirect_stdout(f):
                        print('{0:5d}   {1: 3.6f}   {2: 3.6f}'.format(iteration, obj, obj_rel))
                        print('{0:5d}   WARNING!!!! Objective increased!!! Starting from initial solution'.format(iteration))
                print('{0:5d}   {1: 3.6f}   {2: 3.6f}'.format(iteration, obj, obj_rel))
                print('{0:5d}   WARNING!!!! Objective increased!!! Starting from initial solution'.format(iteration))
                self.A = A_initial
                self.rho = rho_initial
                self.c = c_initial
#                 obj = np.inf
                self.eta *= 0.5
                print("Updated learning rate: ", self.eta)
                count = 0
            else:
                obj_rel = np.absolute((obj - obj_prev) / obj)
                if iteration%100==0:
                    with open(logging_file, 'a') as f:
                        with redirect_stdout(f):
                            print('{0:5d}   {1: 3.6f}   {2: 3.6f}'.format(iteration, obj, obj_rel))
                    print('{0:5d}   {1: 3.6f}   {2: 3.6f}'.format(iteration, obj, obj_rel))
                if np.absolute(obj_rel) < convergence_tolerance and count >= 10:
                    with open(logging_file, 'a') as f:
                        with redirect_stdout(f):
                            print('Solution converged in {} iterations with final objective={:.5f} and relative objective={:.8f}<={}'.format(iteration, obj, obj_rel, convergence_tolerance))
                    print('Solution converged in {} iterations with final objective={:.5f} and relative objective={:.8f}<={}'.format(iteration, obj, obj_rel, convergence_tolerance))
                    break
            count += 1
              


    def proximal_update(self):
        """Performs proximal update.

        Returns:
           obj: objective, float scalar.

        Raises:
            ValueError: if unsupported regularizer_rho called.
        """
        assert len(self.X.shape)==3
        T, N, p = self.X.shape

        mse, grad_A, grad_rho, error_residual = utils_gradient.compute_varpc_grad(self.Y, self.X, self.A, self.c, self.rho, self.mask_rho)

        # Compute Objective
        row, col = np.where(np.tril(np.isfinite(self.mask_rho), -1))
        obj = mse + self.lambda_A * np.sum(np.absolute(self.A) / np.absolute(self.A_hat))
        if self.regularizer_rho == 'L0L2':
            obj += self.lambda_rho[0] * np.sum(self.mask_rho[row, col] * (self.rho[row, col] != 0)) + self.lambda_rho[1] * np.sum(np.tril(self.rho, -1)**2)
        elif self.regularizer_rho in ['Lasso', 'AdaptiveLasso']:
            obj += self.lambda_rho[0] * np.sum(np.absolute(self.rho[row, col]) * np.absolute(self.mask_rho[row, col]) / np.absolute(self.rho_hat[row, col]))
        else:
            raise ValueError("{} regularization is not supported.".format(self.regularizer_rho))

        # Update parameters
        # Update A with proximal operation
        A_bar = self.A - self.eta * grad_A
        self.A = np.maximum((np.absolute(A_bar) - self.eta * self.lambda_A * (1 / np.absolute(self.A_hat))),
                       np.zeros_like(self.A)) * np.sign(A_bar) # Soft-thresholding

        # Update rho with proximal operation
        if self.regularizer_rho == 'L0L2':
            grad_rho += 2 * self.lambda_rho[1] * self.rho # Add gradient of L2 regularization for L0L2
            rho_bar = self.rho - self.eta * grad_rho
            rho_bar[(rho_bar**2) <= 2 * self.eta * self.lambda_rho[0] * np.absolute(self.mask_rho)] = 0.0 # Hard-thresholding
            self.rho = rho_bar.copy()
        elif self.regularizer_rho in ['Lasso', 'AdaptiveLasso']:
            rho_bar = self.rho - self.eta * grad_rho
            self.rho = np.maximum(
                (np.absolute(rho_bar) - self.eta * self.lambda_rho[0] * (np.absolute(self.mask_rho) / np.absolute(self.rho_hat))),
                np.zeros_like(self.rho)) * np.sign(rho_bar)# Soft-thresholding
#             self.rho = np.maximum(
#                 np.where(
#                     np.isfinite(np.absolute(self.mask_rho)),
#                     np.absolute(rho_bar) - self.eta * self.lambda_rho[0] * (np.absolute(self.mask_rho) / np.absolute(self.rho_hat)),
#                     -np.inf*np.ones_like(self.rho)
#                 ),
#                 np.zeros_like(self.rho)
#             ) * np.sign(rho_bar)# Soft-thresholding
        else:
            raise ValueError("{} regularization is not supported.".format(self.regularizer_rho))

        # Compute Objective
        T, N, p = self.X.shape
        f = self.X.reshape(T, -1)@self.A.reshape(N, -1).T
        eps = self.Y - f
        Theta = self.rho * np.sqrt(self.c / self.c[:, None])
        error = eps - eps@Theta.T
        mse = np.linalg.norm(error,'fro')**2 / (N * T)
        row, col = np.where(np.tril(np.isfinite(self.mask_rho), -1))
        obj_new = mse + self.lambda_A * np.sum(np.absolute(self.A) / np.absolute(self.A_hat))
        if self.regularizer_rho == 'L0L2':
            obj_new += self.lambda_rho[0] * np.sum(self.mask_rho[row, col] * (self.rho[row, col] != 0)) + self.lambda_rho[1] * np.sum(np.tril(self.rho, -1)**2)
        elif self.regularizer_rho in ['Lasso', 'AdaptiveLasso']:
            obj_new += self.lambda_rho[0] * np.sum(np.absolute(self.rho[row, col]) * np.absolute(self.mask_rho[row, col]) / np.absolute(self.rho_hat[row, col]))
        else:
            raise ValueError("{} regularization is not supported.".format(self.regularizer_rho))

        # Update c
        self.c = 1 / np.var(error, axis=0)


        return obj, obj_new

    def fit(self, X, Y, lambda_A, lambda_rho, eta, max_iter=10000, convergence_tolerance=1e-4, warm_starts=False):
        self.X = X
        self.Y = Y
        self.lambda_A = lambda_A
        self.lambda_rho = lambda_rho
        self.eta = eta
        self.warm_starts = warm_starts
        self.proximal_gradient_descent(max_iter=max_iter, convergence_tolerance=convergence_tolerance)

    def forecast(self, X):
        T, N, p = X.shape
        return X.reshape(T, -1)@self.A.reshape(N, -1).T

    def evaluate(self, X, Y):
        N = Y.shape[1]
        fmse, mse, frsquared, rsquared = utilities.compute_metrics(X=X, Y=Y, f=self.forecast(X), rho=self.rho, c=self.c)
        nnz = (np.count_nonzero(self.rho) / (N * N)) * 100
        return fmse, mse, frsquared, rsquared, nnz
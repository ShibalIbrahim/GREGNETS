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
    def __init__(
        self,
        G,
        regularizer_rho,
        masking,
        masking_level,
        regularizer_A='AdaptiveLasso',
        path=None,
        max_iter=10000,
        convergence_tolerance=1e-4,
        convergence_patience=10,
        no_progress_patience=20,
        warm_start=False,        
    ):
        self.G = G
        self.regularizer_rho = regularizer_rho
        self.masking = masking
        self.masking_level = masking_level
        self.regularizer_A = regularizer_A
        if path is not None:
            self.path = path
        else:
            self.path = "./logs"
        self.logging_file = os.path.join(self.path, "optimization.txt")
        self.max_iter = max_iter
        self.convergence_tolerance = convergence_tolerance
        self.convergence_patience = convergence_patience
        self.no_progress_patience = no_progress_patience
        self.warm_start = warm_start
        self.warm_start_investigated_ = False
 
    def compute_preestimators(self, X, Y, A_eta=10, A_lam=None, sample_size='small'):
        """Computes pre-estimators based on two-stage approach for adaptive regularizer scenarios for joint learning.

        Args:
            sample_size: number of firms in the financial networks dictates sample_size, str.
              - 'small', it may be reasonable to select small when num_time_samples < num_firms.
              - 'large'
        """
        (self.c_, self.A_hat_, self.rho_hat_) = pre_estimators.generate_pre_estimators(X, Y, f_eta=A_eta, f_lam=A_lam, regularizer_rho=self.regularizer_rho, sample_size=sample_size)
        if self.regularizer_A == 'Lasso':
            self.A_hat_ = np.ones_like(self.A_hat_)
        self.mask_rho_ = pre_estimators.get_mask_partialcorrelation(G=self.G, masking=self.masking, masking_level=self.masking_level, alpha=0)

        
    def _restore_initial_optimization_state(self):
        self.A_ = self.A_initial_
        self.rho_ = self.rho_initial_
        self.c_ = self.c_initial_
        self.obj_ = self.obj_initial_

    def _initialize_optimization_state(self):
        if not self.warm_start or not self.warm_start_investigated_:
            with open(self.logging_file, 'a') as f:
                with redirect_stdout(f):
                    print('Warm start either not set or it is yet to warm up.')
                    print('Initializing using arguments from init or as a default zero matrix')
            print('Warm start either not set or it is yet to warm up.')
            print('Initializing using arguments from init or as a default zero matrix')
            self.A_ = np.zeros_like(self.A_hat_)
            self.rho_ = np.zeros_like(self.rho_hat_)
            self.warm_start_investigated_ = True
            self.obj_ = np.inf
        self.A_initial_ = deepcopy(self.A_)
        self.rho_initial_ = deepcopy(self.rho_)
        self.c_initial_ = deepcopy(self.c_)
        self.obj_ = np.inf
        self.obj_initial_ = self.obj_
        return

    def _save_optimal_state(self):
        self.A_opt_ = deepcopy(self.A_)
        self.rho_opt_ = deepcopy(self.rho_)
        self.c_opt_ = deepcopy(self.c_)
        self.obj_opt_ = self.obj_

    def _restore_optimal_state(self):
        self.A_ = self.A_opt_
        self.rho_ = self.rho_opt_
        self.c_ = self.c_opt_
        self.obj_ = self.obj_opt_

    def _reset_progress(self):
        self.convergence_count_ = 0
        self.no_progress_count_ = 0        

    def _compute_objective(self, mse):
        row, col = np.where(np.tril(np.isfinite(self.mask_rho_), -1))
        obj = mse + self.lambda_A * np.sum(np.absolute(self.A_) / np.absolute(self.A_hat_))
        if self.regularizer_rho in ['Lasso', 'AdaptiveLasso']:
            obj += (self.lambda_rho * np.sum(np.absolute(self.rho_[row, col])
                                                * np.absolute(self.mask_rho_[row, col])
                                                / np.absolute(self.rho_hat_[row, col])))
        else:
            raise ValueError("{} regularization is not supported.".format(self.regularizer_rho))
        return obj


    def _proximal_gradient_descent(self, x, y):
        # Initialize progress parameters, optimization state, and save optimal (initial) state.
        max_iter = self.max_iter
        convergence_tolerance = self.convergence_tolerance
        self._reset_progress()
        self._initialize_optimization_state()
        self._save_optimal_state()
        # Perform proximal gradient descent, and check convergence
        count = 0
        
        with open(self.logging_file, 'a') as f:
            with redirect_stdout(f):
                print('Starting optimization for lambda_A:', self.lambda_A, 'lambda_rho:', self.lambda_rho)
                print('{0:5s}   {1:9s}   {2:9s}   {3:2s}'.format('Iter',  'objective',  'convergence_count', 'no_progress_count'))
    
        print('Starting optimization for lambda_A:', self.lambda_A, 'lambda_rho:', self.lambda_rho)
        print('{0:5s}   {1:9s}   {2:9s}   {3:2s}'.format('Iter',  'objective',  'convergence_count', 'no_progress_count'))
        
        for iteration in range(max_iter):
            self.obj_, obj_intermediate = self._proximal_update(x, y)
            if obj_intermediate > self.obj_:
                with open(self.logging_file, 'a') as f:
                    with redirect_stdout(f):
                        print('{0:5d}   WARNING!!!! Objective increased!!! Starting from initial solution'.format(iteration))
                print('{0:5d}   WARNING!!!! Objective increased!!! Starting from initial solution'.format(iteration))
                
                self._restore_initial_optimization_state()
                self._reset_progress()
                self.eta *= 0.5
                with open(self.logging_file, 'a') as f:
                    with redirect_stdout(f):
                        print('Updated learning rate: {}'.format(self.eta))
                print('Updated learning rate: {}'.format(self.eta))
            elif count >= self.convergence_patience:    # Run for minimum these many before checking progress
                obj_rel_global_progress = (self.obj_opt_ - self.obj_) / np.maximum(np.absolute(self.obj_), 1.0)
                if self.obj_ < self.obj_opt_:
                    self.no_progress_count_ = 0
                    self._save_optimal_state()
                    if obj_rel_global_progress <= convergence_tolerance:
                        self.convergence_count_ += 1
                    if self.convergence_count_ >= self.convergence_patience:
                        with open(self.logging_file, 'a') as f:
                            with redirect_stdout(f):
                                print("""Solution converged in {} iterations with final
                                            objective={:.9f}, slow progress with global relative
                                            objective={:.9f}<={} for {} non-consecutive
                                            iterations""".format(iteration, self.obj_,
                                                                 obj_rel_global_progress,
                                                                 convergence_tolerance,
                                                                 self.convergence_count_))
                        print("""Solution converged in {} iterations with final
                                            objective={:.9f}, slow progress with global relative
                                            objective={:.9f}<={} for {} non-consecutive
                                            iterations""".format(iteration, self.obj_,
                                                                 obj_rel_global_progress,
                                                                 convergence_tolerance,
                                                                 self.convergence_count_))
                        
                        
                        
                        break   # return solution
                else:
                    self.no_progress_count_ += 1
                    if self.no_progress_count_ == self.no_progress_patience:
                        with open(self.logging_file, 'a') as f:
                            with redirect_stdout(f):
                                print("""Solution converged in {} iterations with final
                                            objective={:.9f}, no progress for consecutive {} 
                                            iterations and relative
                                            objective={:.9f}""".format(iteration, self.obj_,
                                                                       self.no_progress_count_,
                                                                       obj_rel_global_progress))
                        print("""Solution converged in {} iterations with final
                                            objective={:.9f}, no progress for consecutive {} 
                                            iterations and relative
                                            objective={:.9f}""".format(iteration, self.obj_,
                                                                       self.no_progress_count_,
                                                                       obj_rel_global_progress))
                        break   # return solution
            count += 1
            if iteration % 10 == 0:
                with open(self.logging_file, 'a') as f:
                    with redirect_stdout(f):
                        print('{0:5d}   {1: 3.9f}   {2: 2.0f}  {3: 2.0f}'.format(iteration,
                                                                                self.obj_,
                                                                                self.convergence_count_,
                                                                                self.no_progress_count_))
                print('{0:5d}   {1: 3.9f}   {2: 2.0f}  {3: 2.0f}'.format(iteration,
                                                                                self.obj_,
                                                                                self.convergence_count_,
                                                                                self.no_progress_count_))
        with open(self.logging_file, 'a') as f:
            with redirect_stdout(f):
                print('{0:5d}   {1: 3.9f}   {2: 2.0f}  {3: 2.0f}'.format(iteration,
                                                                         self.obj_,
                                                                         self.convergence_count_,
                                                                         self.no_progress_count_))
        print('{0:5d}   {1: 3.9f}   {2: 2.0f}  {3: 2.0f}'.format(iteration,
                                                                 self.obj_,
                                                                 self.convergence_count_,
                                                                 self.no_progress_count_))
                
        self._restore_optimal_state()
        return
                
    def _proximal_update(self, x, y):
        mse, error_residual, grad_A, grad_rho = utils_gradient.compute_varpc_errors_and_gradient(y, x, self.A_,
                                                                                  self.c_, self.rho_,
                                                                                  self.mask_rho_,
                                                                                  compute_gradient=True)
        obj = self._compute_objective(mse)
        # Update A with proximal operation
        A_bar = self.A_ - self.eta * grad_A
        self.A_ = np.maximum((np.absolute(A_bar) - self.eta * self.lambda_A * (1 / np.absolute(self.A_hat_))),
                             np.zeros_like(self.A_)) * np.sign(A_bar)  # Soft-thresholding
        # Update rho with proximal operation
        if self.regularizer_rho in ['Lasso', 'AdaptiveLasso']:
            rho_bar = self.rho_ - self.eta * grad_rho
            self.rho_ = np.maximum((np.absolute(rho_bar) - self.eta * self.lambda_rho
                                    * (np.absolute(self.mask_rho_) / np.absolute(self.rho_hat_))),
                                   np.zeros_like(self.rho_)) * np.sign(rho_bar)  # Soft-thresholding
        else:
            raise ValueError("{} regularization is not supported.".format(self.regularizer_rho))
        # Compute Updated Objective
        mse, error_residual, *_ = utils_gradient.compute_varpc_errors_and_gradient(y, x, self.A_, self.c_, self.rho_,
                                                    self.mask_rho_, compute_gradient=False)
        obj_new = self._compute_objective(mse)
        # Update c (c update is outside the objective change condition because
        # monotonicity of objective is not guaranteed with probablistic model-based update for c)
        self.c_ = 1 / np.var(error_residual, axis=0)

        return obj, obj_new
    
    def fit(self, X, y, lambda_A, lambda_rho, eta):
        """fit function.

        Args:
            X (np.ndarray): Sequences constructed out of multivariate time series
                and is a float numpy array of shape (T, N, p) where T is the size of time-series,
                N is the number of variables, and p dictates the number of past values each univariate
                time-series can look into.
            y (np.ndarray): Output values of shape (T, N) where T is the size of time-series,
                N is the number of variables.

        Returns:
            self
        """
        self.lambda_A = lambda_A
        self.lambda_rho = lambda_rho
        self.eta = eta
        self._proximal_gradient_descent(X, y)
        return self

    def forecast(self, x):
        """Forecast values using learned parameters.

        Args:
            x (np.ndarray): Sequences constructed out of multivariate time series
                and is a float numpy array of shape (T, N, p) where T is the size of time-series,
                N is the number of variables, and p dictates the number of past values each univariate
                time-series can look into.

        Returns:
            np.ndarray: Forecasts
        """
        T, N, p = x.shape
        return x.reshape(T, -1) @ self.A_.reshape(N, -1).T
    
        
    def evaluate(self, x, y):
        """Evaluate performance of learned model.

        Args:
            x (np.ndarray): Sequences constructed out of multivariate time series
                and is a float numpy array of shape (T, N, p) where T is the size of time-series,
                N is the number of variables, and p dictates the number of past values each univariate
                time-series can look into.
            y (np.ndarray): Output values of shape (T, N) where T is the size of time-series,
                N is the number of variables.

        Returns:
            tuple: A tuple of FMSE, MSE, f_squared, and r_squared
        """
        N = y.shape[1]
        _, mse, _, rsquared = utilities.compute_metrics(
            X=x,
            Y=y,
            f=self.forecast(x),
            rho=self.rho_,
            c=self.c_
        )
        nnz = (np.count_nonzero(self.rho_) / (N * N)) * 100
        return mse, rsquared, nnz
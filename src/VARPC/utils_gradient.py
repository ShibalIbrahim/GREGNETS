"""Gradient calculation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as sp

def masked_matmul(A, B, W):
    """Performs sparse (masked) matrix multiplication.

    Args:
        A: First Matrix, a numpy array of shape (l, m)
        B: Second Matrix, a numpy array of shape (m, n)
        W: boolean matrix defining the entries that need to be computed in the matrix multiplication A x B,
            a numpy array of shape (l, n).
    """
    row, col, _ = sp.find(W)
    # Get the sum-reduction using valid rows and corresponding cols from A, B
    out = np.einsum('ij, ji -> i', A[row], B[:, col])

    # Store as sparse matrix
    out_sparse = sp.coo_matrix((out, (row, col)), shape=W.shape)
    return out_sparse


def compute_varpc_errors_and_gradient(Y, X, A, c, rho, M,
                                      compute_gradient=True,
                                      sparse=False):
    """Computes Closed-form gradients of VARPC.
    
    Args:
        Y (np.ndarray): target responses, a float numpy array of shape (T, N).
        X (np.ndarray): past-time samples as features, a float numpy array
            of shape (T, N, p).
        A (np.ndarray): parameters of VAR component, a float numpy array
            of shape (N, N, p).
        c (np.ndarray): inverse of conditional variances, a float
            numpy array of shape (N, ).
        rho (np.ndarray): partial correlation parameters (symmetric),
            a float numpy array of shape (N, N).
        M (np.ndarray): weighted masking, a float numpy array of shape (N, N).
        compute_gradient (bool): Whether to compute gradients or not.
        sparse (bool): Whether to use sparse operations to compute gradients.

    Returns:
        tuple: A tuple consisiting of mse (pseudolikelihood loss, a float scalar),
            unnormalized_error (a float numpy array of shape (T, N)), grad_A
            (gradient of VAR parameters, a float numpy array of shape (N, N, p)),
            grad_rho (gradient of partial correlation parameters, a float
            numpy array of shape (N, N)). Note that grad_A and grad_rho are both
            None if compute_gradient is set to False.
    """
    T, N, p = X.shape
    f = X.reshape(T, -1) @ A.reshape(N, -1).T
    eps = Y - f
    Theta = rho * np.sqrt(c / c[:, None])
    unnormalized_error = eps - eps @ Theta.T
    mse = np.linalg.norm(unnormalized_error, 'fro') ** 2 / (N * T)

    # Gradient of partial correlation
    if compute_gradient:
        # The "if" block below uses sparse operations, we currently use dense
        # operations to compute grad_rho by default.
        if sparse:
            grad_rho = (-np.sqrt(c / c[:, None])
                        * ((masked_matmul(unnormalized_error.T, eps, np.isfinite(M))).toarray())
                        ) * 2 / (N * T)
        else:
            grad_rho = (-np.sqrt(c / c[:, None])
                        * np.where(np.isfinite(M),
                                   np.matmul(unnormalized_error.T, eps), 0)
                        ) * 2 / (N * T)
        grad_rho += grad_rho.transpose()
        np.fill_diagonal(grad_rho, 0.)
        # Gradient of VAR parameters
        grad_A = np.tensordot((unnormalized_error@(Theta - np.eye(N))* 2 / (N * T)).T,
                              X, axes=(1, 0))
    else:
        grad_A, grad_rho = None, None

    return mse, unnormalized_error, grad_A, grad_rho

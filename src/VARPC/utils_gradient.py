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


def compute_varpc_grad(Y, X, A, c, rho, M):
    """Computes Closed-form gradients of VARPC.

    Args:
        Y: target responses, a float numpy array of shape (T, N).
        X: past-time samples as features, a float numpy array of shape (T, N, p).
        A: parameters of VAR component, a float numpy array of shape (N, N, p).
        c: inverse of conditional variances, a float numpy array of shape (N, ).
        rho: partial correlation parameters (symmetric), a float numpy array of shape (N, N).
        M: weighted masking, a float numpy array of shape (N, N).

    Returns:
        mse: pseudolikelihood loss, a float scalar.
        grad_A: gradient of VAR parameters, a float numpy array of shape (N, N, p).
        grad_rho: gradient of partial correlation parameters, a float numpy array of shape (N, N).
        error: residual, a float numpy array of shape (T, N).
    """
    T, N, p = X.shape
    f = X.reshape(T, -1)@A.reshape(N, -1).T
    eps = Y - f
    Theta = rho * np.sqrt(c / c[:, None])
    error = eps - eps@Theta.T
    mse = np.linalg.norm(error,'fro')**2 / (N * T)

    # Gradient of partial correlation
    grad_rho = (-np.sqrt(c / c[:, None]) * np.matmul(error.T, eps)*np.isfinite(M)) * 2 / (N * T)
#     grad_rho = (-np.sqrt(c / c[:, None]) * ((masked_matmul(error.T, eps, np.isfinite(M))).toarray())) * 2 / (N * T)
    grad_rho += grad_rho.transpose()
    np.fill_diagonal(grad_rho, 0.)

    # Gradient of VAR parameters
    grad_A = np.tensordot((error@(Theta - np.eye(N))* 2 / (N * T)).T,
                          X,
                          axes=(1,0))

    return mse, grad_A, grad_rho, error


def compute_gcnpc_grad(Y, GX, w, c, rho, M):
    """Computes Closed-form gradients of GCNPC.

    Args:
        Y: target responses, a float numpy array of shape (T, N).
        GX: graph processed past-time samples, a float numpy array of shape (N, T, p).
        w: parameters of GCN, a float numpy array of shape (p, ).
        c: inverse of conditional variances, a float numpy array of shape (N, ).
        rho: partial correlation parameters (symmetric), a float numpy array of shape (N, N).
        M: weighted masking, a float numpy array of shape (N, N).

    Returns:
        mse: pseudolikelihood loss, a float scalar.
        grad_W: gradient of GCN parameters, a float numpy array of shape (p, ).
        grad_rho: gradient of partial correlation parameters, a float numpy array of shape (N, N).
        error: error residual, a float numpy array of shape (T, N).
    """
    T, N = Y.shape
    f = np.tensordot(GX, w, axes=([2], [0])).transpose()
    eps = Y - f
    Theta = rho * np.sqrt(c / c[:, None])
    error = eps - eps@Theta.T
    mse = np.linalg.norm(error,'fro')**2 / (N * T)

    # Gradient of partial correlation
    grad_rho = (-np.sqrt(c / c[:, None]) * np.matmul(error.T, eps)*np.isfinite(M)) * 2 / (N * T)
#     grad_rho = (-np.sqrt(c / c[:, None]) * masked_matmul(error.T, eps, np.isfinite(M)).toarray()) * 2 / (N * T)
    grad_rho += grad_rho.transpose()
    np.fill_diagonal(grad_rho, 0.)

    # Gradient of GCN parameters
    grad_w = 2 / (N * T) * np.tensordot(
        error,
        np.tensordot(Theta - np.eye(N), GX, axes=([1], [0])),
        axes=([0, 1], [1, 0])
    )
    return mse, grad_w, grad_rho, error

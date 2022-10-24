"""Tests for gradient calculation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

import utils_gradient

_SEED = 1
np.random.seed(_SEED)

#
N = 30
T = 100
p = 5

# Initialize correlation
c = np.abs(np.random.randn(N)) * 5
rho = np.random.rand(N, N) * 2 - 1
rho = (rho + rho.T) / 2
np.fill_diagonal(rho, 0.)

# Define masking matrix
M = np.ones((N,N))
r = (int)(np.ceil(0.2 * N * N))
indices_i = np.random.randint(0, high=N, size=r)
indices_j = np.random.randint(0, high=N, size=r)
M[indices_i, indices_j] = np.inf
M[indices_j, indices_i] = np.inf
np.fill_diagonal(M, np.inf)
rho[indices_i, indices_j] = 0. # updated to ensure rho is 0 in masked entries
rho[indices_j, indices_i] = 0. # updated to ensure rho is 0 in masked entries
np.fill_diagonal(rho, 0.) # updated to ensure rho is 0 in masked entries

# Define input, output
X = np.random.randn(T, N, p)
Y = np.random.randn(T, N)

# Initialize VAR
A = np.random.randn(N, N, p)

# Initialize GCN
G = np.random.randn(N,N)
w = np.random.randn(p)


def compute_varpc_grad_tf(Y, X, A, c, rho, M):
    """Computes gradients of VARPC using Tensorflow Gradient Tape.

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
    """
    _rho = tf.Variable(rho, dtype=tf.float64)
    _A = tf.Variable(A, dtype=tf.float64)
    _M = tf.constant(M, dtype=tf.float64)
    _boolean_mask = tf.where(tf.math.is_finite(_M),
                     tf.ones_like(M),
                     tf.zeros_like(M)
                    )

    @tf.function
    def pseudolikelihood_objective(x, y):
        Theta = tf.multiply(tf.multiply(_rho, _boolean_mask), np.sqrt(c / c[:, None]))
        f = tf.matmul(tf.reshape(x, [T, -1]), tf.reshape(_A, [N, -1]), transpose_b=True)
        eps = y - f
        u = eps - tf.matmul(eps, Theta, transpose_b=True)
        return tf.reduce_mean(tf.square(u))
    with tf.GradientTape() as tape:
        mse = pseudolikelihood_objective(X, Y)
        grad_A = tape.gradient(mse, _A)
    with tf.GradientTape() as tape:
        mse = pseudolikelihood_objective(X, Y)
        grad_rho = tape.gradient(mse, _rho)

    mse = mse.numpy()
    grad_A = grad_A.numpy()
    grad_rho = grad_rho.numpy()
    np.fill_diagonal(grad_rho, 0.)
    grad_rho += grad_rho.T
    return mse, grad_A, grad_rho


def compute_gcnpc_grad_tf(Y, GX, w, c, rho, M):
    """Computes gradients of GCNPC using Tensorflow Gradient Tape.

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
    """
    _rho = tf.Variable(rho, dtype=tf.float64)
    _w = tf.Variable(w, dtype=tf.float64)
    _M = tf.constant(M, dtype=tf.float64)
    _boolean_mask = tf.where(tf.math.is_finite(_M),
                     tf.ones_like(M),
                     tf.zeros_like(M)
                    )
    @tf.function
    def pseudolikelihood_objective(gx, y):
        Theta = tf.multiply(tf.multiply(_rho, _boolean_mask), np.sqrt(c / c[:,None]))
        f = tf.transpose(tf.tensordot(gx, _w, axes=[[2], [0]]))
        eps = y - f
        u = eps - tf.matmul(eps, Theta, transpose_b=True)
        return tf.reduce_mean(tf.square(u))
    with tf.GradientTape() as tape:
        mse = pseudolikelihood_objective(GX, Y)
        grad_w = tape.gradient(mse, _w)
    with tf.GradientTape() as tape:
        mse = pseudolikelihood_objective(GX, Y)
        grad_rho = tape.gradient(mse, _rho)

    mse = mse.numpy()
    grad_w = grad_w.numpy()
    grad_rho = grad_rho.numpy()
    np.fill_diagonal(grad_rho, 0.)
    grad_rho += grad_rho.T
    return mse, grad_w, grad_rho


class GradientsTest(tf.test.TestCase):

    def testThetaCalculation(self):
        Theta_old = np.array([[rho[i, h] * np.sqrt(c[h] / c[i]) for h in range(N)] for i in range(N)])
        Theta = rho * np.sqrt(c / c[:, None])
        np.testing.assert_array_almost_equal(Theta, Theta_old)

    def testVARCalculation(self):
        f_old = np.sum([A[:, :, i]@X[:, :, i].T for i in range(p)], axis=0).T
        f = X.reshape(T, -1)@A.reshape(N, -1).T
        np.testing.assert_array_almost_equal(f, f_old, decimal=1e-16)

    def testGradientsVARPC(self):
        mse, grad_A, grad_rho, _ = utils_gradient.compute_varpc_grad(Y, X, A, c, rho, M)
        mse_tf, grad_A_tf, grad_rho_tf = compute_varpc_grad_tf(Y, X, A, c, rho, M)
        np.testing.assert_array_almost_equal(mse, mse_tf)
        np.testing.assert_array_almost_equal(grad_A, grad_A_tf, decimal=1e-16)
        np.testing.assert_array_almost_equal(grad_rho, grad_rho_tf, decimal=1e-16)

    def testGradientsGCNPC(self):
        GX = np.tensordot(G, np.transpose(X, axes=[1, 0, 2]), axes=([1],[0]))
        mse, grad_w, grad_rho, _ = utils_gradient.compute_gcnpc_grad(Y, GX, w, c, rho, M)
        mse_tf, grad_w_tf, grad_rho_tf = compute_gcnpc_grad_tf(Y, GX, w, c, rho, M)
        np.testing.assert_array_almost_equal(mse, mse_tf)
        np.testing.assert_array_almost_equal(grad_w, grad_w_tf, decimal=1e-16)
        np.testing.assert_array_almost_equal(grad_rho, grad_rho_tf, decimal=1e-16)

if __name__ == '__main__':
    tf.test.main()

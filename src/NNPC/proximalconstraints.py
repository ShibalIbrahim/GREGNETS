"""Proximal/Projected Operations using tensorflow constraints.

Classes from the tf.keras.constraints module allow setting constraints (eg. non-negativity) on model parameters during training. They are per-variable projection functions applied to the target variable after each gradient update (when using fit()).
"""
import tensorflow as tf
import numpy as np

class Lasso(tf.keras.constraints.Constraint):
    """Proximal operation for masked Lasso/Adaptive Lasso regularizers.

    This constraint function can only work correctly with Stochastic Gradient Descent as proximal operator requires 
    knowledge of learning rate. 

    Attributes:
    w_hat: adaptive weights matrix from pre-estimation, (num nodes, num nodes)
    weighted_mask: weighted masking weight matrix, (num nodes, num nodes). 
    regularization_weight: regularization weight, scalar float.
    learning_rate: learning rate, scalar float.

    Call arguments:
    w: gradient updated weights.
    """
    def __init__(self, w_hat, weighted_mask, regularization_weight, learning_rate):
        assert np.all(w_hat.shape == weighted_mask.shape), "Shapes for w_hat: %s and weighted_mask :%s are incompatible."%(w_hat.shape, weighted_mask.shape)
        self.w_hat = tf.constant(w_hat, dtype=tf.float32)
        self.weighted_mask = tf.constant(weighted_mask, dtype=tf.float32)
        self.zeros = tf.zeros_like(w_hat, dtype=tf.float32)
        self.regularization_weight = regularization_weight
        self.learning_rate = learning_rate

    def __call__(self, w):
        w = tf.maximum(
          tf.abs(w) - self.learning_rate * self.regularization_weight * tf.divide(
              tf.abs(self.weighted_mask), 
              tf.abs(self.w_hat)
          ), 
          self.zeros
        ) * tf.sign(w)
        return w

    def get_config(self):
        return {'w_hat': self.w_hat.numpy(),
                'weighted_mask': self.weighted_mask.numpy().
                'regularization_weight': self.regularization_weight,
                'learning_rate': self.learning_rate}


class L0(tf.keras.constraints.Constraint):
    """Proximal operation for masked L0 regularizer.

    This constraint function can only work correctly with Stochastic Gradient Descent as proximal operator requires 
    knowledge of learning rate. 

    Attributes:
    weighted_mask: weighted masking weight matrix, (num nodes, num nodes). 
    regularization_weight: regularization weight, scalar float.
    learning_rate: learning rate, scalar float.

    Call arguments:
    w: gradient updated weights.
    """
    def __init__(self, w_hat, weighted_mask, regularization_weight, learning_rate):
        self.weighted_mask = tf.constant(weighted_mask, dtype=tf.float32)
        self.zeros = tf.zeros_like(weighted_mask, dtype=tf.float32)
        self.regularization_weight = regularization_weight
        self.learning_rate = learning_rate

    def __call__(self, w):
        w = tf.where(
          tf.math.square(w) <= 2 * self.learning_rate * self.regularization_weight * tf.abs(self.weighted_mask),
          self.zeros,
          w
        )
        return w

    def get_config(self):
        return {'weighted_mask': self.weighted_mask.numpy().
                'regularization_weight': self.regularization_weight,
                'learning_rate': self.learning_rate}
"""Includes Keras layers for time-series forecasting under (Generalized) Pseudolikelihood.


Creates custom Keras layers for Multivariate Time-Series Regression under (generalized)
pseudolikelihood.
"""

import tensorflow as tf
import numpy as np
import json

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
from tensorflow.keras import activations
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.python.keras.utils import tf_utils

class VAR(Layer):
    """Vector Autoregressive layer.
    
    This is a Keras implementation of Vector Autoregressive Layer for the task of time-series forecasting in a network. 
    
    `VAR` implements the operation:
    `output = X@A + bias `
    with shapes:
      X: (num samples, num nodes, num steps)
      A: (num steps, num nodes, num nodes)
        
    Attributes:
      steps: num input steps.
      use_bias: whether the layer uses a bias, scalar boolean.
      kernel_initializer: Initializer for the 'kernel' weights matrix.
      bias_initializer: Initializer for the bias.
      kernel_regularizer: Regularizer function applied to the "kernel" weights matrix
      bias_regularizer: Regularizer function applied to the bias.
      kernel_constraint: Constraint function applied to the "kernel" weights matrices for all supports.
      bias_constraint: Constraint function applied to the bias vector.
      
    Input shape:
      N-D tensor with shape: (batch_size, num_nodes, input_dim).
    
    Output shape:
      N-D tensor with shape: (batch_size, num_nodes, units).
    """
    def __init__(self, 
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(VAR, self).__init__(**kwargs)        
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not dtype.is_floating:
            raise TypeError('Unable to build `VAR` layer with non-floating point dtype %s' % (dtype, )) 
        
        input_shape = tensor_shape.TensorShape(input_shape)
        num_nodes = tensor_shape.dimension_value(input_shape[-2])
        steps = tensor_shape.dimension_value(input_shape[-1])
        if steps is None:
            raise ValueError('The last dimension of the inputs to `VAR` should be defined, Found `None`.')
        
        # Initialize weights for each support
        self.vars = {}
        self.vars['kernel_A'] = self.add_weight(name='kernel_A', 
                                                shape=(steps, num_nodes, num_nodes),
                                                initializer=self.kernel_initializer,
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint,
                                                dtype=self.dtype,
                                                trainable=True)
#         for s in range(steps):
#             self.vars['kernel_A_{}'.format(s)] = self.add_weight(name='kernel_A_{}'.format(s), 
#                                                     shape=(num_nodes, num_nodes),
#                                                     initializer=self.kernel_initializer,
#                                                     regularizer=self.kernel_regularizer,
#                                                     constraint=self.kernel_constraint,
#                                                     dtype=self.dtype,
#                                                     trainable=True)
#         self.vars['kernel_A'] = tf.stack([self.vars['kernel_A_{}'.format(s)] for s in range(steps)]) 
        if self.use_bias:
            self.vars['bias'] = self.add_weight(name='bias', 
                                                shape=(1,), 
                                                initializer=self.bias_initializer,
                                                regularizer=self.bias_regularizer,
                                                constraint=self.bias_constraint,
                                                dtype=self.dtype,
                                                trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        output = tf.tensordot(inputs, self.vars['kernel_A'], [[2, 1], [0, 1]]) # X.reshape(T, -1)@A.reshape(N, -1).T
        if self.use_bias:
            output += self.vars['bias']
        return output
    
    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The innermost dimension of input_shape must be defined, but saw: %s' % input_shape)
        return input_shape[:-1]
    
    def get_config(self):
        config = super(VAR, self).get_config()
        config.update({
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        })
        return config

    
class GraphConvolution(Layer):
    """Graph Convolution layer.
    
    This is a Keras implementation of Graph Convolutional Layer for the task of time-series forecasting in a network. This layer
    handles multiple (time) samples at each node in the network, not considered by the original authors of Graph Convolution 
    Networks. 
    
    `GraphConvolution` implements the operation:
    `output = activation( sum (dot(G, X, W) for all supports) + bias )`
    with shapes:
      G: (num nodes, num nodes)
      X: (num nodes, num samples, num steps)
      W: (num steps, num hidden steps)
    
    References:
      - ["Semi-Supervised Classification with Graph Convolutional Networks",
         by Thomas N. Kipf, Max Welling ICLR 2017.]
         (http://arxiv.org/abs/1609.02907)
    
    Attributes:
      units: num hidden/output steps.
      supports: list of knowledge graphs' adjacency matrices, [(num nodes, num nodes)] 
      activation: Activation function to use.
        If you don't specify anything, no activation is applied (i.e. "linear" activation 'a(x) = x').
      use_bias: whether the layer uses a bias, scalar boolean.
      kernel_initializer: Initializer for the 'kernel' weights matrix.
      bias_initializer: Initializer for the bias.
      kernel_regularizer: Regularizer function applied to the "kernel" weights matrix
      bias_regularizer: Regularizer function applied to the bias.
      kernel_constraint: Constraint function applied to the "kernel" weights matrices for all supports.
      bias_constraint: Constraint function applied to the bias vector.
      
    Input shape:
      N-D tensor with shape: (batch_size, num_nodes, input_dim).
    
    Output shape:
      N-D tensor with shape: (batch_size, num_nodes, units).
    """
    def __init__(self, 
                 units=None,
                 supports=None,
                 activation='relu',
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)        
        self.units = units
        self.supports = [tf.constant(support, dtype=tf.float32) for support in supports]
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not dtype.is_floating:
            raise TypeError('Unable to build `GraphConvolution` layer with non-floating point dtype %s' % (dtype, )) 
        
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `GraphConvolution` should be defined, Found `None`.')
        
        # Initialize weights for each support
        self.vars = {}
        for i in range(len(self.supports)):
            self.vars['kernel_' + str(i)] = self.add_weight(name='kernel_' + str(i), 
                                                            shape=(last_dim, self.units),
                                                            initializer=self.kernel_initializer,
                                                            regularizer=self.kernel_regularizer,
                                                            constraint=self.kernel_constraint,
                                                            dtype=self.dtype,
                                                            trainable=True)                                    
        if self.use_bias:
            self.vars['bias'] = self.add_weight(name='bias', 
                                                shape=(self.units,), 
                                                initializer=self.bias_initializer,
                                                regularizer=self.bias_regularizer,
                                                constraint=self.bias_constraint,
                                                dtype=self.dtype,
                                                trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        inputs = tf.transpose(inputs, perm=[1,0,2])
        supports = list()
        for i in range(len(self.supports)):
            pre_sup = tf.keras.backend.dot(inputs, self.vars['kernel_' + str(i)])
            support = tf.transpose(tf.keras.backend.dot(tf.transpose(pre_sup), tf.transpose(self.supports[i])))
            supports.append(support)
        output = tf.add_n(supports)

        if self.use_bias:
            output += self.vars['bias']
        
        output = self.activation(output)
        output = tf.transpose(output, perm=[1,0,2])
        return output
    
    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The innermost dimension of input_shape must be defined, but saw: %s' % input_shape)
        return input_shape[:-1].concatenate(self.units)
    
    def get_config(self):
        config = super(GraphConvolution, self).get_config()
        config.update({
            'units': self.units,
            'supports': [supp.numpy() for supp in self.supports],
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        })
        return config
    
    
class NGraphConvolution(Layer):
    """N-Graph Convolution layer.
    This is a Keras implementation of N-Graph Convolutional Layer for the task of time-series forecasting in a network. This 
    layer handles multiple (time) samples at each node in the network, not considered by the original authors of NGCN.
    
    `NGraphConvolution` implements the operation:
    `output =  dense (dot(G^j, X, W_j) for j \in K) + bias)
    with shapes:
      G: (num nodes, num nodes)
      X: (num nodes, num samples, num steps)
      W: (num steps, num hidden steps)
    
    References:
      - ["N-GCN: Multi-scale Graph Convolution for Semi-supervised Node Classification", 
         by Sami Abu-El-Haija, Bryan Perozzi, Amol Kapoor UAI 2019.]
         (https://arxiv.org/pdf/1802.08888.pdf)
    
    Attributes:
      units: num hidden/output steps.
      supports: list of knowledge graphs' adjacency matrices, [(num nodes, num nodes)] 
      activation: Activation function to use.
        If you don't specify anything, no activation is applied (i.e. "linear" activation 'a(x) = x').
      use_bias: whether the layer uses a bias, scalar boolean.
      kernel_initializer: Initializer for the 'kernel' weights matrices.
      bias_initializer: Initializer for the biases.
      kernel_regularizer: Regularizer function applied to the "kernel" weights matrix
      bias_regularizer: Regularizer function applied to the bias.
      kernel_constraint: Constraint function applied to the "kernel" weights matrices for all supports.
      bias_constraint: Constraint function applied to the bias vector.
      
    Input shape:
      N-D tensor with shape: (batch_size, num_nodes, input_dim).
    
    Output shape:
      N-D tensor with shape: (batch_size, num_nodes, units).
    """
    def __init__(self,
                 latent_units=None,
                 units=None,
                 supports = None,
                 activation='relu',
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(NGraphConvolution, self).__init__(**kwargs)
        
        self.latent_units = latent_units
        self.units = units
        self.supports = [tf.constant(supp, dtype=tf.float32) for supp in supports]
        self.num_of_supports = len(supports)
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        
    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not dtype.is_floating:
            raise TypeError('Unable to build `NGraphConvolution` layer with non-floating point dtype %s' % (dtype, )) 
        
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `NGraphConvolution` should be defined, Found `None`.')

        # Initialize weights for each support
        self.vars = {}
        for i in range(self.num_of_supports):
            self.vars['kernel_' + str(i)] = self.add_weight(
                name='kernel_' + str(i), 
                shape=(last_dim, self.latent_units), 
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True
            )
                                                         
            if self.use_bias:
                self.vars['bias_' + str(i)] = self.add_weight(
                    name='bias_' + str(i), 
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    shape=(self.latent_units,), 
                    trainable=True
                )
        self.vars['kernel_dense'] = self.add_weight(
            name='kernel_dense', 
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            shape=(self.num_of_supports*self.latent_units, self.units), 
            trainable=True
        )
        self.vars['bias_dense'] = self.add_weight(
            name='bias_dense', 
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            shape=(self.units,), 
            trainable=True
        )
        super(NGraphConvolution, self).build(input_shape)

    def call(self, inputs, **kwargs):
        batch_size = K.int_shape(inputs)[0]
        num_nodes = K.int_shape(inputs)[1]
        x = tf.transpose(inputs, perm=[1,0,2])
        
        # convolve
        supports = [None]*self.num_of_supports
        for i in range(self.num_of_supports):
            pre_sup = tf.keras.backend.dot(x, self.vars['kernel_' + str(i)])
            support = tf.transpose(tf.keras.backend.dot(tf.transpose(pre_sup), tf.transpose(self.supports[i])))
            if self.use_bias:
                supports[i] = support+self.vars['bias_' + str(i)] 
            else:
                supports[i] = support 
        output = tf.convert_to_tensor(supports, dtype=tf.float32)
        output = tf.transpose(output, perm=[2,1,3,0])
        if batch_size is not None:
            output = K.reshape(output, (batch_size, num_nodes, self.num_of_supports*self.latent_units)) 
        else:
            output = K.reshape(output, (-1, num_nodes, self.num_of_supports*self.latent_units))   
        output = tf.keras.backend.dot(output, self.vars['kernel_dense'])
        output += self.vars['bias_dense']
#         output = tf.reduce_sum(output, -1)
                
        return self.activation(output)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The innermost dimension of input_shape must be defined, but saw: %s' % input_shape)
        return input_shape[:-1].concatenate(self.units)
    
    def get_config(self):
        config = super(NGraphConvolution, self).get_config()
        config.update({
            'latent_units': self.latent_units,
            'units': self.units,
            'supports': [supp.numpy() for supp in self.supports],
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        })
        return config

class Symmetric(tf.keras.constraints.Constraint):
    """Symmetrizes weights after gradient update."""

    def __init__(self, **kwargs):
        super(Symmetric, self).__init__(**kwargs)
        pass

    def __call__(self, w):
        w = tf.constant(0.5, dtype=w.dtype) * (w + tf.transpose(w))
        return w

    def get_config(self):
        config = super(Symmetric, self).get_config()
        return config


class PseudoLikelihood(Layer):
    """Pseudo-Likelihood layer.
    This is a Keras implementation of Pseudo-Likelihood Layer for joint estimation of precision matrix with forecasting 
    in a network. 
    
    References:
      - ["NETS: Network estimation for time series", 
         by Matteo Barigozzi  Christian Brownlees Journal of Applied Econometrics 2019.]
         (https://onlinelibrary.wiley.com/doi/full/10.1002/jae.2676)
    
    Attributes:
      knowledge_graph_weighted_mask: weighted hard/soft masks, 
        N-D array with shape (num nodes, num nodes).
      partial_correlation_preestimator: pre-estimator of partial correlation for adaptive weighting,
        N-D array with shape (num nodes, num nodes).
      c: inverse variance of correlated errors, 1-D array with shape (num nodes, ).
      regularization_weight: regularization weight for regularizer on partial correlation weights. 
    
    Call arguments:
      inputs: list of 2 input tensors.
      training: Python boolean indicating whether the layer should behave in training mode (update c) 
        or in inference mode (doing nothing).
        
    Input shape:
      List of 2 N-D tensors with shape: [(batch_size, num_nodes), (batch_size, num_nodes)].
    
    Output shape:
      N-D tensor with shape: (batch_size, num_nodes).
    """
    def __init__(self,
                 knowledge_graph_weighted_mask=None,
                 partial_correlation_preestimator=None,
                 c=None,
                 regularization_weight=None,
                 partial_correlation_constraint=Symmetric(),
                 **kwargs):
        super(PseudoLikelihood, self).__init__(**kwargs)
        np.fill_diagonal(knowledge_graph_weighted_mask, np.inf)
        self.M = tf.constant(knowledge_graph_weighted_mask, dtype=tf.float32) # M contains finite and infinite entries
        
        # Boolean mask corresponding to finite and infinite entries
        self.boolean_mask = tf.constant(
            tf.where(tf.math.is_finite(self.M),
                     tf.ones(self.M.shape),
                     tf.zeros(self.M.shape)
                    ),
            dtype=tf.float32
        )
        self.c = tf.Variable(initial_value=tf.reshape(tf.cast(c, dtype=tf.float32), shape=(1, c.shape[0])),
                             trainable=False)

        self.partial_correlation_preestimator = tf.constant(partial_correlation_preestimator, dtype=tf.float32)
        self.partial_correlation_constraint = partial_correlation_constraint
        self.regularization_weight = regularization_weight

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not dtype.is_floating:
            raise TypeError("Unable to build `PseudoLikelihood` layer with non-floating point dtype %s"%(dtype, )) 
        
        first_input_shape = tensor_shape.TensorShape(input_shape[0])
        first_input_shape = first_input_shape.with_rank_at_least(2)
        second_input_shape = tensor_shape.TensorShape(input_shape[1])
        second_input_shape = second_input_shape.with_rank_at_least(2)       
        if tensor_shape.dimension_value(first_input_shape[-1]) is None: 
            raise ValueError("The innermost dimension of input_shape for ypred must be defined, but saw: %s"%input_shape[0])
        if tensor_shape.dimension_value(second_input_shape[-1]) is None:
            raise ValueError("The innermost dimension of input_shape for ytrue must be defined, but saw: %s"%input_shape[1])
        
        self.partial_correlation = self.add_weight(name='partial_correlation',
                                   initializer=tf.keras.initializers.Zeros(),
                                   constraint=self.partial_correlation_constraint,
                                   shape=self.M.shape,
                                   trainable=True)
        super(PseudoLikelihood, self).build(input_shape)
    
    def _get_training_value(self, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        return training
            
    def compute_eps_theta(self, eps):
        theta = tf.multiply(tf.sqrt(self.c / tf.transpose(self.c)),
                            tf.multiply(self.partial_correlation, self.boolean_mask))
        eps_theta = tf.matmul(eps, theta, transpose_b=True)
        return eps_theta

    def probabalistic_c_update(self, eps):
        eps_theta = self.compute_eps_theta(eps)
        _, variance = tf.nn.moments(eps - eps_theta, axes=[0], keepdims=True)
        self.c.assign(1 / variance)
        return eps_theta
    
        
        
    def call(self, inputs, training=None):
        
        self.training = self._get_training_value(training)
        ypred = inputs[0]
        ytrue = inputs[1]
        
        eps = ytrue - ypred
#         eps_theta = tf_utils.smart_cond(self.training,
#                                         lambda: self.probabalistic_c_update(eps),
#                                         lambda: self.compute_eps_theta(eps))
        eps_theta = tf.cond(tf.equal(tf.cast(self.training, dtype=tf.bool), tf.constant(True)),
                            lambda: self.probabalistic_c_update(eps),
                            lambda: self.compute_eps_theta(eps))
        
        # Add Lasso/Adaptive Lasso regularizer
        masking_weight = tf.where(tf.math.is_finite(self.M),
                                  self.M,
                                  tf.zeros_like(self.M))
        weighted_partial_correlation = tf.divide(self.partial_correlation, self.partial_correlation_preestimator)
        self.add_loss(self.regularization_weight * tf.math.reduce_sum(tf.abs(tf.multiply(weighted_partial_correlation, masking_weight))))

        return ypred + eps_theta
   
    def compute_output_shape(self, input_shape):
        first_input_shape = tensor_shape.TensorShape(input_shape[0])
        first_input_shape = first_input_shape.with_rank_at_least(2)
        second_input_shape = tensor_shape.TensorShape(input_shape[1])
        second_input_shape = second_input_shape.with_rank_at_least(2)       
        if tensor_shape.dimension_value(first_input_shape[-1]) is None: 
            raise ValueError('The innermost dimension of input_shape for ypred must be defined, but saw: %s'%input_shape[0])
        if tensor_shape.dimension_value(second_input_shape[-1]) is None:
            raise ValueError('The innermost dimension of input_shape for ytrue must be defined, but saw: %s'%input_shape[1])

        return first_input_shape

    def get_config(self):
        config = super(PseudoLikelihood, self).get_config()
        config.update({
            'knowledge_graph_weighted_mask': self.M.numpy(),
            'c': self.c.numpy(),
            'partial_correlation_preestimator': self.partial_correlation_preestimator.numpy(),
            'regularization_weight': self.regularization_weight,
        })
        return config
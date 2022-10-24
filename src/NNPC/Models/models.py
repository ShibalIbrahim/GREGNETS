"""Includes Neural Network Forecasting Models under (Generalized) Pseudolikelihood.

Creates Keras Models for Multivariate Time-Series Regression under (generalized)
pseudolikelihood.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Dropout, Input, Layer, Lambda, RNN
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import clone_model
from tensorflow.keras.activations import relu, softmax, tanh
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras import regularizers
from stellargraph.layer import GCN_LSTM

from src.Layers import layers
from src.Layers import layers_stgcn
from src.Layers import layers_gcrnn

_SEED = 8
np.random.seed(_SEED)
tf.random.set_seed(_SEED)

# ------------------------------------------ VAR -----------------------------------------
def create_var_model(
        input_shape=None,
        use_bias=True,
        kernel_regularizer=None,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=_SEED),
        kernel_constraint=None,
        bias_regularizer=None,
        bias_initializer='zeros',
        bias_constraint=None,
    ):
    """Instantiates VAR model.
    
    Args:
      input_shape: input shape to setup model input, (num (time) samples, num nodes, num past time samples)
      kernel_regularizer: Regularizer function applied to the "kernel" weights matrix
      kernel_initializer: Initializer for the 'kernel' weights matrix.
    
    Returns:
      VAR Keras model with output shape: (num samples, num nodes).
    """
    x_input = Input(name='input', shape=input_shape)
    y_pred = layers.VAR(
        use_bias=use_bias, 
        kernel_initializer=kernel_initializer,
        bias_initializer='zeros',
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
    )(x_input)
              
    model = Model(inputs=[x_input], outputs=[y_pred])
        
    return model


# ------------------------------------------ SVAR -----------------------------------------
def create_svar_model(
        input_shape=None,
        use_bias=True,
        kernel_regularizer=None,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=_SEED),
        kernel_constraint=None,
        bias_regularizer=None,
        bias_initializer='zeros',
        bias_constraint=None,
    ):
    """Instantiates SVAR model.
    
    Args:
      input_shape: input shape to setup model input, (num (time) samples, num nodes, num past time samples)
      kernel_regularizer: Regularizer function applied to the "kernel" weights matrix
      kernel_initializer: Initializer for the 'kernel' weights matrix.
    
    Returns:
      VAR Keras model with output shape: (num samples, num nodes).
    """
    x_input = Input(name='input', shape=input_shape)
    y_pred = layers.SVAR(
        use_bias=use_bias, 
        kernel_initializer=kernel_initializer,
        bias_initializer='zeros',
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
    )(x_input)
              
    model = Model(inputs=[x_input], outputs=[y_pred])
        
    return model


# ------------------------------------------ GCN -----------------------------------------
def create_gcn_model(
        input_shape=None,
        units=(1,),
        activation=None,
        output_activation="linear",
        supports=None,
        use_bias=True,
        kernel_regularizer=None,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=_SEED),
        kernel_constraint=None,
        bias_regularizer=None,
        bias_initializer='zeros',
        bias_constraint=None,
    ):
    """Instantiates GCN model.
    
    Args:
      input_shape: input shape to setup model input, (num (time) samples, num nodes, num past time samples)
      hidden_units: units for each hidden layer, tuple.
      activation: Activation function to use for hidden layers.
        If you don't specify anything, no activation is applied (i.e. "linear" activation 'a(x) = x').
      supports: list of knowledge graphs' adjacency matrices, [(num nodes, num nodes)] 
      kernel_regularizer: Regularizer function applied to the "kernel" weights matrix
      kernel_initializer: Initializer for the 'kernel' weights matrix.
    
    Returns:
      GCN Keras model with output shape: (num samples, num nodes).
    """
    x_input = Input(name='input', shape=input_shape)
    y_pred = x_input
    for i, u in enumerate(units):
        y_pred = layers.GraphConvolution(
            units=u, 
            supports=supports, 
            activation=None,
            use_bias=use_bias, 
            kernel_initializer=kernel_initializer,
            bias_initializer='zeros',
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
        )(y_pred)
        if i<len(units)-1:
            y_pred = tf.keras.layers.Activation(activation)(y_pred)
        else:
            y_pred = tf.keras.layers.Activation(output_activation)(y_pred)
        
    y_pred = Lambda(lambda x: tf.squeeze(x, [-1]), 'Squeeze')(y_pred)
        
    model = Model(inputs=[x_input], outputs=[y_pred])
        
    return model


# ------------------------------------------ PGCN -----------------------------------------
def create_pgcn_model(
        input_shape=None,
        units=(1,),
        activation=None,
        output_activation="linear",
        supports=None,
        regularization_weight=None,
        use_bias=True,
        kernel_regularizer=None,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=_SEED),
        kernel_constraint=None,
        bias_regularizer=None,
        bias_initializer='zeros',
        bias_constraint=None,
    ):
    """Instantiates PGCN model.
    
    Args:
      input_shape: input shape to setup model input, (num (time) samples, num nodes, num past time samples)
      hidden_units: units for each hidden layer, tuple.
      activation: Activation function to use for hidden layers.
        If you don't specify anything, no activation is applied (i.e. "linear" activation 'a(x) = x').
      supports: list of knowledge graphs' adjacency matrices, [(num nodes, num nodes)] 
      kernel_regularizer: Regularizer function applied to the "kernel" weights matrix
      kernel_initializer: Initializer for the 'kernel' weights matrix.
    
    Returns:
      GCN Keras model with output shape: (num samples, num nodes).
    """
    x_input = Input(name='input', shape=input_shape)
    y_pred = x_input
    for i, u in enumerate(units):
        y_pred = layers.PerturbedGraphConvolution(
            units=u, 
            supports=supports,
            regularization_weight=regularization_weight,
            activation=None,
            use_bias=use_bias, 
            kernel_initializer=kernel_initializer,
            bias_initializer='zeros',
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
        )(y_pred)
        if i<len(units)-1:
            y_pred = tf.keras.layers.Activation(activation)(y_pred)
        else:
            y_pred = tf.keras.layers.Activation(output_activation)(y_pred)
        
    y_pred = Lambda(lambda x: tf.squeeze(x, [-1]), 'Squeeze')(y_pred)
        
    model = Model(inputs=[x_input], outputs=[y_pred])
        
    return model


# ------------------------------------------- N-GCN -----------------------------------------
def create_ngcn_model(
        input_shape=None,
        latent_units=(1,),
        units=(1,),
        activation=None,
        output_activation=None,
        supports=None,
        use_bias=True,
        kernel_regularizer=None,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=_SEED),
        kernel_constraint=None,
        bias_regularizer=None,
        bias_initializer='zeros',
        bias_constraint=None
    ):
    """Instantiates N-GCN model.
    
    Args:
      input_shape: input shape to setup model input, (num (time) samples, num nodes, num past time samples)
      hidden_units: units for each hidden layer, tuple.
      activation: Activation function to use for hidden layers.
        If you don't specify anything, no activation is applied (i.e. "linear" activation 'a(x) = x').
      supports: list of knowledge graphs' adjacency matrices, [(num nodes, num nodes)] 
      kernel_regularizer: Regularizer function applied to the "kernel" weights matrix
      kernel_initializer: Initializer for the 'kernel' weights matrix.
    
    Returns:
      N-GCN Keras model with output shape: (num samples, num nodes).
    """
    x_input = Input(name='input', shape=input_shape) 
    y_pred = x_input
    for i, (lu, u) in enumerate(zip(latent_units, units)):
        y_pred = layers.NGraphConvolution(
            latent_units=lu,
            units=u, 
            supports=supports, 
            activation=None, 
            use_bias=use_bias, 
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint
        )(y_pred)
        if i<len(units)-1:
            y_pred = tf.keras.layers.Activation(activation)(y_pred)
        else:
            y_pred = tf.keras.layers.Activation(output_activation)(y_pred)
    y_pred = Lambda(lambda x: tf.squeeze(x, [-1]), 'Squeeze')(y_pred)
    
    
    model = Model(inputs=[x_input], outputs = [y_pred])
        
    return model

# ------------------------------------------- ST-GCN -----------------------------------------
def create_stgcn_model(
        input_shape=None,
        Ks=None,
        graph_kernel=None,
        Kt=None,
        channel_blocks=None,
        dropout=None,
        temporal_activation='glu',
        spatial_activation='relu',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(1.0),
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=_SEED)
    ):
    """Instantiates ST-GCN model.
    
    Args:
      input_shape: input shape to setup model input, (batch_size, past_time_steps, num_nodes, num_channels/features)
      Ks: kernel size of spatial convolution, int scaler.
      graph_kernel: list of graph kernel matrices, [(num_nodes, Ks*num_nodes)]
      Kt: kernel size of temporal convolution, int scaler.
      channel_blocks: channel configs of SpatioTemporalConvolutionBlock, list.
      temporal_activation: Activation function for temporal convolution layers.
      spatial_activation: Activation function for spatial convolution layers.
      use_bias: whether to use bias, boolean.
      kernel_regularizer: Regularizer function applied to the "kernel" weights matrix
      kernel_initializer: Initializer for the 'kernel' weights matrix.
    
    Returns:
      ST-GCN Keras model with output shape: (num samples, num nodes).
    """
    x_input = Input(name='input', shape=input_shape)    
    
    # Ko>0: kernel size of temporal convolution in the output layer.
    Ko = input_shape[0]
    # ST-Block
    x = x_input
    for i, channels in enumerate(channel_blocks):
        x = layers_stgcn.SpatioTemporalConvolutionBlock(
            channels,
            Ks, 
            graph_kernel,
            Kt,
            temporal_activation=temporal_activation,
            spatial_activation=spatial_activation,
            use_bias=use_bias,
            dropout=dropout,
            name='STBlock{}'.format(i+1),
            )(x)
        
        Ko -= 2 * (Kt - 1)

    # Output Layer
    if Ko > 1:
        y_pred = layers_stgcn.OutputBlock(
            Ko,
            temporal_activations=[temporal_activation, temporal_activation], # [temporal_activation, 'sigmoid']
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name='OutputBlock',
        )(x)
    else:
        raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

    y_pred = Lambda(lambda x: tf.squeeze(x, [1, 3]), 'Squeeze')(y_pred)
    
    model = Model(inputs=[x_input], outputs=[y_pred])
    return model


# ------------------------------------------- ST-GCN-P -----------------------------------------
def create_stgcnp_model(
        input_shape=None,
        Ks=None,
        graph_kernel=None,
        Kt=None,
        channel_blocks=None,
        dropout=None,
        temporal_activation='glu',
        spatial_activation='relu',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(1.0),
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=_SEED),
        graph_regularizer=tf.keras.regularizers.l2(1.0),
        graph_initializer='zeros',
    ):
    """Instantiates ST-GCN model.
    
    Args:
      input_shape: input shape to setup model input, (batch_size, past_time_steps, num_nodes, num_channels/features)
      Ks: kernel size of spatial convolution, int scaler.
      graph_kernel: list of graph kernel matrices, [(num_nodes, Ks*num_nodes)]
      Kt: kernel size of temporal convolution, int scaler.
      channel_blocks: channel configs of SpatioTemporalConvolutionBlock, list.
      temporal_activation: Activation function for temporal convolution layers.
      spatial_activation: Activation function for spatial convolution layers.
      use_bias: whether to use bias, boolean.
      kernel_regularizer: Regularizer function applied to the "kernel" weights matrix
      kernel_initializer: Initializer for the 'kernel' weights matrix.
    
    Returns:
      ST-GCN Keras model with output shape: (num samples, num nodes).
    """
    x_input = Input(name='input', shape=input_shape)    
    
    # Ko>0: kernel size of temporal convolution in the output layer.
    Ko = input_shape[0]
    # ST-Block
    x = x_input
    for i, channels in enumerate(channel_blocks):
        x = layers_stgcn.SpatioTemporalConvolutionPerturbBlock(
            channels,
            Ks, 
            graph_kernel,
            Kt,
            temporal_activation=temporal_activation,
            spatial_activation=spatial_activation,
            use_bias=use_bias,
            graph_initializer=graph_initializer,
            graph_regularizer=graph_regularizer,
            dropout=dropout,
            name='STBlock{}'.format(i+1),
            )(x)
        
        Ko -= 2 * (Kt - 1)

    # Output Layer
    if Ko > 1:
        y_pred = layers_stgcn.OutputBlock(
            Ko,
            temporal_activations=[temporal_activation, temporal_activation], # [temporal_activation, 'sigmoid']
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name='OutputBlock',
        )(x)
    else:
        raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

    y_pred = Lambda(lambda x: tf.squeeze(x, [1, 3]), 'Squeeze')(y_pred)
    
    model = Model(inputs=[x_input], outputs=[y_pred])
    return model


# ------------------------------------------- T-GCN -----------------------------------------
def create_tgcn_model(
        n_steps=None,
        adj=None,
        gc_layer=None,
        gc_layer_size=None,
        gc_activation=None,
        lstm_layer=None,
        lstm_layer_size=None,
        lstm_activation=None,
        dropout=None
    ):
    """Instantiates T-GCN model.
    
    Args:
      n_steps: num of past time steps, int scaler.
      adj: graph kernel/adjacency matrices of shape (num_nodes, Ks*num_nodes).
      gc_layer: number of graph convolution layers, int scaler.
      gc_layer_size: graph convolution layer units, int scaler.
      gc_activation: Activation function for graph convolution layers.
      lstm_layer: number of LSTM layers, int scaler.
      lstm_layer_size: number of units for LSTM layers, int scaler.
      lstm_activation: Activation function for LSTM layers.
      dropout: dropout, float scaler.
    
    Returns:
      T-GCN Keras model with output shape: (num samples, num nodes).
    """
    gcn_lstm = GCN_LSTM(
        seq_len=n_steps,
        adj=adj,
        gc_layer_sizes=[gc_layer_size]*gc_layer,
        gc_activations=[gc_activation]*gc_layer,
        lstm_layer_sizes=[lstm_layer_size]*lstm_layer,
        lstm_activations=[lstm_activation]*lstm_layer,
        dropout=dropout)

    x_input, x_output = gcn_lstm.in_out_tensors()
    model = Model(inputs=x_input, outputs=x_output)
    model.layers[-1].activation = tf.keras.activations.linear
    return model

# ------------------------------------------- GCRNN -----------------------------------------
def create_gcrnn_model(
        units=8,
        inputs_dim=1,
        output_dim=1,
        adj=None,
        filter_type='norm_laplacian',
        activation='tanh',
        use_bias=True,
        dropout=None
    ):
    num_nodes=adj.shape[0]
    cell = layers_gcrnn.GCRNNCell(units*num_nodes, adj_mx=adj, use_bias=use_bias, activation=activation, filter_type=filter_type)
    cell_out = layers_gcrnn.GCRNNCell(units*num_nodes, adj_mx=adj, use_bias=use_bias, num_proj=1, activation=activation, filter_type=filter_type)
    inp = Input((None, inputs_dim*num_nodes))
    
    gcn_in = inp
    gcn_in = RNN(cell, return_sequences=True)(gcn_in)
    gcn_in = Dropout(dropout)(gcn_in)
    #gcn_in = RNN(cell,return_sequences=True)(gcn_in)
    layer_out = RNN(cell_out, return_sequences=False)
    ypred = layer_out(gcn_in)
#     ypred = Dense(num_nodes, activation='linear')(ypred)
    model = Model(inp, ypred)
    return model

# --------------------------------------- LSTM --------------------------------------------
def create_lstm_model(
        input_shape=None,
        output_units=None, 
        conv_layers=None,
        conv_filtersize=None,
        conv_filters=None,
        conv_activation='relu',
        conv_dropout=None,
        conv_kernel_regularizer=None,
        conv_kernel_initializer=tf.keras.initializers.GlorotUniform(seed=_SEED),
        lstm_layers=None,
        lstm_units=None,
        lstm_recurrent_activation="sigmoid",
        lstm_activation="tanh",
        lstm_recurrent_regularizer=None,
        lstm_kernel_regularizer=None,
        lstm_kernel_initializer=tf.keras.initializers.GlorotUniform(seed=_SEED),
        lstm_recurrent_initializer=tf.keras.initializers.Orthogonal(seed=_SEED),
        lstm_dropout=0.0,
        lstm_recurrent_dropout=0.0,
        dense_layers=None,
        dense_units=None,
        dense_kernel_regularizer=None,
        dense_kernel_initializer=tf.keras.initializers.GlorotUniform(seed=_SEED),
        dense_activation='relu',
        output_activation='linear',
        use_bias=True,
    ):
    
    x_input = Input(name='input', shape=input_shape)
    x = x_input
    
    ### Convolution Layers
    for j in range(0, conv_layers):
        x = Conv1D(conv_filters,
                   kernel_size=conv_filtersize,
                   use_bias=use_bias,
                   activation=conv_activation,
                   kernel_regularizer=conv_kernel_regularizer,
                   kernel_initializer=conv_kernel_initializer)(x) 
        x = Dropout(conv_dropout)(x)
        # x = MaxPooling1D(pool_size=2)(x)

    ### LSTM layers
    for j in range(1, lstm_layers):
        x = LSTM(lstm_units,
                 use_bias=use_bias,
                 activation=lstm_activation,
                 recurrent_activation=lstm_recurrent_activation,
                 kernel_regularizer=lstm_kernel_regularizer,
                 recurrent_regularizer=lstm_recurrent_regularizer,
                 kernel_initializer=lstm_kernel_initializer,
                 recurrent_initializer=lstm_recurrent_initializer,
#                  dropout=lstm_dropout,
#                  recurrent_dropout=lstm_recurrent_dropout, # so that cuDNN kernel can be used instead of generic GPU kernel
                 return_sequences=True)(x) 
    x = LSTM(lstm_units,
             use_bias=use_bias,
             activation=lstm_activation,
             recurrent_activation=lstm_recurrent_activation,
             kernel_regularizer=lstm_kernel_regularizer,
             recurrent_regularizer=lstm_recurrent_regularizer,
             kernel_initializer=lstm_kernel_initializer,
#              dropout=lstm_dropout,
#              recurrent_dropout=lstm_recurrent_dropout, # so that cuDNN kernel can be used instead of generic GPU kernel
             recurrent_initializer=lstm_recurrent_initializer,
             return_sequences=False)(x) # last lstm layer

    ### Dense layers
    for units in range(1, dense_layers):
        x = Dense(dense_units,
                  use_bias=use_bias,
                  activation=dense_activation,
                  kernel_regularizer=dense_kernel_regularizer)(x)
    y_pred = Dense(output_units,
              use_bias=use_bias,
              activation=output_activation,
              kernel_regularizer=dense_kernel_regularizer)(x) # output layer

    model = Model(inputs=[x_input], outputs=[y_pred])
    return model
 
    
# ------------------------------ PseudoLikelihood with model ------------------------------------
def create_model_with_pseudolikelihood(base_model,
                                       base_model_kwargs,
                                       knowledge_graph_weighted_mask=None,
                                       partial_correlation_preestimator=None,
                                       c=None,
                                       partial_correlation_regularization_weight=None,
                                       metrics=['mse']):
    """Instantiates any model with pseudolikelihood last layer.
    
    Args:
      base_model: A multivariate time-series Keras model object.
      base_model_kwargs: arguments to instantiate the Keras base model.
      knowledge_graph_weighted_mask: weighted hard/soft masks, 
        N-D array with shape (num nodes, num nodes).
      partial_correlation_preestimator: pre-estimator of partial correlation for adaptive weighting,
        N-D array with shape (num nodes, num nodes).
      c: inverse variance of correlated errors, 1-D array with shape (num nodes, ).
      regularization_weight: regularization weight for regularizer on partial correlation weights. 
    
    Returns:
      Keras model with output shape: (num (time) samples, num nodes).
    """
    model = base_model(**base_model_kwargs)
#     print(model.summary())
    x_input = model.input
    y_pred = model(x_input)
    y_true = tf.keras.layers.Input(name='y_true', shape=y_pred.shape[1:])
    
    y_pseudopred = layers.PseudoLikelihood(
        knowledge_graph_weighted_mask=knowledge_graph_weighted_mask,
        partial_correlation_preestimator=partial_correlation_preestimator,
        c=c,
        regularization_weight=partial_correlation_regularization_weight,
    )([y_pred, y_true])
            
    model = tf.keras.Model(inputs=[x_input, y_true], outputs = [y_pseudopred])
    model.compile(metrics=metrics)
    model.outputs[0]._uses_learning_phase = True
#     print(model.summary())

    return model


# ---------------------------- Generalized PseudoLikelihood with model --------------------------

"""Tests for all layers defined in layers.py and layers_stgcn.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from numpy.testing import assert_allclose
import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import has_arg
from tensorflow.keras import Model, Input
from tensorflow.python.keras import backend as K

import layers

def get_test_data(num_train=1000, num_test=500, input_shape=(10,),
                  output_shape=(2,),
                  classification=True, num_classes=2):
    """Generates test data to train a model on.
    classification=True overrides output_shape
    (i.e. output_shape is set to (1,)) and the output
    consists in integers in [0, num_classes-1].
    Otherwise: float output with shape output_shape.
    """
    samples = num_train + num_test
    if classification:
        y = np.random.randint(0, num_classes, size=(samples,))
        X = np.zeros((samples,) + input_shape, dtype=np.float32)
        for i in range(samples):
            X[i] = np.random.normal(loc=y[i], scale=0.7, size=input_shape)
    else:
        y_loc = np.random.random((samples,))
        X = np.zeros((samples,) + input_shape, dtype=np.float32)
        y = np.zeros((samples,) + output_shape, dtype=np.float32)
        for i in range(samples):
            X[i] = np.random.normal(loc=y_loc[i], scale=0.7, size=input_shape)
            y[i] = np.random.normal(loc=y_loc[i], scale=0.7, size=output_shape)

    return (X[:num_train], y[:num_train]), (X[num_train:], y[num_train:])

def layer_test(layer_cls, kwargs={}, input_shape=None, input_dtype=None,
              input_data=None, expected_output=None,
              expected_output_dtype=None, fixed_batch_size=False,
              custom_objects={}):
    """Test routine for a layer with a single input tensor
    and single output tensor.
    """
    # generate input data
    if input_data is None:
        assert input_shape
        if not input_dtype:
            input_dtype = K.floatx()
        input_data_shape = list(input_shape)
        for i, e in enumerate(input_data_shape):
            if e is None:
                input_data_shape[i] = np.random.randint(1, 4)
        input_data = (10 * np.random.random(input_data_shape))
        input_data = input_data.astype(input_dtype)
    else:
        if input_shape is None:
            input_shape = input_data.shape
        if input_dtype is None:
            input_dtype = input_data.dtype
    if expected_output_dtype is None:
        expected_output_dtype = input_dtype

    # instantiation
    layer = layer_cls(**kwargs)

    # test get_weights , set_weights at layer level
    weights = layer.get_weights()
    layer.set_weights(weights)
    
    expected_output_shape = layer.compute_output_shape(input_shape)
    
    # test in functional API
    if fixed_batch_size:
        x = Input(batch_shape=input_shape, dtype=input_dtype)
    else:
        x = Input(shape=input_shape[1:], dtype=input_dtype)
    y = layer(x)
    assert K.dtype(y) == expected_output_dtype, "Layer output dtype: {} does not match the expected output dtype: {}.".format(K.dtype(y), expected_output_dtype)

    # check with the functional API
    model = Model(x, y)

    actual_output = model.predict(input_data)
    actual_output_shape = actual_output.shape
    for expected_dim, actual_dim in zip(expected_output_shape,
                                        actual_output_shape):
        if expected_dim is not None:
            assert expected_dim == actual_dim, "expected_dim:{} doesn't match actual_dim:{}".format(expected_dim, actual_dim)

    if expected_output is not None:
        assert_allclose(actual_output, expected_output, rtol=1e-3)

    # test instantiation from layer config
    layer_config = layer.get_config()
    layer_config['batch_input_shape'] = input_shape
    layer = layer.__class__.from_config(layer_config)
        
    # test serialization, weight setting at model level
    model_config = model.get_config()
    recovered_model = model.__class__.from_config(model_config, custom_objects=custom_objects)
    
    if model.weights:
        weights = model.get_weights()
        recovered_model.set_weights(weights)
        _output = recovered_model.predict(input_data)
        assert_allclose(_output, actual_output, rtol=1e-3)

    # test training mode (e.g. useful when the layer has a
    # different behavior at training and testing time).
    if has_arg(layer.call, 'training'):
        model.compile('rmsprop', 'mse')
        model.train_on_batch(input_data, actual_output)

    # for further checks in the caller function
    return actual_output

class CustomLayersTest(tf.test.TestCase):

    def testVARLayer(self):
        input_dim = 3
        num_nodes = 4
        num_samples = 10
        x = np.ones((num_samples, num_nodes, input_dim), dtype=np.float32)
        output = layer_test(
            layers.VAR, 
            kwargs={'name': 'var'},
            input_shape=(num_samples, num_nodes, input_dim),
            input_dtype=tf.float32,
            input_data=x,
            expected_output=None,
            expected_output_dtype=tf.float32,
            fixed_batch_size=False,
            custom_objects={'VAR': layers.VAR}
        )        
        
    def testGraphConvolutionLayer(self):
        input_dim = 3
        num_nodes = 4
        num_samples = 10
        units = 2        
        knowledge_graph = np.ones((num_nodes, num_nodes), dtype=np.float32)
        x = np.ones((num_samples, num_nodes, input_dim), dtype=np.float32)
        output = layer_test(
            layers.GraphConvolution, 
            kwargs={'name': 'gc', 'units': units,
                    'supports': [knowledge_graph]},
            input_shape=(num_samples, num_nodes, input_dim),
            input_dtype=tf.float32,
            input_data=x,
            expected_output=None,
            expected_output_dtype=tf.float32,
            fixed_batch_size=False,
            custom_objects={'GraphConvolution': layers.GraphConvolution}
        )        
        
    def testNGraphConvolutionLayer(self):
        input_dim = 3
        num_nodes = 4
        num_samples = 10
        num_layers = 1 
        latent_units = 8
        units = 2        
        supports = [np.identity(num_nodes, dtype=np.float32), np.ones((num_nodes, num_nodes), dtype=np.float32)]
        x = np.ones((num_samples, num_nodes, input_dim), dtype=np.float32)
        output = layer_test(
            layers.NGraphConvolution, 
            kwargs={'name': 'NGraphConvolution',
                    'latent_units': latent_units,
                    'units': units,
                    'supports': supports},
            input_shape=(num_samples, num_nodes, input_dim),
            input_dtype=tf.float32,
            input_data=x,
            expected_output=None,
            expected_output_dtype=tf.float32,
            fixed_batch_size=False,
            custom_objects={'NGraphConvolution': layers.NGraphConvolution}
        )    
                    
    def testPseudolikelihoodLayer(self):
        pass
        
if __name__ == '__main__':
    tf.test.main()
import numpy as np

import tensorflow as tf
from tensorflow.python.layers import convolutional as conv_layers
DEFAULT_PADDING = 'SAME'


def define_scope(f):

    property_name = f.__name__
    attribute = '_' + property_name

    def decorator(self):
        if not hasattr(self, attribute) or getattr(self, attribute) is None:
            with tf.variable_scope(self.name):
                setattr(self, attribute, f(self))
        return getattr(self, attribute)

    return decorator


class Layer(object):
    """
    General Layer class
    """
    def __init__(self, name):
        self.name = name
        self.inputs = None
        self._outputs = None
        self.activation = tf.nn.relu

    def outputs(self):
        pass

    def __repr__(self):
        return '{self.__class__.__name__}: {self.name}'.format(self=self)


class Conv2dLayer2(Layer):
    """
    Layer implementing a 2D convolution-based transformation of its inputs.
    """

    def __init__(self, num_input_channels, num_output_channels, kernel_dim_1, kernel_dim_2, name, data_format='NHWC', data_type=tf.float32):
        super(Conv2dLayer2, self).__init__(name)
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.kernel_dim_1 = kernel_dim_1
        self.kernel_dim_2 = kernel_dim_2
        self.data_format = data_format
        self.data_type = data_type

    @define_scope
    def outputs(self):

        conv = conv_layers.conv2d(
            self.inputs,
            self.num_output_channels,
            [self.kernel_dim_1, self.kernel_dim_2],
            padding='SAME',
            use_bias=False,
            data_format='channels_last'
        )

        biases = tf.get_variable('biases', [self.num_output_channels], self.data_type, tf.constant_initializer(0.0))
        conv_bias = tf.reshape(tf.nn.bias_add(conv, biases, data_format=self.data_format), conv.get_shape())

        return self.activation(conv_bias)


class AffineLayer(Layer):
    """
    Fully Connected Layer
    """
    def __init__(self, name, num_channels_in, num_channels_out, flatten_inputs=False, final_layer=False):
        super(AffineLayer, self).__init__(name)
        self.final_layer = final_layer
        self.flatten_inputs = flatten_inputs
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out
        self.weights = None

    @define_scope
    def outputs(self):
        init_factor = 2. if self.activation == tf.nn.relu else 1.

        self.weights = tf.get_variable(
            'weights',
            [self.num_channels_in, self.num_channels_out],
            tf.float32,
            tf.random_normal_initializer(stddev=np.sqrt(init_factor / self.num_channels_in))
        )
        biases = tf.get_variable('biases', [self.num_channels_out],
                                 tf.float32,
                                 tf.constant_initializer(0.0))
        # self.weights = _weights([input_dim, output_dim])
        # biases = tf.Variable(tf.zeros([output_dim]), name='biases')

        if self.final_layer:
            return tf.add(tf.matmul(self.inputs, self.weights), biases)
        else:
            return self.activation(tf.matmul(self.inputs, self.weights) + biases)


class ReshapeLayer(Layer):

    def __init__(self, output_shape=None, name='reshape'):
        super(ReshapeLayer, self).__init__(name)
        self.output_shape = (-1,) if output_shape is None else output_shape

    @define_scope
    def outputs(self):
        return tf.reshape(self.inputs, self.output_shape)


class PoolLayer(Layer):
    """
    Layer of non-overlapping 1D pools of inputs.
    SRC: https://www.tensorflow.org/api_docs/python/tf/nn/max_pool
    Stride, ksize = 2
    """
    def __init__(self, name):
        super(PoolLayer, self).__init__(name)

    @define_scope
    def outputs(self):
        return tf.nn.max_pool(
            self.inputs,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='VALID',
            name=self.name)


class NormLayer(Layer):
    """
    Local Response Norm. Layer
    https://www.tensorflow.org/versions/master/api_docs/python/nn/normalization

    """
    def __init__(self, name):
        super(NormLayer, self).__init__(name)

    def outputs(self):
        with tf.name_scope(self.name):
            self._outputs = tf.nn.lrn(self.inputs, 4, bias=1.0, alpha=0.0001, beta=0.75, name=self.name)


class DropoutLayer(Layer):
    """
    DropOut Layer
    """

    def __init__(self, keep_prob, name):
        super(DropoutLayer, self).__init__(name)
        self.keep_prob = keep_prob

    def outputs(self):
        with tf.name_scope(self.name):
            self._outputs = tf.nn.dropout(self.inputs, keep_prob=self.keep_prob, name=self.name)


# LAYER HELPER METHODS

def _bn(inputs, output_dim):
    """
    Batch normalization on convolutional layers.
    """
    beta = tf.Variable(tf.constant(0.0, shape=[output_dim]), name='beta')
    gamma = tf.Variable(tf.constant(1.0, shape=[output_dim]), name='gamma')
    batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2], name='moments')
    epsilon = 1e-3
    return tf.nn.batch_normalization(
        inputs, batch_mean, batch_var, beta, gamma, epsilon, 'bn'
    )


def _weights(shape, stddev=0.1, name='weights'):
    weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev, name=name))
    return weights


def _biases(shape, name='biases'):
    return tf.Variable(tf.constant(0.0, shape=shape, name=name))

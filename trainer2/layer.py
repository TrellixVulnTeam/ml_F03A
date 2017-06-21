import numpy as np
import tensorflow as tf
from tensorflow.python.layers import convolutional as conv_layers
DEFAULT_PADDING = 'SAME'


class Layer(object):
    """
    General Layer class
    """
    def __init__(self, name):
        self.name = name
        self.inputs = None
        self.outputs = None
        self.activation = tf.nn.relu

    def set_inputs(self, inputs):
        """
        Sets input for Layer
        :param inputs: tf.Tensor
        """
        self.inputs = inputs
        self.set_outputs()

    def set_outputs(self):
        pass

    def flatten(self):
        """
        Flatten inputs of layer
        """
        layer_shape = self.inputs.get_shape()
        num_features = layer_shape[1:4].num_elements()
        self.inputs = tf.reshape(self.inputs, [-1, num_features])

    def set_activation(self, activation):
        """
        Set layer activation function
        :param activation:
        :return:
        """
        self.activation = activation


class Conv2dLayer(Layer):
    """
    Convolutional Layer
    """
    def __init__(self, weights, biases, name, apply_batch_norm=False):
        super(Conv2dLayer, self).__init__(name)
        self.weights = weights
        self.biases = biases
        self.apply_batch_norm = apply_batch_norm

    def set_outputs(self):
        with tf.name_scope(self.name):

            self.biases = _biases(self.biases)
            self.weights = _weights(self.weights)

            conv = tf.nn.conv2d(self.inputs, self.weights, [1, 1, 1, 1], padding=DEFAULT_PADDING)
            conv_bias = tf.nn.bias_add(conv, self.biases, name='{}_bias'.format(self.name))

            if self.apply_batch_norm:
                conv_bias = _bn(conv_bias, conv_bias.get_shape()[-1].value)
            self.outputs = self.activation(conv_bias)


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

    def set_outputs(self):

        with tf.variable_scope(self.name):
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
            self.outputs = self.activation(conv_bias)
            print('CONV: {} => {}'.format(self.inputs.get_shape(), self.outputs.get_shape()))


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

    def set_inputs(self, inputs):
        self.inputs = inputs
        if self.flatten_inputs:
            self.flatten()
        self.set_outputs()

    def set_outputs(self):
        with tf.variable_scope(self.name):

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
                self.outputs = tf.add(tf.matmul(self.inputs, self.weights), biases)
            else:
                self.outputs = self.activation(tf.matmul(self.inputs, self.weights) + biases)
            print('CONV: {} => {}'.format(self.inputs.get_shape(), self.outputs.get_shape()))


class ReshapeLayer(Layer):

    def __init__(self, output_shape=None, name='reshape'):
        super(ReshapeLayer, self).__init__(name)
        self.output_shape = (-1,) if output_shape is None else output_shape

    def set_outputs(self):
        with tf.variable_scope(self.name):
            self.outputs = tf.reshape(self.inputs, self.output_shape)
            print('CONV: {} => {}'.format(self.inputs.get_shape(), self.outputs.get_shape()))


class PoolLayer(Layer):
    """
    Layer of non-overlapping 1D pools of inputs.
    SRC: https://www.tensorflow.org/api_docs/python/tf/nn/max_pool
    Stride, ksize = 2
    """
    def __init__(self, name):
        super(PoolLayer, self).__init__(name)

    def set_outputs(self):
        with tf.name_scope(self.name):
            self.outputs = tf.nn.max_pool(
                self.inputs,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='VALID',
                name=self.name)
            print('CONV: {} => {}'.format(self.inputs.get_shape(), self.outputs.get_shape()))


class NormLayer(Layer):
    """
    Local Response Norm. Layer
    https://www.tensorflow.org/versions/master/api_docs/python/nn/normalization

    """
    def __init__(self, name):
        super(NormLayer, self).__init__(name)

    def set_outputs(self):
        with tf.name_scope(self.name):
            self.outputs = tf.nn.lrn(self.inputs, 4, bias=1.0, alpha=0.0001, beta=0.75, name=self.name)


class DropoutLayer(Layer):
    """
    DropOut Layer
    """

    def __init__(self, keep_prob, name):
        super(DropoutLayer, self).__init__(name)
        self.keep_prob = keep_prob

    def set_outputs(self):
        with tf.name_scope(self.name):
            self.outputs = tf.nn.dropout(self.inputs, keep_prob=self.keep_prob, name=self.name)


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

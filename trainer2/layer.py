import tensorflow as tf
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


class AffineLayer(Layer):
    """
    Fully Connected Layer
    """
    def __init__(self, name, flatten_inputs=False, final_layer=False):
        super(AffineLayer, self).__init__(name)
        self.final_layer = final_layer
        self.flatten_inputs = flatten_inputs
        self.weights = None

    def set_inputs(self, inputs):
        self.inputs = inputs
        if self.flatten_inputs:
            self.flatten()
        self.set_outputs()

    def set_outputs(self):
        with tf.name_scope(self.name):

            input_dim = self.inputs.get_shape()[1].value
            output_dim = 10 if self.final_layer else int(input_dim/2)

            self.weights = _weights([input_dim, output_dim])
            biases = tf.Variable(tf.zeros([output_dim]), name='biases')

            if self.final_layer:
                self.outputs = tf.add(tf.matmul(self.inputs, self.weights), biases)
            else:
                self.outputs = self.activation(tf.matmul(self.inputs, self.weights) + biases)


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
                ksize=[1, 3, 3, 1],
                strides=[1, 2, 2, 1],
                padding=DEFAULT_PADDING,
                name=self.name)


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

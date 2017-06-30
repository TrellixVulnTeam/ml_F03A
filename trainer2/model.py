# Working example for my blog post at:
# https://danijar.github.io/structuring-your-tensorflow-models
import tensorflow as tf
from trainer2.layer import PoolLayer, AffineLayer, Conv2dLayer2, ReshapeLayer


class Model:

    def __init__(self, name, layers, activation=tf.nn.relu, train_epochs=10, initial_lr=0.005, l2_loss=None):
        self.name = name
        self.layers = layers
        self.train_epochs = train_epochs
        self.learning_rate = initial_lr
        self.batch_size = 32
        self.activation = activation
        self.l2_loss = l2_loss
        self.__check_init()

    def __check_init(self):
        assert self.learning_rate > 0.0
        assert len(self.layers) > 0
        assert self.name is not None

    def get_layers(self, inputs):

        next_inputs = inputs

        for layer in self.layers:
            layer.activation = self.activation
            layer.inputs = next_inputs
            next_inputs = layer.outputs()
            assert next_inputs is not None

        return next_inputs

    def get_l2_losses(self):
        weights = []
        for layer in self.layers:
            if isinstance(layer, (Conv2dLayer2, AffineLayer)):
                weights.append(tf.nn.l2_loss(layer.weights))
        return self.l2_loss * sum(weights)

    @classmethod
    def trial(cls):
        layers = [
            Conv2dLayer2(3, 32, 5, 5, 'conv_1'),
            PoolLayer('pool_1'),
            Conv2dLayer2(32, 64, 5, 5, 'conv_2'),
            PoolLayer('pool_2'),
            Conv2dLayer2(64, 64, 5, 5, 'conv_3'),
            PoolLayer('pool_3'),
            ReshapeLayer(output_shape=[-1, 64 * 8 * 8]),
            AffineLayer('fc_1', 4096, 512),
            AffineLayer('output', 512, 1001, final_layer=True)
        ]
        return cls(name='trial', layers=layers)

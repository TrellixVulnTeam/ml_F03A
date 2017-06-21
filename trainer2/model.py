# Working example for my blog post at:
# https://danijar.github.io/structuring-your-tensorflow-models
import tensorflow as tf
from trainer2.layer import Conv2dLayer, PoolLayer, AffineLayer, Conv2dLayer2, ReshapeLayer


class Model:

    def __init__(self, name, layers, activation=tf.nn.relu, train_epochs=10, initial_lr=0.001, l2_loss=None):
        self.name = name or 'unnamed model'
        self.layers = layers
        self.train_epochs = train_epochs
        self.initial_lr = initial_lr
        self.batch_size = 32
        self.activation = activation
        self.l2_loss = l2_loss
        self.__check_init()

    def __check_init(self):
        assert self.initial_lr > 0.0
        assert len(self.layers) > 0

    def get_layers(self, inputs):
        for i, layer in enumerate(self.layers):
            layer.set_activation(self.activation)
            if i == 0:
                layer.set_inputs(inputs)
            else:
                layer.set_inputs(self.layers[i - 1].outputs)

        return self.layers[-1].outputs

    def get_l2_losses(self):
        weights = []
        for layer in self.layers:
            if isinstance(layer, (Conv2dLayer, AffineLayer)):
                weights.append(tf.nn.l2_loss(layer.weights))
        return self.l2_loss * sum(weights)


    @classmethod
    def trial(cls):
        layers = [
            Conv2dLayer2(3, 32, 5, 5, 'conv_1'),
            PoolLayer('pool_1'),
            Conv2dLayer2(32, 64, 5, 5, 'conv_2'),
            PoolLayer('pool_2'),
            ReshapeLayer(output_shape=[-1, 64 * 7 * 7]),
            AffineLayer('fc_1', 3136, 512),
            AffineLayer('output', 512, 1001, final_layer=True)
        ]
        return cls(name='trial', layers=layers)

# cnn.conv(32, 5, 5)
#         cnn.mpool(2, 2)
#         cnn.conv(64, 5, 5)
#         cnn.mpool(2, 2)
#         cnn.reshape([-1, 64 * 7 * 7])
#         cnn.affine(512)

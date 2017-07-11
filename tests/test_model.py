#
# test_model
# ml
#

"""

"""
from unittest import TestCase

import tensorflow as tf

from trainer2.model import Model


class SuperTestModel(TestCase):

    def setUp(self):
        self.model = Model.trial()
        self.images = tf.truncated_normal([2, 2, 2, 1], dtype=tf.float32)

    def tearDown(self):
        del self.images
        del self.model


class TestModel(SuperTestModel):
    def test_get_layers(self):
        pass

    def test_get_l2_losses(self):
        pass

    def test_build_graph(self):
        pass

    def test_cpu_gpu_copy(self):
        pass

    def test_get_optimizer(self):
        self.assertRaises(AssertionError, self.model.get_optimizer, 'incorrect_optimizer_string')
        self.assertIsInstance(self.model.get_optimizer('sgd'), tf.train.GradientDescentOptimizer)

    def test_reformat_images(self):
        self.model.data_format = 'NCHW'
        self.model.data_type = tf.float16

        new_images = self.model.reformat_images(self.images)
        self.assertEqual(new_images.get_shape(), [2, 1, 2, 2])
        self.assertEqual(new_images.dtype, tf.float16)

    def test_trial(self):
        pass

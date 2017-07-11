#
# test_dataset
# ml
#

from unittest import TestCase

import tensorflow as tf

from trainer2 import datasets


class SuperDatasetConfig(TestCase):
    def setUp(self):
        self.imageDataset = datasets.DatasetFactory.create_dataset(data_dir='../data/data')
        self.syntheticDataset = datasets.DatasetFactory.create_dataset(None)

    def tearDown(self):
        self.imageDataset = None
        self.syntheticDataset = None


class TestDataset(SuperDatasetConfig):

    def test_dataset(self):
        with self.assertRaises(AssertionError):
            datasets.DatasetFactory.create_dataset('this/very/unlikely/path')
            datasets.DatasetFactory.create_dataset(45)
            datasets.DatasetFactory.create_dataset(None)

        self.assertFalse(self.imageDataset.synthetic)
        self.assertTrue(self.syntheticDataset.synthetic)

    def test_tf_record_pattern(self):
        pass

    def test_preprocess(self):
        with self.assertRaises(AssertionError):
            self.imageDataset.preprocess(32, 0, train=True)
            self.syntheticDataset.preprocess(32, 0, train=True)

    def test_reader(self):
        self.assertIsInstance(self.imageDataset.reader(), tf.TFRecordReader)

    def test_num_classes(self):
        self.assertEqual(self.syntheticDataset.num_classes(), 1000)
        # self.assertEqual(self.imageDataset.num_classes(), 18)

    def test_num_examples_per_epoch(self):
        self.assertRaises(ValueError, self.imageDataset.num_examples_per_epoch, 'train__')
        self.assertRaises(AssertionError, self.syntheticDataset.num_examples_per_epoch, 'train')

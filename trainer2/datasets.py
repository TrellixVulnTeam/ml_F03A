# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Benchmark dataset utilities.
"""

from abc import abstractmethod
import os

import tensorflow as tf

from trainer2.input import ImagePreprocessor as ImProc
from trainer2 import flags
FLAGS = flags.get_flags()


class DatasetFactory(object):
    @staticmethod
    def create_dataset(data_dir):
        if data_dir:
            return ImgData(data_dir=data_dir)
        else:
            return SyntheticData()


class Dataset(object):
    """Abstract class for cnn benchmarks dataset."""

    def __init__(self, name, data_dir=None, image_size=32):
        self.name = name
        if data_dir is None and isinstance(self, (ImgData, ImagenetData)):
            raise ValueError('Data directory not specified')
        self.data_dir = data_dir
        self.image_size = image_size
        self.input_data_type = tf.float32
        self.resize_method = FLAGS.resize_method

    def tf_record_pattern(self, subset):
        return os.path.join(self.data_dir, '%s-*-of-*' % subset)

    def preprocess(self, batch_size, num_comp_devices, train=True):
        pass

    @staticmethod
    def reader(self):
        return tf.TFRecordReader()

    @abstractmethod
    def num_classes(self):
        pass

    @abstractmethod
    def num_examples_per_epoch(self, subset):
        pass

    def __repr__(self):
        return '{self.__class__.__name__}: {self.name}'.format(self=self)


class ImgData(Dataset):
    def __init__(self, data_dir=None):
        super(ImgData, self).__init__('custom-data', data_dir=data_dir, image_size=64)

    def num_classes(self):
        return 11

    def num_examples_per_epoch(self, subset='train'):
        if subset == 'train':
            return 13812
        elif subset == 'validation':
            return 1540
        else:
            raise ValueError('Invalid data subset "%s"' % subset)

    def preprocess(self, batch_size, num_comp_devices, train=True):
        pre_proc_result = ImProc(
            height=self.image_size,
            width=self.image_size,
            batch_size=batch_size,
            device_count=num_comp_devices,
            dtype=self.input_data_type,
            train=train,
            resize_method=self.resize_method)

        subset = 'train' if train else 'validation'
        images, labels = pre_proc_result.minibatch(self, subset=subset)

        return self.num_classes, images, labels


class SyntheticData(Dataset):
    def __init__(self, data_dir=None):
        super(SyntheticData, self).__init__('synthetic-data', data_dir)
        self.input_channels = 3

    def num_classes(self):
        return 1000

    def num_examples_per_epoch(self, subset):
        return 1000

    def preprocess(self, batch_size, num_comp_devices, train=True):
        """Add image Preprocessing ops to tf graph."""
        nclass = 1001
        input_shape = [batch_size, self.image_size, self.image_size, self.input_channels]
        images = tf.truncated_normal(
            input_shape,
            dtype=self.input_data_type,
            stddev=1e-1,
            name='synthetic_images')
        labels = tf.random_uniform(
            [batch_size],
            minval=1,
            maxval=nclass,
            dtype=tf.int32,
            name='synthetic_labels')

        images = tf.contrib.framework.local_variable(images, name='images')
        labels = tf.contrib.framework.local_variable(labels, name='labels')

        labels -= 1
        if num_comp_devices == 1:
            images_splits = [images]
            labels_splits = [labels]
        else:
            images_splits = tf.split(images, num_comp_devices, 0)
            labels_splits = tf.split(labels, num_comp_devices, 0)

        return nclass, images_splits, labels_splits


class ImagenetData(Dataset):
    def __init__(self, data_dir=None):
        super(ImagenetData, self).__init__('ImageNet', data_dir)

    def num_classes(self):
        return 1000

    def num_examples_per_epoch(self, subset='train'):
        if subset == 'train':
            return 1281167
        elif subset == 'validation':
            return 50000
        else:
            raise ValueError('Invalid data subset "%s"' % subset)

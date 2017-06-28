#
# build_data
# ml
#

"""
Script for building image data
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading
import tarfile

import numpy as np
import tensorflow as tf

from data.image_coder import ImageCoder
from data import image_util

tf.app.flags.DEFINE_string('data_dir', 'raw_data', 'Raw data directory')
tf.app.flags.DEFINE_string('output_dir', 'data', 'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 100, 'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 20, 'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 2, 'Number of threads to preprocess the images.')

tf.app.flags.DEFINE_string('labels_file', 'img_data/synset.txt', 'Labels file')

FLAGS = tf.app.flags.FLAGS


class UnpackInfo(object):
    def __init__(self, name, num_shards):
        self.name = name
        self.num_shard = num_shards
        self.labels = np.empty(0, dtype=int)
        self.filenames = np.empty(0)
        self.classnames = np.empty(0)

    def add_data(self, filenames, label, classname):
        self.filenames = np.append(self.filenames, filenames)
        self.labels = np.append(self.labels, [label]*filenames.size)
        self.classnames = np.append(self.classnames, [classname]*filenames.size)

    def shuffle_data(self):
        shuffle_array = np.arange(self.labels.shape[0])
        np.random.shuffle(shuffle_array)

        self.filenames = self.filenames[shuffle_array]
        self.labels = self.labels[shuffle_array]
        self.classnames = self.classnames[shuffle_array]

    def __repr__(self):
        return 'Info: {self.name}'.format(self=self)


def unpack_data(data_dir, output_dir):
    """Unpack and split raw data into training and validation sets."""
    tar_format = '%s/*.tar' % data_dir
    data_files = tf.gfile.Glob(tar_format)
    validation_data = UnpackInfo(name='validation', num_shards=FLAGS.validation_shards)
    train_data = UnpackInfo(name='train', num_shards=FLAGS.train_shards)

    label = 0
    for file in data_files:
        filename = file.split('.')[0].split('/')[-1]
        label += 1

        with tarfile.open(file) as tar:
            names = np.array(tar.getnames())
            # tar.extractall(path='output_dir/{}/'.format(output_dir, filename))

        train_names, valid_names = np.split(names, [int(0.9 * names.size)])

        train_data.add_data(train_names, label, filename)
        validation_data.add_data(valid_names, label, filename)

    train_data.shuffle_data()
    validation_data.shuffle_data()
    return train_data, validation_data


def main(_):
    assert not FLAGS.train_shards % FLAGS.num_threads, \
        'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards'
    assert not FLAGS.validation_shards % FLAGS.num_threads,\
        'Please make the FLAGS.num_threads commensurate with FLAGS.validation_shards'

    print('Saving results to => %s' % FLAGS.output_dir)
    train, valid = unpack_data(FLAGS.data_dir, FLAGS.output_dir)

    image_util.process_image_files(train, FLAGS.num_threads)


if __name__ == '__main__':
    tf.app.run()

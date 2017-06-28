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

from data import image_util
from data import process_bounding_boxes

tf.app.flags.DEFINE_string('data_dir', 'raw_data', 'Raw data directory')
tf.app.flags.DEFINE_string('output_dir', 'data', 'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 100, 'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 20, 'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 2, 'Number of threads to preprocess the images.')

tf.app.flags.DEFINE_string('labels_file', 'synset.txt', 'Labels file')
tf.app.flags.DEFINE_string('labels_dir', 'Annotation', 'Labels root directory')
tf.app.flags.DEFINE_string('imagenet_metadata_file', 'imagenet_metadata.txt', 'ImageNet metadata file')
tf.app.flags.DEFINE_string('bounding_box_file', 'bounding_boxes.csv', 'Bounding box file')


FLAGS = tf.app.flags.FLAGS


class UnpackInfo(object):
    def __init__(self, name, num_shards, output_dir=FLAGS.output_dir):
        self.name = name
        self.num_shards = num_shards
        self.output_dir = '{}'.format(output_dir)
        self.labels = np.empty(0, dtype=int)
        self.filenames = np.empty(0)
        self.classnames = np.empty(0)
        self.human_classnames = np.empty(0)

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

    def set_human_classnames(self, synset_lookup):
        humans = []
        for s in self.classnames:
            assert s in synset_lookup, ('Failed to find: %s' % s)
            humans.append(synset_lookup[s])
        self.human_classnames = np.array(humans, dtype=str)

    def __repr__(self):
        return 'Info: {self.name}'.format(self=self)


def _write_synsets_file(synsets):
    with open(FLAGS.labels_file, 'w') as f:
        f.write('\n'.join(synsets))


def _unpack_data(data_dir, output_dir):
    """Unpack and split raw data into training and validation sets."""
    tar_format = '%s/*.tar' % data_dir
    data_files = tf.gfile.Glob(tar_format)
    validation_data = UnpackInfo(name='validation', num_shards=FLAGS.validation_shards)
    train_data = UnpackInfo(name='train', num_shards=FLAGS.train_shards)
    synset_labels = []
    label = 0

    for file in data_files:
        filename = file.split('.')[0].split('/')[-1]
        synset_labels.append(filename)
        label += 1

        with tarfile.open(file) as tar:
            path = '{}/{}/'.format(output_dir, filename)
            names = np.array(tar.getnames(), dtype=str)
            file_prefix = np.full_like(names, fill_value=path)
            names = np.core.defchararray.add(file_prefix, names)
            # tar.extractall(path=path)

        train_names, valid_names = np.split(names, [int(0.9 * names.size)])

        train_data.add_data(train_names, label, filename)
        validation_data.add_data(valid_names, label, filename)

    train_data.shuffle_data()
    validation_data.shuffle_data()

    _write_synsets_file(synset_labels)

    return train_data, validation_data, synset_labels


def main(_):
    assert not FLAGS.train_shards % FLAGS.num_threads, \
        'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards'
    assert not FLAGS.validation_shards % FLAGS.num_threads,\
        'Please make the FLAGS.num_threads commensurate with FLAGS.validation_shards'

    # Unpack raw data
    train, valid, u_labels = _unpack_data(FLAGS.data_dir, FLAGS.output_dir)

    # Process bounding boxes
    process_bounding_boxes.run([FLAGS.labels_dir, FLAGS.labels_file])

    # Build lookups
    synset_lookup, bbox_lookup = image_util.build_lookups(FLAGS.imagenet_metadata_file, FLAGS.bounding_box_file, u_labels)
    train.set_human_classnames(synset_lookup)
    print(train)
    # image_util.process_image_files(train, FLAGS.num_threads)
    # image_util.process_image_files(valid, FLAGS.num_threads)


if __name__ == '__main__':
    tf.app.run()

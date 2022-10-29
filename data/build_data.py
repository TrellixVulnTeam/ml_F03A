#
# build_data
# ml.data
#

"""
Script for building image data
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import shutil
import sys
import tarfile

import numpy as np
import tensorflow as tf

from data import image_util
from data import process_bounding_boxes

if sys.version_info >= (3,):
    from urllib import request as urllib2

tf.app.flags.DEFINE_string('data_dir', 'raw_data', 'Raw data directory')
tf.app.flags.DEFINE_string('output_dir', 'data', 'Output data directory')
tf.app.flags.DEFINE_string('download_list', 'txt_util/list.csv', 'Image-net synsets to download and use')

tf.app.flags.DEFINE_integer('train_shards', 90, 'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 10, 'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 2, 'Number of threads to preprocess the images.')

tf.app.flags.DEFINE_string('labels_file', 'txt_util/synset.txt', 'Labels file')
tf.app.flags.DEFINE_string('labels_dir', 'data/Annotation', 'Labels root directory')
tf.app.flags.DEFINE_string('imagenet_metadata_file', 'txt_util/imagenet_metadata.txt', 'ImageNet metadata file')
tf.app.flags.DEFINE_string('bounding_box_file', 'txt_util/bounding_boxes.csv', 'Bounding box file')

FLAGS = tf.app.flags.FLAGS

USER_ID = 'eskil'
ACCESS_PASS = os.environ.get('IMAGENET_PASS')
SYNSET_URL = 'http://www.image-net.org/download/synset?wnid={}&username={}&accesskey={}&release=latest&src=stanford'
BBOX_URL = 'http://www.image-net.org/downloads/bbox/bbox/{}.tar.gz'


class UnpackInfo(object):
    def __init__(self, name, num_shards, output_dir=FLAGS.output_dir):
        self.name = name
        self.num_shards = num_shards
        self.output_dir = '{}/{}'.format(output_dir, name)
        self.labels = np.empty(0, dtype=int)
        self.filenames = np.empty(0)
        self.classnames = np.empty(0)
        self.human_classnames = np.empty(0)
        self.bboxes = np.empty(0)

    def add_data(self, filenames, label, classname):
        self.filenames = np.append(self.filenames, filenames)
        self.labels = np.append(self.labels, [label] * filenames.size)
        self.classnames = np.append(self.classnames, [classname] * filenames.size)

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

    def set_bboxes(self, bbox_lookup):
        num_image_bbox = 0
        bboxes = []
        for f in self.filenames:
            basename = os.path.basename(f)
            if basename in bbox_lookup:
                bboxes.append(bbox_lookup[basename])
                num_image_bbox += 1
            else:
                bboxes.append([])
        print('Found %d images with bboxes out of %d images' % (
            num_image_bbox, len(self.filenames)))
        self.bboxes = bboxes

    def __repr__(self):
        return 'Info: {self.name}'.format(self=self)


def _get_dl_urls(wnid):

    """
    Returns download url for images and bounding boxes
    :param str wnid: Wordnet id
    :return: Synset url and bbox url for the given wordnet id.
    """
    assert isinstance(ACCESS_PASS, str), 'Imagenet Access pass must be provided.'

    synset_url = SYNSET_URL.format(wnid, USER_ID, ACCESS_PASS)
    bbox_url = BBOX_URL.format(wnid)
    return [synset_url, bbox_url]


def _maybe_download(download_list):
    """Download Image-net synsets from file"""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    with open(download_list) as csvfile:
        list_reader = csv.reader(csvfile)
        for row in list_reader:

            wordnet_id = row[0]
            print(wordnet_id)
            urls = _get_dl_urls(wordnet_id)
            filenames = ['{}.tar'.format(wordnet_id), '{}.tar.gz'.format(wordnet_id)]

            for i, filename in enumerate(filenames):
                filepath = os.path.join(dest_directory, filename)

                if not os.path.exists(filepath):
                    download_file(urls[i], filepath)

                if row[2] == '0':
                    break

    # wnids = tf.gfile.FastGFile(download_list, 'r').readlines()
    # wnids = [x.strip() for x in wnids]
    # for wordnet_ID in wnids:
    #
    #     urls = _get_dl_urls(wordnet_ID)
    #     filenames = ['{}.tar'.format(wordnet_ID), '{}.tar.gz'.format(wordnet_ID)]
    #
    #     for i, filename in enumerate(filenames):
    #         filepath = os.path.join(dest_directory, filename)
    #
    #         if not os.path.exists(filepath):
    #             download_file(urls[i], filepath)


def download_file(url, filename):
    u = urllib2.urlopen(url)

    with open(filename, 'wb') as f:
        meta = u.info()
        meta_func = meta.getheaders if hasattr(meta, 'getheaders') else meta.get_all
        meta_length = meta_func("Content-Length")
        file_size = None
        if meta_length:
            file_size = int(meta_length[0])
        print("Downloading: {0} Bytes: {1}".format(url, file_size))

        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)

            status = "{0:16}".format(file_size_dl)
            if file_size:
                status += "   [{0:6.2f}%]".format(file_size_dl * 100 / file_size)
            status += chr(13)


def _write_synsets_file(synsets):
    with open(FLAGS.labels_file, 'w') as f:
        f.write('\n'.join(synsets))


def _unpack_data(data_dir, output_dir):
    """Unpack and split raw data into training and validation sets."""
    synset_labels = []
    label = 0

    validation_data = UnpackInfo(name='validation', num_shards=FLAGS.validation_shards)
    train_data = UnpackInfo(name='train', num_shards=FLAGS.train_shards)

    tar_format = '%s/*.tar' % data_dir
    data_files = tf.gfile.Glob(tar_format)

    for file in data_files:
        filename = file.split('.')[0].split('/')[-1]
        synset_labels.append(filename)
        label += 1

        with tarfile.open(file) as tar:
            path = '{}/{}/'.format(output_dir, filename)
            names = np.array(tar.getnames(), dtype=str)
            # noinspection PyTypeChecker
            file_prefix = np.full_like(names, fill_value=path)
            names = np.core.defchararray.add(file_prefix, names)
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=path)

        # Extract bbox
        bbox_file = '{}.gz'.format(file)
        if os.path.isfile(bbox_file):
            with tarfile.open(bbox_file) as tar_gz:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar_gz, path=output_dir)

        train_names, valid_names = np.split(names, [int(0.9 * names.size)])

        train_data.add_data(train_names, label, filename)
        validation_data.add_data(valid_names, label, filename)

    train_data.shuffle_data()
    validation_data.shuffle_data()

    _write_synsets_file(synset_labels)

    return train_data, validation_data, synset_labels


def _clean_temp_data(output_dir, u_labels):
    print(u_labels)
    for label in u_labels:
        shutil.rmtree('{}/{}'.format(output_dir, label))


def main(_):
    assert not FLAGS.train_shards % FLAGS.num_threads, \
        'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards'
    assert not FLAGS.validation_shards % FLAGS.num_threads, \
        'Please make the FLAGS.num_threads commensurate with FLAGS.validation_shards'

    # _maybe_download(FLAGS.download_list)

    # Unpack raw data
    train, valid, labels = _unpack_data(FLAGS.data_dir, FLAGS.output_dir)

    # Process bounding boxes
    process_bounding_boxes.run(FLAGS.labels_dir, labels, FLAGS.bounding_box_file)

    # # Build lookups
    synset_lookup, bbox_lookup = image_util.build_lookups(FLAGS.imagenet_metadata_file, FLAGS.bounding_box_file,
                                                          labels)
    train.set_human_classnames(synset_lookup)
    train.set_bboxes(bbox_lookup)
    valid.set_human_classnames(synset_lookup)
    valid.set_bboxes(bbox_lookup)
    #
    # # Process images and write TF Records (shards)
    image_util.process_image_files(train, FLAGS.num_threads)
    image_util.process_image_files(valid, FLAGS.num_threads)
    #
    # # Clean temp JPEG files
    _clean_temp_data(FLAGS.output_dir, labels)


if __name__ == '__main__':
    tf.app.run()

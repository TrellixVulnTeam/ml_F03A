#
# build_data
# ml
#

"""
Image util
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf
from data.image_coder import ImageCoder


def _build_synset_lookup(imagenet_metadata_file, u_labels):
    """Build lookup for synset to human-readable label.

    Args:
      imagenet_metadata_file: string, path to file containing mapping from
        synset to human-readable label.

        Assumes each line of the file looks like:

          n02119247    black fox
          n02119359    silver fox
          n02119477    red fox, Vulpes fulva

        where each line corresponds to a unique mapping. Note that each line is
        formatted as <synset>\t<human readable label>.

    Returns:
      Dictionary of synset to human labels, such as:
        'n02119022' --> 'red fox, Vulpes vulpes'
    """
    lines = tf.gfile.FastGFile(imagenet_metadata_file, 'r').readlines()
    synset_to_human = {}
    for l in lines:
        if l:
            parts = l.strip().split('\t')
            assert len(parts) == 2
            synset = parts[0]
            if synset in u_labels:
                human = parts[1]
                print(human)
                synset_to_human[synset] = human
    return synset_to_human


def _build_bounding_box_lookup(bounding_box_file):
    """Build a lookup from image file to bounding boxes.

    Args:
      bounding_box_file: string, path to file with bounding boxes annotations.

        Assumes each line of the file looks like:

          n00007846_64193.JPEG,0.0060,0.2620,0.7545,0.9940

        where each line corresponds to one bounding box annotation associated
        with an image. Each line can be parsed as:

          <JPEG file name>, <xmin>, <ymin>, <xmax>, <ymax>

        Note that there might exist mulitple bounding box annotations associated
        with an image file. This file is the output of process_bounding_boxes.py.

    Returns:
      Dictionary mapping image file names to a list of bounding boxes. This list
      contains 0+ bounding boxes.
    """
    lines = tf.gfile.FastGFile(bounding_box_file, 'r').readlines()
    images_to_bboxes = {}
    num_bbox = 0
    num_image = 0
    for l in lines:
        if l:
            parts = l.split(',')
            assert len(parts) == 5, ('Failed to parse: %s' % l)
            filename = parts[0]
            xmin = float(parts[1])
            ymin = float(parts[2])
            xmax = float(parts[3])
            ymax = float(parts[4])
            box = [xmin, ymin, xmax, ymax]

            if filename not in images_to_bboxes:
                images_to_bboxes[filename] = []
                num_image += 1
            images_to_bboxes[filename].append(box)
            num_bbox += 1

    print('Successfully read %d bounding boxes '
          'across %d images.' % (num_bbox, num_image))
    return images_to_bboxes


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, synset, human, bbox,
                        height, width):
    """Build an Example proto for an example.

    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image_buffer: string, JPEG encoding of RGB image
      label: integer, identifier for the ground truth for the network
      synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
      human: string, human-readable label, e.g., 'red fox, Vulpes vulpes'
      bbox: list of bounding boxes; each box is a list of integers
        specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong to
        the same label as the image label.
      height: integer, image height in pixels
      width: integer, image width in pixels
    Returns:
      Example proto
    """

    xmin = []
    ymin = []
    xmax = []
    ymax = []

    for b in bbox:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], b)]

    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/class/synset': _bytes_feature(tf.compat.as_bytes(synset)),
        'image/class/text': _bytes_feature(tf.compat.as_bytes(human)),
        'image/object/bbox/xmin': _float_feature(xmin),
        'image/object/bbox/xmax': _float_feature(xmax),
        'image/object/bbox/ymin': _float_feature(ymin),
        'image/object/bbox/ymax': _float_feature(ymax),
        'image/object/bbox/label': _int64_feature([label] * len(xmin)),
        'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
        'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
    return example


def _is_png(filename):
    """Determine if a file contains a PNG format image.

    Args:
      filename: string, path of the image file.

    Returns:
      boolean indicating if the image is a PNG.
    """
    return '.png' in filename


def _process_image(filename, coder):
    """Process a single image file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    # Convert any PNG to JPEG's for consistency.
    if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, dataset):
    """Processes and saves list of images as TFRecord in 1 thread.

    Args:
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
      thread_index: integer, unique batch to run index is within [0, len(ranges)).
      ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      texts: list of strings; each string is human readable, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth
      num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).

    name = dataset.name
    filenames = dataset.filenames
    texts = dataset.classnames
    labels = dataset.labels
    num_shards = dataset.num_shards
    output_dir = dataset.output_dir
    bboxes = dataset.bboxes
    humans = dataset.human_classnames

    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1], num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            text = texts[i]
            human = humans[i]
            bbox = bboxes[i]

            try:
                image_buffer, height, width = _process_image(filename, coder)
            except Exception as e:
                print(e)
                print('SKIPPED: Unexpected eror while decoding %s.' % filename)
                continue

            example = _convert_to_example(filename, image_buffer, label,
                                          text, human, bbox, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        sys.stdout.flush()
    sys.stdout.flush()


def process_image_files(dataset, num_threads):
    """Process and save list of images as TFRecord of Example protos.

    Args:
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      texts: list of strings; each string is human readable, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth
      num_shards: integer number of shards for this data set.
    """
    filenames = dataset.filenames
    texts = dataset.classnames
    labels = dataset.labels

    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)

    os.makedirs('{}'.format(dataset.output_dir), exist_ok=True)

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, dataset)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' % (datetime.now(), len(filenames)))
    sys.stdout.flush()


def build_lookups(imagenet_metadata_file, bounding_box_file, u_labels):
    synset_to_human = _build_synset_lookup(imagenet_metadata_file, u_labels)
    image_to_bboxes = _build_bounding_box_lookup(bounding_box_file)
    return synset_to_human, image_to_bboxes

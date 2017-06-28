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


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, text, height, width):
    """Build an Example proto for an example.

    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image_buffer: string, JPEG encoding of RGB image
      label: integer, identifier for the ground truth for the network
      text: string, unique human-readable, e.g. 'dog'
      height: integer, image height in pixels
      width: integer, image width in pixels
    Returns:
      Example proto
    """

    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
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


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               texts, labels, num_shards):
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
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            text = texts[i]

            try:
                image_buffer, height, width = _process_image(filename, coder)
            except Exception as e:
                print(e)
                print('SKIPPED: Unexpected eror while decoding %s.' % filename)
                continue

            example = _convert_to_example(filename, image_buffer, label,
                                          text, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
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
    name = dataset.name
    filenames = dataset.filenames
    texts = dataset.classnames
    labels = dataset.labels
    num_shards = dataset.num_shards

    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)

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
        args = (coder, thread_index, ranges, name, filenames,
                texts, labels, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()

#!/usr/bin/python
# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Process the ImageNet Challenge bounding boxes for TensorFlow model training.

This script is called as

process_bounding_boxes.py <dir> [synsets-file]

Where <dir> is a directory containing the downloaded and unpacked bounding box
data. If [synsets-file] is supplied, then only the bounding boxes whose
synstes are contained within this file are returned. Note that the
[synsets-file] file contains synset ids, one per line.

The script dumps out a CSV text file in which each line contains an entry.
  n00007846_64193.JPEG,0.0060,0.2620,0.7545,0.9940

The entry can be read as:
  <JPEG file name>, <xmin>, <ymin>, <xmax>, <ymax>

The bounding box for <JPEG file name> contains two points (xmin, ymin) and
(xmax, ymax) specifying the lower-left corner and upper-right corner of a
bounding box in *relative* coordinates.

The user supplies a directory where the XML files reside. The directory
structure in the directory <dir> is assumed to look like this:

<dir>/nXXXXXXXX/nXXXXXXXX_YYYY.xml

Each XML file contains a bounding box annotation. The script:

 (1) Parses the XML file and extracts the filename, label and bounding box info.

 (2) The bounding box is specified in the XML files as integer (xmin, ymin) and
    (xmax, ymax) *relative* to image size displayed to the human annotator. The
    size of the image displayed to the human annotator is stored in the XML file
    as integer (height, width).

    Note that the displayed size will differ from the actual size of the image
    downloaded from image-net.org. To make the bounding box annotation useable,
    we convert bounding box to floating point numbers relative to displayed
    height and width of the image.

    Note that each XML file might contain N bounding box annotations.

    Note that the points are all clamped at a range of [0.0, 1.0] because some
    human annotations extend outside the range of the supplied image.

    See details here: http://image-net.org/download-bboxes

(3) By default, the script outputs all valid bounding boxes. If a
    [synsets-file] is supplied, only the subset of bounding boxes associated
    with those synsets are outputted. Importantly, one can supply a list of
    synsets in the ImageNet Challenge and output the list of bounding boxes
    associated with the training images of the ILSVRC.

    We use these bounding boxes to inform the random distortion of images
    supplied to the network.

If you run this script successfully, you will see the following output
to stderr:
> Finished processing 544546 XML files.
> Skipped 0 XML files not in ImageNet Challenge.
> Skipped 0 bounding boxes not in ImageNet Challenge.
> Wrote 615299 bounding boxes from 544546 annotated images.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import glob
import os.path
import sys
import xml.etree.ElementTree as ET


class BoundingBox(object):
    pass


def GetItem(name, root, index=0):
    count = 0
    for item in root.iter(name):
        if count == index:
            return item.text
        count += 1
    # Failed to find "index" occurrence of item.
    return -1


def GetInt(name, root, index=0):
    # In some XML annotation files, the point values are not integers, but floats.
    # So we add a float function to avoid ValueError.
    return int(float(GetItem(name, root, index)))


def FindNumberBoundingBoxes(root):
    index = 0
    while True:
        if GetInt('xmin', root, index) == -1:
            break
        index += 1
    return index


def process_bbox_xml(xml_file):
    """Process a single XML file containing a bounding box."""
    # pylint: disable=broad-except
    try:
        tree = ET.parse(xml_file)
    except Exception:
        print('Failed to parse: ' + xml_file, file=sys.stderr)
        return None
    # pylint: enable=broad-except
    root = tree.getroot()

    num_boxes = FindNumberBoundingBoxes(root)
    boxes = []

    for index in range(num_boxes):
        box = BoundingBox()
        # Grab the 'index' annotation.
        box.xmin = GetInt('xmin', root, index)
        box.ymin = GetInt('ymin', root, index)
        box.xmax = GetInt('xmax', root, index)
        box.ymax = GetInt('ymax', root, index)

        box.width = GetInt('width', root)
        box.height = GetInt('height', root)
        box.filename = GetItem('filename', root) + '.JPEG'
        box.label = GetItem('name', root)

        xmin = float(box.xmin) / float(box.width)
        xmax = float(box.xmax) / float(box.width)
        ymin = float(box.ymin) / float(box.height)
        ymax = float(box.ymax) / float(box.height)

        min_x = min(xmin, xmax)
        max_x = max(xmin, xmax)
        box.xmin_scaled = min(max(min_x, 0.0), 1.0)
        box.xmax_scaled = min(max(max_x, 0.0), 1.0)

        min_y = min(ymin, ymax)
        max_y = max(ymin, ymax)
        box.ymin_scaled = min(max(min_y, 0.0), 1.0)
        box.ymax_scaled = min(max(max_y, 0.0), 1.0)

        boxes.append(box)

    return boxes


def run(labels_dir, labels, bounding_box_file):

    xml_files = glob.glob(labels_dir + '/*/*.xml')
    print('Identified %d XML files in %s' % (len(xml_files), labels_dir))

    # labels = set([l.strip() for l in open(labels_file).readlines()])
    print('Identified %d synset IDs' % (len(labels)))
    assert labels is not None and len(labels) > 1

    skipped_boxes = 0
    skipped_files = 0
    found_boxes = 0
    saved_boxes = 0
    saved_files = 0

    with open(bounding_box_file, 'w', newline='') as csvfile:
        bb_writer = csv.writer(csvfile)
        for file_index, one_file in enumerate(xml_files):

            label = os.path.basename(os.path.dirname(one_file))
            # Determine if the annotation is from an ImageNet Challenge label.
            if label not in labels:
                raise ValueError('Invalid bbox file')

            bboxes = process_bbox_xml(one_file)
            assert bboxes is not None, 'No bounding boxes found in ' + one_file

            found_box = False
            for bbox in bboxes:
                found_boxes += 1
                if bbox.label != label:
                    skipped_boxes += 1
                    # raise ValueError('Invalid bbox label')
                    continue

                # Guard against improperly specified boxes.
                if bbox.xmin_scaled >= bbox.xmax_scaled or bbox.ymin_scaled >= bbox.ymax_scaled:
                    skipped_boxes += 1
                    # raise ValueError('Invalid bbox scale')
                    continue

                image_filename = os.path.splitext(os.path.basename(one_file))[0]

                bb_writer.writerow(
                    ['%s.JPEG' % image_filename, '%.4f' % bbox.xmin_scaled,
                     '%.4f' % bbox.ymin_scaled, '%.4f' % bbox.xmax_scaled, '%.4f' % bbox.ymax_scaled]
                )

                saved_boxes += 1
                found_box = True

            if found_box:
                saved_files += 1
            else:
                skipped_files += 1

    print('Finished processing %d XML files.' % len(xml_files))
    print('Found {} bboxes in total'.format(found_boxes))
    print('Skipped %d XML files not in ImageNet Challenge.' % skipped_files)
    print('Skipped %d bounding boxes.' % skipped_boxes)
    print('Wrote %d bounding boxes from %d annotated images.' % (saved_boxes, saved_files))


if __name__ == '__main__':
    run(sys.argv)

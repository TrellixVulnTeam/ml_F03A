import os

import tensorflow as tf

try:
    from trainer2.input import ImagePreprocessor as ImProc
    from trainer2 import flags
except ImportError:
    from input import ImagePreprocessor as ImProc
    import flags

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

    def __init__(self, name, data_dir=None, image_size=64):
        self.name = name

        assert image_size > 0, 'Image size must be positive.'

        if isinstance(self, (ImgData, ImagenetData)):
            assert data_dir is not None, 'Invalid data_dir type, in Dataset constructor.'
            try:
                assert os.path.isdir(data_dir)
            except AssertionError:
                if 'gs://' in data_dir:
                    pass
                else:
                    raise AssertionError('Invalid data_dir parameter')

        self.data_dir = data_dir
        self.image_size = image_size
        self.input_channels = 3
        self.input_data_type = tf.float32
        self.resize_method = FLAGS.resize_method
        self.synthetic = False

    def tf_record_pattern(self, subset):
        return os.path.join(self.data_dir, '%s-*-of-*' % subset)

    def preprocess(self, batch_size, num_comp_devices, train=True):
        pass

    @staticmethod
    def reader():
        return tf.TFRecordReader()

    def num_classes(self):
        pass

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
        """
        Preprocess image data.
        :param batch_size:
        :param num_comp_devices:
        :param train:
        :return:
        """
        assert num_comp_devices > 0
        assert batch_size > 0
        assert isinstance(train, bool)

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

        return images, labels


class SyntheticData(Dataset):
    def __init__(self, data_dir=None):
        super(SyntheticData, self).__init__('synthetic-data', data_dir)
        self.synthetic = True

    def num_classes(self):
        return 1000

    def num_examples_per_epoch(self, subset='train'):
        assert False, 'Should not be called for Synthetic data.'

    def preprocess(self, batch_size, num_comp_devices, train=True):
        """Add image Preprocessing ops to tf graph."""
        assert num_comp_devices > 0
        assert batch_size > 0
        assert isinstance(train, bool)

        nclass = 1001
        input_shape = [batch_size, self.image_size, self.image_size, self.input_channels]
        images = tf.truncated_normal(input_shape, dtype=self.input_data_type, stddev=1e-1, name='synthetic_images')
        labels = tf.random_uniform([batch_size], minval=1, maxval=nclass, dtype=tf.int32, name='synthetic_labels')

        images = tf.contrib.framework.local_variable(images, name='images')
        labels = tf.contrib.framework.local_variable(labels, name='labels')

        labels -= 1
        if num_comp_devices == 1:
            images_splits = [images]
            labels_splits = [labels]
        else:
            images_splits = tf.split(images, num_comp_devices, 0)
            labels_splits = tf.split(labels, num_comp_devices, 0)

        return images_splits, labels_splits


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

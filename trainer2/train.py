import tensorflow as tf


class Trainer(object):
    """Class for model training."""

    def __init__(self, model, task):
        self.model = model
        self.task = task
        self.sv = None
        self.saver = None
        self.summary_op = None
        self.devices = ['1', '2']

    def run(self):
        """Run trainer."""
        with tf.Graph().as_default():
            self.train()

    def train(self):
        is_chief = self.task.index == 0
        device_fn = ''
        global_step = tf.contrib.framework.get_global_step()

        # Handle cluster config

        # Build model
        with tf.device(device_fn):
            # Create a saver for writing training checkpoints.
            self.saver = tf.train.Saver()
            self.summary_op = tf.summary.merge_all()

        self.set_sv(is_chief, global_step)

        with self.sv.managed_session('', config=None) as sess:
            print('sess running')
            # while not self.sv.should_stop() and global_step < 1000:

    def build_graph(self):

        with tf.device():
            global_step = tf.contrib.framework.get_or_create_global_step()
            nclass, images_splits, labels_splits = add_image_preprocessing()

        update_ops = None

        for dev in self.devices:
            print(dev)

    def set_sv(self, is_chief, global_step):
        self.sv = tf.train.Supervisor(
            is_chief=is_chief,
            logdir='train_dir',
            init_op=tf.global_variables_initializer(),
            saver=tf.train.Saver(),
            summary_op=None,
            global_step=global_step,
            save_model_secs=0,
            summary_writer=None
        )


def add_image_preprocessing(
                            input_nchan=3,
                            image_size=32,
                            batch_size=32,
                            num_compute_devices=1,
                            input_data_type=tf.float32,
                            ):
    """Add image Preprocessing ops to tf graph."""
    nclass = 1001
    input_shape = [batch_size, image_size, image_size, input_nchan]
    images = tf.truncated_normal(
        input_shape,
        dtype=input_data_type,
        stddev=1e-1,
        name='synthetic_images')
    labels = tf.random_uniform(
        [batch_size],
        minval=1,
        maxval=nclass,
        dtype=tf.int32,
        name='synthetic_labels')
    # Note: This results in a H2D copy, but no computation
    # Note: This avoids recomputation of the random values, but still
    #         results in a H2D copy.
    images = tf.contrib.framework.local_variable(images, name='images')
    labels = tf.contrib.framework.local_variable(labels, name='labels')
    # Change to 0-based (don't use background class like Inception does)
    labels -= 1
    if num_compute_devices == 1:
        images_splits = [images]
        labels_splits = [labels]
    else:
        images_splits = tf.split(images, num_compute_devices, 0)
        labels_splits = tf.split(labels, num_compute_devices, 0)
    return nclass, images_splits, labels_splits

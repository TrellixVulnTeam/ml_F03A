# https://danijar.github.io/structuring-your-tensorflow-models
import numpy as np
import six
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops as df_ops

from trainer2 import datasets
from trainer2 import flags
from trainer2.layer import PoolLayer, AffineLayer, Conv2dLayer2, ReshapeLayer

FLAGS = flags.get_flags()


class Model:
    def __init__(self, name, layers, run_training=FLAGS.run_training, activation=tf.nn.relu,
                 initial_lr=FLAGS.learning_rate, data_format=FLAGS.data_format, num_batches=FLAGS.num_batches,
                 batch_size=FLAGS.batch_size):
        """

        :param name:
        :param layers:
        :param run_training:
        :param activation:
        :param initial_lr:
        :param data_format:
        """
        assert initial_lr > 0.0
        assert len(layers) > 0
        assert name is not None

        self.name = name
        self.layers = layers
        self.run_training = run_training
        self.num_batches = num_batches
        self.learning_rate = initial_lr
        self.batch_size = batch_size
        self.activation = activation
        self.data_type = tf.float32
        self.data_format = data_format
        # self.input_data_type = tf.float32
        self.dataset = datasets.DatasetFactory().create_dataset(data_dir=FLAGS.data_dir)

    def get_layers(self, inputs):

        next_inputs = inputs

        for layer in self.layers:
            layer.activation = self.activation
            layer.inputs = next_inputs
            next_inputs = layer.outputs()
            assert next_inputs is not None

        return next_inputs

    # def get_l2_losses(self):
    #     weights = []
    #     for layer in self.layers:
    #         if isinstance(layer, (Conv2dLayer2, AffineLayer)):
    #             weights.append(tf.nn.l2_loss(layer.weights))
    #     return self.l2_loss * sum(weights)

    def build_graph(self, config, manager):

        tf.set_random_seed(1234)
        np.random.seed(4321)

        enqueue_ops = []
        losses = []
        device_grads = []
        all_logits = []
        all_top_1_ops = []
        all_top_5_ops = []

        gpu_copy_stage_ops = []
        gpu_compute_stage_ops = []
        gpu_grad_stage_ops = []

        with tf.device(config.global_step_device):
            global_step = tf.contrib.framework.get_or_create_global_step()

        with tf.device(config.cpu_device):
            images_splits, labels_splits = self.dataset.preprocess(self.batch_size, len(manager.devices),
                                                                   self.run_training)

        update_ops = None
        staging_delta_ops = []

        for dev in range(len(manager.devices)):

            with manager.create_outer_variable_scope(dev), tf.name_scope('tower_%i' % dev) as name_scope:

                if self.dataset.synthetic:
                    images = tf.truncated_normal(images_splits[dev].get_shape(), dtype=self.data_type, stddev=1e-1,
                                                 name='synthetic_images')
                    images = tf.contrib.framework.local_variable(images, name='gpu_cached_images')
                    labels = labels_splits[dev]
                else:
                    images, labels = self.cpu_gpu_copy(config.cpu_device, config.raw_devices[dev], images_splits[dev],
                                                       labels_splits[dev], gpu_copy_stage_ops, gpu_compute_stage_ops)

                results = self.forward_pass(images, labels, dev, manager.devices[dev], gpu_compute_stage_ops,
                                            manager.trainable_variables_on_device)

                if self.run_training:
                    losses.append(results[0])
                    device_grads.append(results[1])
                else:
                    all_logits.append(results[0])
                    all_top_1_ops.append(results[1])
                    all_top_5_ops.append(results[2])

                if manager.retain_tower_updates(dev):
                    # Retain the Batch Normalization updates operations only from the
                    # first tower. Ideally, we should grab the updates from all towers but
                    # these stats accumulate extremely fast so we can ignore the other
                    # stats from the other towers without significant detriment.
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
                    staging_delta_ops = list(manager.staging_delta_ops)

        if not update_ops:
            assert isinstance(name_scope, object)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)

        enqueue_ops.append(tf.group(*gpu_copy_stage_ops))

        if manager.supports_staged_vars():
            for staging_ops in manager.staging_vars_on_devices:
                gpu_compute_stage_ops.extend(
                    [put_op for _, (put_op, _) in six.iteritems(staging_ops)])

        enqueue_ops.append(tf.group(*gpu_compute_stage_ops))

        if gpu_grad_stage_ops:
            staging_delta_ops += gpu_grad_stage_ops
        if staging_delta_ops:
            enqueue_ops.append(tf.group(*staging_delta_ops))

        if not self.run_training:
            all_top_1_ops = tf.reduce_sum(all_top_1_ops)
            all_top_5_ops = tf.reduce_sum(all_top_5_ops)
            fetches = [all_top_1_ops, all_top_5_ops] + enqueue_ops
            return enqueue_ops, fetches

        extra_nccl_ops = []
        apply_gradient_devices, gradient_state = manager.preprocess_device_grads(device_grads)

        training_ops = []
        for d, device in enumerate(apply_gradient_devices):
            with tf.device(device):
                total_loss = tf.reduce_mean(losses)
                avg_grads = manager.get_gradients_to_apply(d, gradient_state)

                gradient_clip = FLAGS.gradient_clip

                if (not self.dataset.synthetic) and FLAGS.num_epochs_per_decay > 0:
                    num_batches_per_epoch = (self.dataset.num_examples_per_epoch() / self.batch_size)
                    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

                    # Decay the learning rate exponentially based on the number of steps.
                    self.learning_rate = tf.train.exponential_decay(self.learning_rate, global_step,
                                                                    decay_steps, FLAGS.learning_rate_decay_factor,
                                                                    staircase=True)

                if gradient_clip is not None:
                    clipped_grads = [(tf.clip_by_value(grad, -gradient_clip, +gradient_clip), var)
                                     for grad, var in avg_grads]
                else:
                    clipped_grads = avg_grads

                # Set optimizer for training
                optimizer = self.get_optimizer(FLAGS.optimizer)
                # optimizer = tf.train.SyncReplicasOptimizer(self.get_optimizer(FLAGS.optimizer), 2, 2)
                manager.append_apply_gradients_ops(gradient_state, optimizer, clipped_grads, training_ops)

                # Track the moving averages of all trainable variables.
                # variable_averages = tf.train.ExponentialMovingAverage(0.999, global_step)
                # variables_averages_op = variable_averages.apply(tf.trainable_variables())
                # training_ops.append(variables_averages_op)
                # training_ops = tf.group(training_ops, variables_averages_op)

        train_op = tf.group(*(training_ops + update_ops + extra_nccl_ops))

        with tf.device(config.cpu_device):
            if config.is_chief and FLAGS.write_summary > 0:
                tf.summary.scalar('learning_rate', self.learning_rate)
                tf.summary.scalar('total_loss', total_loss)
                for grad, var in avg_grads:
                    if grad is not None:
                        tf.summary.histogram(var.op.name + '/gradients', grad)
                for var in tf.trainable_variables():
                    tf.summary.histogram(var.op.name, var)
        fetches = [train_op, total_loss] + enqueue_ops
        return enqueue_ops, fetches

    def get_optimizer(self, optimizer_flag):
        """
        Get Training Optimizer from runtime flag.
        :param optimizer_flag:
        :return: Training Optimizer
        """
        assert optimizer_flag in ['momentum', 'sgd', 'rmsprop'], 'Invalid optimizer value: {}'.format(optimizer_flag)

        if optimizer_flag == 'momentum':
            return tf.train.MomentumOptimizer(self.learning_rate, FLAGS.momentum, use_nesterov=True)
        elif optimizer_flag == 'sgd':
            return tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            return tf.train.RMSPropOptimizer(self.learning_rate, FLAGS.rmsprop_decay, momentum=FLAGS.momentum,
                                             epsilon=FLAGS.rmsprop_epsilon)

    @staticmethod
    def cpu_gpu_copy(cpu_device, raw_device, host_images, host_labels, gpu_copy_stage_ops, gpu_compute_stage_ops):
        with tf.device(cpu_device):
            images_shape = host_images.get_shape()
            labels_shape = host_labels.get_shape()
            gpu_copy_stage = df_ops.StagingArea([tf.float32, tf.int32], shapes=[images_shape, labels_shape])
            gpu_copy_stage_op = gpu_copy_stage.put([host_images, host_labels])
            gpu_copy_stage_ops.append(gpu_copy_stage_op)
            host_images, host_labels = gpu_copy_stage.get()

        with tf.device(raw_device):
            gpu_compute_stage = df_ops.StagingArea([tf.float32, tf.int32], shapes=[images_shape, labels_shape])
            # The CPU-to-GPU copy is triggered here.
            gpu_compute_stage_op = gpu_compute_stage.put([host_images, host_labels])
            images, labels = gpu_compute_stage.get()
            images = tf.reshape(images, shape=images_shape)
            gpu_compute_stage_ops.append(gpu_compute_stage_op)
            return images, labels

    def reformat_images(self, images):
        # Rescale to [0, 1)
        images *= 1. / 256
        # Rescale to [-1,1] instead of [0, 1)
        images = tf.subtract(images, 0.5)
        images = tf.multiply(images, 2.0)

        if self.data_format == 'NCHW':
            images = tf.transpose(images, [0, 3, 1, 2])

        if self.dataset.input_data_type != self.data_type:
            images = tf.cast(images, self.data_type)
        return images

    @staticmethod
    def get_grads(loss, params, gpu_grad_stage_ops):
        aggmeth = tf.AggregationMethod.DEFAULT
        grads = tf.gradients(loss, params, aggregation_method=aggmeth)

        if FLAGS.staged_vars:
            grad_dtypes = [grad.dtype for grad in grads]
            grad_shapes = [grad.shape for grad in grads]
            grad_stage = df_ops.StagingArea(grad_dtypes, grad_shapes)
            grad_stage_op = grad_stage.put(grads)
            # In general, this decouples the computation of the gradients and the updates of the weights.
            # During the pipeline warm up, this runs enough training to produce  the first set of gradients.
            gpu_grad_stage_ops.append(grad_stage_op)
            grads = grad_stage.get()

        return grads

    def forward_pass(self, images, labels, device_num, context_device, gpu_grad_stage_ops, trainable_variables_on_device):

        """Add ops for forward-pass and gradient computations."""

        with tf.device(context_device):
            images = self.reformat_images(images)
            logits = self.get_layers(images)

            if not self.run_training:
                top_1_op = tf.reduce_sum(tf.cast(tf.nn.in_top_k(logits, labels, 1), self.data_type))
                top_5_op = tf.reduce_sum(tf.cast(tf.nn.in_top_k(logits, labels, 5), self.data_type))
                return logits, top_1_op, top_5_op

            loss = loss_function(logits, labels)
            params = trainable_variables_on_device(device_num)

            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in params])
            weight_decay = FLAGS.weight_decay
            if weight_decay is not None and weight_decay != 0.:
                loss += weight_decay * l2_loss

            grads = self.get_grads(loss, params, gpu_grad_stage_ops)

            # aggmeth = tf.AggregationMethod.DEFAULT
            # grads = tf.gradients(loss, params, aggregation_method=aggmeth)
            #
            # if FLAGS.staged_vars:
            #     grad_dtypes = [grad.dtype for grad in grads]
            #     grad_shapes = [grad.shape for grad in grads]
            #     grad_stage = df_ops.StagingArea(grad_dtypes, grad_shapes)
            #     grad_stage_op = grad_stage.put(grads)
            #     # In general, this decouples the computation of the gradients and the updates of the weights.
            #     # During the pipeline warm up, this runs enough training to produce  the first set of gradients.
            #     gpu_grad_stage_ops.append(grad_stage_op)
            #     grads = grad_stage.get()

            param_refs = trainable_variables_on_device(device_num, writable=True)
            gradvars = list(zip(grads, param_refs))
            return loss, gradvars

    @classmethod
    def trial(cls):
        layers = [
            Conv2dLayer2(3, 32, 5, 5, 'conv_1'),
            PoolLayer('pool_1'),
            Conv2dLayer2(32, 64, 5, 5, 'conv_2'),
            PoolLayer('pool_2'),
            Conv2dLayer2(64, 128, 5, 5, 'conv_3'),
            PoolLayer('pool_3'),
            ReshapeLayer(output_shape=[-1, 128 * 8 * 8]),
            AffineLayer('fc_1', 8192, 512),
            AffineLayer('output', 512, 1001, final_layer=True)
        ]
        return cls(name='trial', layers=layers, run_training=FLAGS.run_training, activation=tf.nn.relu)


def loss_function(logits, labels):
    # global cross_entropy # HACK TESTING
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy')
    loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    return loss

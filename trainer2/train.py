import tensorflow as tf

from tensorflow.python.ops import data_flow_ops

from trainer2 import manager
from trainer2 import flags

FLAGS = flags.get_flags()


class Trainer(object):
    """Class for model training."""

    def __init__(self, model, task):
        self.model = model
        self.task = task
        self.sv = None
        self.saver = None
        self.summary_op = None
        self.data_format = FLAGS.data_format
        self.manager = manager.VariableMgrLocalFetchFromPS(self)

        self.task_index = FLAGS.task_index
        self.local_parameter_device_flag = FLAGS.local_parameter_device
        self.param_server_device = '/%s:0' % FLAGS.local_parameter_device
        self.sync_queue_devices = [self.param_server_device]

        self.job_name = FLAGS.job_name
        worker_prefix = ''
        self.cpu_device = '%s/cpu:0' % worker_prefix
        self.raw_devices = ['%s/%s:%i' % (worker_prefix, FLAGS.device, i) for i in range(FLAGS.num_gpus)]

        self.devices = self.manager.get_devices()
        if self.job_name:
            self.global_step_device = self.param_server_device
        else:
            self.global_step_device = self.cpu_device

    def run(self):
        """Run trainer."""
        with tf.Graph().as_default():
            self.train()

    def train(self):

        (enqueue_ops, fetches) = self.build_graph()
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

        phase_train = True
        use_synthetic_gpu_images = True

        data_type = tf.float32
        input_data_type = tf.float32
        input_nchan = 3

        enqueue_ops = []
        losses = []
        device_grads = []
        all_logits = []
        all_top_1_ops = []
        all_top_5_ops = []

        gpu_copy_stage_ops = []
        gpu_compute_stage_ops = []
        gpu_grad_stage_ops = []

        with tf.device(self.global_step_device):
            global_step = tf.contrib.framework.get_or_create_global_step()

        with tf.device(self.cpu_device):
            nclass, images_splits, labels_splits = add_image_preprocessing()

        update_ops = None

        for dev in range(len(self.devices)):
            with self.manager.create_outer_variable_scope(dev), tf.name_scope('tower_%i' % dev) as name_scope:
                results = self.add_forward_pass_and_gradients(
                    images_splits[dev], labels_splits[dev], nclass,
                    phase_train, dev, input_data_type, data_type, input_nchan,
                    use_synthetic_gpu_images, gpu_copy_stage_ops, gpu_compute_stage_ops,
                    gpu_grad_stage_ops)
                if phase_train:
                    losses.append(results[0])
                    device_grads.append(results[1])
                else:
                    all_logits.append(results[0])
                    all_top_1_ops.append(results[1])
                    all_top_5_ops.append(results[2])

                if self.manager.retain_tower_updates(dev):
                    # Retain the Batch Normalization updates operations only from the
                    # first tower. Ideally, we should grab the updates from all towers but
                    # these stats accumulate extremely fast so we can ignore the other
                    # stats from the other towers without significant detriment.
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
                    staging_delta_ops = list(self.manager.staging_delta_ops)

        if not update_ops:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)

        enqueue_ops.append(tf.group(*gpu_copy_stage_ops))
        enqueue_ops.append(tf.group(*gpu_compute_stage_ops))

        if gpu_grad_stage_ops:
            staging_delta_ops += gpu_grad_stage_ops
        if staging_delta_ops:
            enqueue_ops.append(tf.group(*(staging_delta_ops)))

        extra_nccl_ops = []
        apply_gradient_devices, gradient_state = (
            self.manager.preprocess_device_grads(device_grads))

        training_ops = []
        for d, device in enumerate(apply_gradient_devices):
            with tf.device(device):
                total_loss = tf.reduce_mean(losses)
                avg_grads = self.manager.get_gradients_to_apply(d, gradient_state)

                gradient_clip = None
                learning_rate = self.model.initial_lr

                if gradient_clip is not None:
                    clipped_grads = [
                        (tf.clip_by_value(grad, -gradient_clip, +gradient_clip), var)
                        for grad, var in avg_grads
                    ]
                else:
                    clipped_grads = avg_grads
                # Set optimizer for training
                opt = tf.train.GradientDescentOptimizer(learning_rate)
                self.manager.append_apply_gradients_ops(
                    gradient_state, opt, clipped_grads, training_ops)
        train_op = tf.group(*(training_ops + update_ops + extra_nccl_ops))

        with tf.device(self.cpu_device):
            if self.task_index == 0 and FLAGS.summary_verbosity > 0:
                tf.summary.scalar('learning_rate', learning_rate)
                tf.summary.scalar('total_loss', total_loss)
                for grad, var in avg_grads:
                    if grad is not None:
                        tf.summary.histogram(var.op.name + '/gradients', grad)
                for var in tf.trainable_variables():
                    tf.summary.histogram(var.op.name, var)
        fetches = [train_op, total_loss] + enqueue_ops
        return enqueue_ops, fetches

    def add_forward_pass_and_gradients(
            self, host_images, host_labels, nclass, phase_train, device_num,
            input_data_type, data_type, input_nchan, use_synthetic_gpu_images,
            gpu_copy_stage_ops, gpu_compute_stage_ops, gpu_grad_stage_ops):
        """Add ops for forward-pass and gradient computations."""
        if not use_synthetic_gpu_images:
            with tf.device(self.cpu_device):
                images_shape = host_images.get_shape()
                labels_shape = host_labels.get_shape()
                gpu_copy_stage = data_flow_ops.StagingArea(
                    [tf.float32, tf.int32],
                    shapes=[images_shape, labels_shape])
                gpu_copy_stage_op = gpu_copy_stage.put(
                    [host_images, host_labels])
                gpu_copy_stage_ops.append(gpu_copy_stage_op)
                host_images, host_labels = gpu_copy_stage.get()

        with tf.device(self.raw_devices[device_num]):
            if not use_synthetic_gpu_images:
                gpu_compute_stage = data_flow_ops.StagingArea(
                    [tf.float32, tf.int32],
                    shapes=[images_shape, labels_shape]
                )
                # The CPU-to-GPU copy is triggered here.
                gpu_compute_stage_op = gpu_compute_stage.put(
                    [host_images, host_labels])
                images, labels = gpu_compute_stage.get()
                images = tf.reshape(images, shape=images_shape)
                gpu_compute_stage_ops.append(gpu_compute_stage_op)
            else:
                # Minor hack to avoid H2D copy when using synthetic data
                images = tf.truncated_normal(
                    host_images.get_shape(),
                    dtype=input_data_type,
                    stddev=1e-1,
                    name='synthetic_images')
                images = tf.contrib.framework.local_variable(
                    images, name='gpu_cached_images')
                labels = host_labels

        with tf.device(self.devices[device_num]):
            # Rescale to [0, 1)
            images *= 1. / 256
            # Rescale to [-1,1] instead of [0, 1)
            images = tf.subtract(images, 0.5)
            images = tf.multiply(images, 2.0)

            if self.data_format == 'NCHW':
                images = tf.transpose(images, [0, 3, 1, 2])

            if input_data_type != data_type:
                images = tf.cast(images, data_type)

            # network = ConvNetBuilder(images, input_nchan, phase_train, self.data_format, data_type)
            # self.model_conf.add_inference(network)
            # # Add the final fully-connected class layer
            # logits = network.affine(nclass, activation='linear')

            logits = self.model.get_layers(images)

            if not phase_train:
                top_1_op = tf.reduce_sum(
                    tf.cast(tf.nn.in_top_k(logits, labels, 1), data_type))
                top_5_op = tf.reduce_sum(
                    tf.cast(tf.nn.in_top_k(logits, labels, 5), data_type))
                return logits, top_1_op, top_5_op
            loss = loss_function(logits, labels)
            params = self.manager.trainable_variables_on_device(device_num)
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in params])
            weight_decay = FLAGS.weight_decay
            if weight_decay is not None and weight_decay != 0.:
                loss += weight_decay * l2_loss

            aggmeth = tf.AggregationMethod.DEFAULT
            grads = tf.gradients(loss, params, aggregation_method=aggmeth)

            if FLAGS.staged_vars:
                grad_dtypes = [grad.dtype for grad in grads]
                grad_shapes = [grad.shape for grad in grads]
                grad_stage = data_flow_ops.StagingArea(grad_dtypes, grad_shapes)
                grad_stage_op = grad_stage.put(grads)
                # In general, this decouples the computation of the gradients and
                # the updates of the weights.
                # During the pipeline warm up, this runs enough training to produce
                # the first set of gradients.
                gpu_grad_stage_ops.append(grad_stage_op)
                grads = grad_stage.get()

            param_refs = self.manager.trainable_variables_on_device(
                device_num, writable=True)
            gradvars = list(zip(grads, param_refs))
            return loss, gradvars

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


def loss_function(logits, labels):
    # global cross_entropy # HACK TESTING
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss

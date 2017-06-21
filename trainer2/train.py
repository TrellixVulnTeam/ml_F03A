from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import gfile
from tensorflow.python.client import timeline

from trainer2 import manager
from trainer2 import flags
from trainer2.global_step import GlobalStepWatcher
FLAGS = flags.get_flags()
log_fn = print


class Trainer(object):
    """Class for model training."""

    def __init__(self, model, task):
        self.model = model
        self.task = task
        self.trace_filename = FLAGS.trace_file
        self.num_batches = FLAGS.num_batches
        self.data_format = FLAGS.data_format
        self.manager = manager.VariableMgrLocalFetchFromPS(self)
        self.ps_hosts = FLAGS.ps_hosts.split(',')
        self.worker_hosts = FLAGS.worker_hosts.split(',')

        autotune_threshold = FLAGS.autotune_threshold if (
            FLAGS.autotune_threshold) else 1
        min_autotune_warmup = 5 * autotune_threshold * autotune_threshold
        self.num_warmup_batches = FLAGS.num_warmup_batches if (
            FLAGS.num_warmup_batches) else max(10, min_autotune_warmup)
        self.graph_file = FLAGS.graph_file
        self.resize_method = FLAGS.resize_method
        self.sync_queue_counter = 0
        self.num_gpus = FLAGS.num_gpus
        self.batch_size = self.model.batch_size * self.num_gpus

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
        main_fetch_group = tf.group(*fetches)

        execution_barrier = None
        if self.job_name and not FLAGS.cross_replica_sync:
            execution_barrier = self.add_sync_queues_and_barrier('execution_barrier_', [])

        global_step = tf.contrib.framework.get_global_step()
        with tf.device(self.global_step_device):
            with tf.control_dependencies([main_fetch_group]):
                inc_global_step = global_step.assign_add(1)
                fetches.append(inc_global_step)

        if self.job_name and FLAGS.cross_replica_sync:
            # Block all replicas until all replicas are ready for next step.
            fetches.append(self.add_sync_queues_and_barrier('sync_queues_step_end_', [main_fetch_group]))

        variable_mgr_post_init_ops = self.manager.get_post_init_ops()
        if variable_mgr_post_init_ops:
            post_init_op_group = tf.group(*variable_mgr_post_init_ops)
        else:
            post_init_op_group = None

        local_var_init_op = tf.local_variables_initializer()
        summary_op = tf.summary.merge_all()
        is_chief = (not self.job_name or self.task_index == 0)
        summary_writer = None
        if is_chief and FLAGS.summary_verbosity and FLAGS.train_dir and FLAGS.save_summaries_steps > 0:
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir, tf.get_default_graph())

        sv = self.set_sv(is_chief, global_step, summary_writer)

        step_train_times = []
        with sv.managed_session(
                master='', config=create_config_proto(), start_standard_services=FLAGS.summary_verbosity > 0) as sess:

            for i in range(len(enqueue_ops)):
                sess.run(enqueue_ops[:(i + 1)])
            sess.run(local_var_init_op)
            if post_init_op_group:
                sess.run(post_init_op_group)

            init_global_step = 0

            global_step_watcher = GlobalStepWatcher(
                sess, global_step,
                len(self.worker_hosts) * self.num_warmup_batches
                + init_global_step,
                len(self.worker_hosts) * (self.num_warmup_batches
                                          + self.num_batches) - 1)
            global_step_watcher.start()

            if self.graph_file is not None:
                path, filename = os.path.split(self.graph_file)
                as_text = filename.endswith('txt')
                log_fn('Writing GraphDef as %s to %s' % (
                    'text' if as_text else 'binary', self.graph_file))
                tf.train.write_graph(sess.graph_def, path, filename, as_text)

            log_fn('Running warm up')
            local_step = -1 * self.num_warmup_batches

            if FLAGS.cross_replica_sync and FLAGS.job_name:
                # In cross-replica sync mode, all workers must run the same number of
                # local steps, or else the workers running the extra step will block.
                done_fn = lambda: local_step == self.num_batches
            else:
                done_fn = lambda: global_step_watcher.done()
            while not done_fn():
                if local_step == 0:
                    log_fn('Done warm up')
                    if execution_barrier:
                        log_fn('Waiting for other replicas to finish warm up')
                        assert global_step_watcher.start_time == 0
                        sess.run([execution_barrier])

                    log_fn('Step\tImg/sec\tloss')
                    assert len(step_train_times) == self.num_warmup_batches
                    step_train_times = []  # reset to ignore warm up batches
                if summary_writer and (local_step + 1) % FLAGS.save_summaries_steps == 0:
                    fetch_summary = summary_op
                else:
                    fetch_summary = None
                summary_str = benchmark_one_step(
                    sess, fetches, local_step, self.batch_size,
                    step_train_times,
                    self.trace_filename, fetch_summary)
                if summary_str is not None and is_chief:
                    sv.summary_computed(sess, summary_str)
                local_step += 1
            # Waits for the global step to be done, regardless of done_fn.
            while not global_step_watcher.done():
                time.sleep(.25)
            images_per_sec = global_step_watcher.steps_per_second() * self.batch_size
            log_fn('-' * 64)
            log_fn('total images/sec: %.2f' % images_per_sec)
            log_fn('-' * 64)

            # Save the model checkpoint.
            if FLAGS.train_dir is not None and is_chief:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                if not gfile.Exists(FLAGS.train_dir):
                    gfile.MakeDirs(FLAGS.train_dir)
                sv.saver.save(sess, checkpoint_path, global_step)

            if execution_barrier:
                # Wait for other workers to reach the end, so this worker
                #  doesn't go away underneath them.
                sess.run([execution_barrier])
        sv.stop()

    def build_graph(self):

        phase_train = not (FLAGS.eval or FLAGS.forward_only)
        use_synthetic_gpu_images = True

        data_type = tf.float32
        input_data_type = tf.float32
        input_nchan = 3
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

        if not phase_train:
            if FLAGS.forward_only:
                all_logits = tf.concat(all_logits, 0)
                fetches = [all_logits] + enqueue_ops
            else:
                all_top_1_ops = tf.reduce_sum(all_top_1_ops)
                all_top_5_ops = tf.reduce_sum(all_top_5_ops)
                fetches = [all_top_1_ops, all_top_5_ops] + enqueue_ops
            return (enqueue_ops, fetches)

        extra_nccl_ops = []
        apply_gradient_devices, gradient_state = (
            self.manager.preprocess_device_grads(device_grads))

        training_ops = []
        for d, device in enumerate(apply_gradient_devices):
            with tf.device(device):
                total_loss = tf.reduce_mean(losses)
                avg_grads = self.manager.get_gradients_to_apply(d, gradient_state)

                gradient_clip = FLAGS.gradient_clip
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

    def add_sync_queues_and_barrier(self, name_prefix, enqueue_after_list):
        """Adds ops to enqueue on all worker queues.

        Args:
          name_prefix: prefixed for the shared_name of ops.
          enqueue_after_list: control dependency from ops.

        Returns:
          an op that should be used as control dependency before starting next step.
        """
        self.sync_queue_counter += 1
        num_workers = self.cluster.num_tasks('worker')
        with tf.device(self.sync_queue_devices[self.sync_queue_counter % len(self.sync_queue_devices)]):
            sync_queues = [
                tf.FIFOQueue(num_workers, [tf.bool], shapes=[[]],
                             shared_name='%s%s' % (name_prefix, i))
                for i in range(num_workers)]
            queue_ops = []
            # For each other worker, add an entry in a queue, signaling that it can
            # finish this step.
            token = tf.constant(False)
            with tf.control_dependencies(enqueue_after_list):
                for i, q in enumerate(sync_queues):
                    if i == self.task_index:
                        queue_ops.append(tf.no_op())
                    else:
                        queue_ops.append(q.enqueue(token))

            # Drain tokens off queue for this worker, one for each other worker.
            queue_ops.append(
                sync_queues[self.task_index].dequeue_many(
                    len(sync_queues) - 1))

            return tf.group(*queue_ops)

    def set_sv(self, is_chief, global_step, summary_writer):
        return tf.train.Supervisor(
            is_chief=is_chief,
            logdir=FLAGS.train_dir,
            saver=tf.train.Saver(tf.global_variables()),
            global_step=global_step,
            summary_op=None,
            save_model_secs=FLAGS.save_model_secs,
            summary_writer=summary_writer
        )


def add_image_preprocessing(
                            input_nchan=3,
                            image_size=28,
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


def create_config_proto():
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.intra_op_parallelism_threads = FLAGS.num_intra_threads
    config.inter_op_parallelism_threads = FLAGS.num_inter_threads
    # config.gpu_options.force_gpu_compatible = FLAGS.force_gpu_compatible
    # config.gpu_options.force_gpu_compatible = True
    return config


def benchmark_one_step(sess, fetches, step, batch_size, step_train_times,
                       trace_filename, summary_op=None):
    """Advance one step of benchmarking."""
    if trace_filename is not None and step == -1:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
    else:
        run_options = None
        run_metadata = None
    summary_str = None
    start_time = time.time()
    if summary_op is None:
        results = sess.run(fetches, options=run_options,
                           run_metadata=run_metadata)
    else:
        (results, summary_str) = sess.run(
            [fetches, summary_op], options=run_options,
            run_metadata=run_metadata)

    if not FLAGS.forward_only:
        lossval = results[1]
    else:
        lossval = 0.

    train_time = time.time() - start_time
    step_train_times.append(train_time)
    if step >= 0 and (step == 0 or (step + 1) % FLAGS.display_every == 0):
        log_fn('%i\t%s\t%.3f' % (
            step + 1, get_perf_timing_str(batch_size, step_train_times),
            lossval))
    if trace_filename is not None and step == -1:
        log_fn('Dumping trace to', trace_filename)
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        with open(trace_filename, 'w') as trace_file:
            trace_file.write(
                trace.generate_chrome_trace_format(show_memory=True))
    return summary_str


def get_perf_timing_str(batch_size, step_train_times, scale=1):
    times = np.array(step_train_times)
    speeds = batch_size / times
    speed_mean = scale * batch_size / np.mean(times)
    if scale == 1:
        speed_uncertainty = np.std(speeds) / np.sqrt(float(len(speeds)))
        speed_madstd = 1.4826 * np.median(np.abs(speeds - np.median(speeds)))
        speed_jitter = speed_madstd
        return 'images/sec: {:.1f} +/- {:.1f} (jitter = {:.1f})'.format(
            speed_mean, speed_uncertainty, speed_jitter)
    else:
        return 'images/sec: {:.1f}'.format(speed_mean)

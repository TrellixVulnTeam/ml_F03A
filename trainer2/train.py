from __future__ import print_function

import time

import six
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import data_flow_ops
# from tensorflow.python.client import timeline

from trainer2 import manager
from trainer2 import flags
from trainer2.global_step import GlobalStepWatcher
from trainer2 import datasets
from trainer2 import util
from trainer2.model import Model
from trainer2.supervisor import Supervisor

FLAGS = flags.get_flags()
WORKER_ARRAY = ['worker', 'master']


def log_fn(string):
    util.log_fn(string)


class Trainer(object):
    """Class for model training."""

    def __init__(self, config):
        self.config = config
        self.supervisor = None
        self.model = Model.trial()
        self.dataset = datasets.DatasetFactory().create_dataset(data_dir=FLAGS.data_dir)

        self.trace_filename = FLAGS.trace_file
        self.num_batches = FLAGS.num_batches
        self.data_format = FLAGS.data_format

        autotune_threshold = FLAGS.autotune_threshold if FLAGS.autotune_threshold else 1
        min_autotune_warmup = 5 * autotune_threshold * autotune_threshold
        self.num_warmup_batches = FLAGS.num_warmup_batches if FLAGS.num_warmup_batches else max(10, min_autotune_warmup)

        self.batch_size = self.model.batch_size * self.config.num_gpus
        self.manager = manager.manager_factory(self)
        self.devices = self.manager.get_devices()

    def run(self):
        """Run trainer."""
        if self.config.job_name in ['master', 'worker', '']:
            with tf.Graph().as_default():
                self.train()

    def train(self):

        enqueue_ops, fetches = self.model.build_graph(self.config, self.manager)
        main_fetch_group = tf.group(*fetches)

        execution_barrier = None
        if self.config.job_name in WORKER_ARRAY and not FLAGS.cross_replica_sync:
            execution_barrier = self.config.create_sync_queue('execution_barrier_', [])

        global_step = tf.contrib.framework.get_global_step()
        with tf.device(self.config.global_step_device):
            with tf.control_dependencies([main_fetch_group]):
                inc_global_step = global_step.assign_add(1)
                fetches.append(inc_global_step)

        if self.config.job_name in WORKER_ARRAY and FLAGS.cross_replica_sync:
            # Block all replicas until all replicas are ready for next step.
            fetches.append(self.config.create_sync_queue('sync_queues_step_end_', [main_fetch_group]))

        variable_mgr_post_init_ops = self.manager.get_post_init_ops()
        if variable_mgr_post_init_ops:
            post_init_op_group = tf.group(*variable_mgr_post_init_ops)
        else:
            post_init_op_group = None

        local_var_init_op = tf.local_variables_initializer()
        summary_op = tf.summary.merge_all()

        self.supervisor = Supervisor.init(self.config.is_chief, global_step)

        step_train_times = []
        with self.supervisor.run_session(self.config.server) as sess:

            for i in range(len(enqueue_ops)):
                sess.run(enqueue_ops[:(i + 1)])
            sess.run(local_var_init_op)
            if post_init_op_group:
                sess.run(post_init_op_group)

            init_global_step = 0

            global_step_watcher = GlobalStepWatcher(
                sess=sess, global_step_op=global_step,
                start_at_global_step=len(self.config.worker_tasks) * self.num_warmup_batches + init_global_step,
                end_at_global_step=len(self.config.worker_tasks) * (self.num_warmup_batches + self.num_batches) - 1)
            global_step_watcher.start()

            self.supervisor.write_graph_file(sess, self.config.job_name)

            log_fn('Running warm up')
            local_step = -1 * self.num_warmup_batches

            def should_stop():
                if FLAGS.cross_replica_sync and self.config.job_name in WORKER_ARRAY:
                    # In cross-replica sync mode, all workers must run the same number of local steps
                    return local_step == self.num_batches
                else:
                    return global_step_watcher.done()

            while not should_stop():

                if local_step == 0:
                    log_fn('Done warm up')
                    if execution_barrier:
                        log_fn('Waiting for other replicas to finish warm up')
                        assert global_step_watcher.start_time == 0
                        sess.run([execution_barrier])

                    log_fn('Step\tImg/sec\tloss')
                    assert len(step_train_times) == self.num_warmup_batches
                    step_train_times = []  # reset to ignore warm up batches

                if self.supervisor.summary_writer and (local_step + 1) % self.supervisor.save_steps == 0:
                    fetch_summary = summary_op
                else:
                    fetch_summary = None

                summary_str = train_step(sess, fetches, local_step, self.batch_size, step_train_times,
                                         self.trace_filename, fetch_summary)

                if summary_str is not None and self.config.is_chief:
                    self.supervisor.summary_computed(sess, summary_str)
                local_step += 1

            # Waits for the global step to be done, regardless of done_fn.
            log_fn('{}-{} finished work: '.format(self.config.job_name, self.config.task_index))

            while not global_step_watcher.done():
                time.sleep(1.0)

            images_per_sec = global_step_watcher.steps_per_second() * self.batch_size
            log_fn('-' * 64)
            log_fn('total images/sec: %.2f' % images_per_sec)
            log_fn('-' * 64)

            self.supervisor.save_model(sess, global_step)

            if execution_barrier:
                # Wait for other workers to reach the end, so this worker doesn't go away underneath them.
                sess.run([execution_barrier])
        self.supervisor.stop()

    def print_info(self):
        """Print basic information."""
        log_fn('Task:       %s' % self.config.job_name)
        log_fn('Batch size:  %s global' % self.batch_size)
        log_fn('             %s per device' % (self.batch_size / len(self.devices)))
        log_fn('Devices:     %s' % self.config.raw_devices)
        log_fn('Data format: %s' % self.data_format)
        log_fn('Optimizer:   %s' % FLAGS.optimizer)
        log_fn('Variables:   %s' % FLAGS.variable_update)
        log_fn('Manager:   %s' % self.manager)
        log_fn('SERVER target:   %s' % self.config.server.target)
        log_fn(self.config.worker_tasks)
        log_fn(self.config.worker_prefix)
        if FLAGS.variable_update == 'replicated':
            log_fn('Use NCCL:    %s' % FLAGS.use_nccl)
        if FLAGS.staged_vars:
            log_fn('Staged vars: %s' % FLAGS.staged_vars)
        log_fn('==========')


def train_step(sess, fetches, step, batch_size, step_train_times, trace_filename, summary_op=None):
    """Train one step"""
    if trace_filename is not None and step == -1:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
    else:
        run_options = None
        run_metadata = None

    summary_str = None
    start_time = time.time()

    if summary_op is None:
        results = sess.run(fetches, options=run_options, run_metadata=run_metadata)
    else:
        results, summary_str = sess.run([fetches, summary_op], options=run_options, run_metadata=run_metadata)
    lossval = results[1]

    train_time = time.time() - start_time
    step_train_times.append(train_time)
    if step >= 0 and (step == 0 or (step + 1) % FLAGS.display_every == 0):
        log_fn('%i\t%s\t%.3f' % (step + 1, get_perf_timing_str(batch_size, step_train_times), lossval))
    if trace_filename is not None and step == -1:
        log_fn('Dumping trace to {}'.format(trace_filename))
        # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        # chrome_trace = trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True)
        # with open(trace_filename, 'w') as trace_file:
        #     trace_file.write(chrome_trace)
    return summary_str


def get_perf_timing_str(batch_size, step_train_times, scale=1):
    times = np.array(step_train_times)
    speeds = batch_size / times
    speed_mean = scale * batch_size / np.mean(times)
    if scale == 1:
        speed_uncertainty = np.std(speeds) / np.sqrt(float(len(speeds)))
        speed_madstd = 1.4826 * np.median(np.abs(speeds - np.median(speeds)))
        speed_jitter = speed_madstd
        return 'images/sec: {:.1f} +/- {:.1f} (jitter = {:.1f})'.format(speed_mean, speed_uncertainty, speed_jitter)
    else:
        return 'images/sec: {:.1f}'.format(speed_mean)

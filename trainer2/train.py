from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf

from trainer2 import config
from trainer2 import datasets
from trainer2 import flags
from trainer2 import manager
from trainer2 import util
from trainer2.global_step import GlobalStepWatcher
from trainer2.model import Model
from trainer2.supervisor import Supervisor

FLAGS = flags.get_flags()
WORKER_ARRAY = ['worker', 'master']


class Trainer(object):
    """Main class for model training."""

    def __init__(self, runtime_config=None):
        self.config = runtime_config
        self.supervisor = None
        self.model = Model.trial()
        self.dataset = datasets.DatasetFactory().create_dataset(data_dir=FLAGS.data_dir)

        self.num_batches = FLAGS.num_batches
        self.data_format = FLAGS.data_format

        autotune_threshold = FLAGS.autotune_threshold if FLAGS.autotune_threshold else 1
        min_autotune_warmup = 5 * autotune_threshold * autotune_threshold
        self.num_warmup_batches = FLAGS.num_warmup_batches if FLAGS.num_warmup_batches else max(10, min_autotune_warmup)

        self.batch_size = self.model.batch_size * len(self.config.op_devices)

        if self.config.num_gpus > 0:
            assert self.config.num_gpus == len(self.config.op_devices)

        self.manager = manager.get_manager(FLAGS.manager_type, self.config)
        self.devices = self.manager.get_devices()

    def run(self):
        """Run trainer or eval checkpoints."""

        if self.config.job_name in ['master', 'worker', '']:
            with tf.Graph().as_default():
                if self.model.run_training:
                    self.train()
                else:
                    self.log('INIT EVAL')
                    self._eval_cnn()

    def train(self):
        """Method for training the given model"""

        # Build the graph, and receive the operations to be executed
        enqueue_ops, fetches = self.model.build_graph(self.config, self.manager)
        main_fetch_group = tf.group(*fetches)

        execution_barrier = None
        if self.config.job_name in WORKER_ARRAY and not FLAGS.sync_training:
            # Create the execution barrier if training asynconously
            # (only used immediately before session closes).
            execution_barrier = self.manager.create_sync_queue('execution_barrier_', [])

        # Initiate global step, and ensure [main_fetch_group] is executed before inc_global_step
        # and fetches.append()
        global_step = tf.contrib.framework.get_global_step()
        with tf.device(self.config.global_step_device):
            with tf.control_dependencies([main_fetch_group]):
                inc_global_step = global_step.assign_add(1)
                fetches.append(inc_global_step)

        if self.config.job_name in WORKER_ARRAY and FLAGS.sync_training:
            # If training in cross replica sync mode: Create barrier to block all replicas until
            # all replicas are ready for next step.
            replica_sync_queue = self.manager.create_sync_queue('sync_queues_step_end_', [main_fetch_group])
            fetches.append(replica_sync_queue)

        # Get ops from manager to execute after variables are initiated. NOTE: The ops are grouped in
        # the get_post_init_ops()-return statement (instead of here).
        post_init_op_group = self.manager.get_post_init_ops()

        # Initialize local variables and create summary operator (by merging all previously created
        # summary actions into one global op).
        local_init = tf.local_variables_initializer()
        summary_op = tf.summary.merge_all()

        self.supervisor = Supervisor.init(self.config.is_chief, global_step)

        step_train_times = []
        with self.supervisor.run_session(self.config.server) as sess:

            for i in range(len(enqueue_ops)):
                sess.run(enqueue_ops[:(i + 1)])

            sess.run(local_init)

            if post_init_op_group:
                sess.run(post_init_op_group)

            init_global_step = 0

            global_step_watcher = GlobalStepWatcher(
                sess=sess, global_step_op=global_step,
                start_at_global_step=len(self.config.worker_tasks) * self.num_warmup_batches + init_global_step,
                end_at_global_step=len(self.config.worker_tasks) * (self.num_warmup_batches + self.num_batches) - 1)
            global_step_watcher.start()

            self.supervisor.write_graph_file(sess, self.config.job_name)

            self.log('Running warm up', 2)
            local_step = -1 * self.num_warmup_batches

            def should_stop():
                # if FLAGS.cross_replica_sync and self.config.job_name in WORKER_ARRAY:
                if self.manager.sync_training:
                    # In cross-replica sync mode, all workers must run the same number of local steps
                    return local_step == self.num_batches
                else:
                    return global_step_watcher.done()

            while not should_stop():

                if local_step == 0:
                    self.log('Done warm up', 3)

                    if execution_barrier:
                        self.log('Waiting for other replicas to finish warm up', 3)
                        assert global_step_watcher.start_time == 0
                        sess.run([execution_barrier])

                    assert len(step_train_times) == self.num_warmup_batches
                    step_train_times = []  # reset to ignore warm up batches

                if self.supervisor.summary_writer and (local_step + 1) % self.supervisor.save_steps == 0:
                    fetch_summary = summary_op
                else:
                    fetch_summary = None

                summary_str = self.train_step(sess, fetches, local_step, self.batch_size, step_train_times,
                                              fetch_summary)

                if summary_str is not None and self.config.is_chief:
                    self.supervisor.summary_computed(sess, summary_str)
                local_step += 1

            # Waits for the global step to be done, regardless of done_fn.
            self.log('{}-{} finished work: '.format(self.config.job_name, self.config.task_index), 5)
            # self.supervisor.save_trace()

            while not global_step_watcher.done():
                time.sleep(0.2)

            images_per_sec = global_step_watcher.steps_per_second() * self.batch_size
            self.log('-' * 64)
            self.log('total images/sec: %.2f' % images_per_sec)
            self.log('-' * 64)

            self.supervisor.save_model(sess, global_step)

            if execution_barrier:
                # Wait for other workers to reach the end, so this worker doesn't go away underneath them.
                sess.run([execution_barrier])
        self.supervisor.stop()

    def _eval_cnn(self):
        """Evaluate the model from a checkpoint using validation dataset."""

        if FLAGS.train_dir is None:
            raise ValueError('Trained model directory not specified')

        eval_dir = os.path.join(FLAGS.train_dir, 'eval')
        self.log('eval_dir: {}'.format(eval_dir))

        enqueue_ops, fetches = self.model.build_graph(self.config, self.manager)
        saver = tf.train.Saver(tf.global_variables())
        summary_writer = tf.summary.FileWriter(eval_dir, tf.get_default_graph())

        with tf.Session(target='', config=config.create_config_proto()) as sess:
            for i in range(len(enqueue_ops)):
                sess.run(enqueue_ops[:(i + 1)])

            self.log('RUNNING EVAL')

            global_step = load_checkpoint(saver, sess, FLAGS.train_dir)

            self.log('Checkpoint loaded.')

            start_time = time.time()
            count_top_1 = 0.0
            count_top_5 = 0.0
            total_eval_count = self.num_batches * self.batch_size

            for step in range(self.num_batches):
                results = sess.run(fetches)
                count_top_1 += results[0]
                count_top_5 += results[1]
                if (step + 1) % 10 == 0:
                    duration = time.time() - start_time
                    ex_per_sec = self.batch_size * self.num_batches / duration
                    self.log('%i\t%.1f examples/sec' % (step + 1, ex_per_sec))
                    start_time = time.time()
            precision_at_1 = count_top_1 / total_eval_count
            recall_at_5 = count_top_5 / total_eval_count
            summary = tf.Summary()
            summary.value.add(tag='eval/Accuracy@1', simple_value=precision_at_1)
            summary.value.add(tag='eval/Recall@5', simple_value=recall_at_5)
            summary_writer.add_summary(summary, global_step)
            self.log('Precision @ 1 = %.4f recall @ 5 = %.4f [%d examples]' %
                     (precision_at_1, recall_at_5, total_eval_count))

    def eval(self):
        """Evaluate the model from a checkpoint using validation dataset."""

        if FLAGS.train_dir is None:
            raise ValueError('Trained model directory not specified')

        eval_dir = os.path.join(FLAGS.train_dir, 'eval')
        self.log('eval_dir: {}'.format(eval_dir))

        enqueue_ops, fetches = self.model.build_graph(self.config, self.manager)
        saver = tf.train.Saver(tf.global_variables())
        summary_writer = tf.summary.FileWriter(eval_dir, tf.get_default_graph())

        with tf.Session(target='', config=config.create_config_proto()) as sess:
            for i in range(len(enqueue_ops)):
                sess.run(enqueue_ops[:(i + 1)])

            self.log('RUNNING EVAL')

            global_step = load_checkpoint(saver, sess, FLAGS.train_dir)

            self.log('Checkpoint loaded.')

            start_time = time.time()
            count_top_1 = 0.0
            count_top_5 = 0.0
            total_eval_count = self.num_batches * self.batch_size

            for step in range(self.num_batches):
                results = sess.run(fetches)
                count_top_1 += results[0]
                count_top_5 += results[1]
                if (step + 1) % 10 == 0:
                    duration = time.time() - start_time
                    ex_per_sec = self.batch_size * self.num_batches / duration
                    self.log('%i\t%.1f examples/sec' % (step + 1, ex_per_sec))
                    start_time = time.time()
            precision_at_1 = count_top_1 / total_eval_count
            recall_at_5 = count_top_5 / total_eval_count
            summary = tf.Summary()
            summary.value.add(tag='eval/Accuracy@1', simple_value=precision_at_1)
            summary.value.add(tag='eval/Recall@5', simple_value=recall_at_5)
            summary_writer.add_summary(summary, global_step)
            self.log('Precision @ 1 = %.4f recall @ 5 = %.4f [%d examples]' %
                     (precision_at_1, recall_at_5, total_eval_count))

    def log(self, string, debug_level=1):
        if FLAGS.debug_level > 3:
            string = '{}-{}: {}'.format(self.config.job_name, self.config.task_index, string)
        util.log_fn(string, debug_level)

    def train_step(self, sess, fetches, step, batch_size, step_train_times, summary_op=None, trace_filename=None):
        """
        Method for training/(advancing) one step

        :param sess: Session object for the training
        :param fetches: Fetches operations
        :param int step: Current local step
        :param batch_size: Training batch size
        :param step_train_times: Historic step train times
        :param trace_filename: Filename for output profiling-file
        :param summary_op: Summary operation to use in step
        :return: Summary string from training step
        """
        run_trace = step % 1000 == 0 and trace_filename
        if run_trace:
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

        if step >= 0:
            self.progress_string(step+1, batch_size, step_train_times, lossval)

        if run_trace:
            self.supervisor.write_profiling_timeline(run_metadata)

        return summary_str

    def progress_string(self, step, batch_size, step_times, loss):
        if step % FLAGS.display_every == 0:
            times = np.array(step_times)
            speeds = batch_size / times
            speed_mean = batch_size / np.mean(times)
            speed_uncertainty = np.std(speeds) / np.sqrt(float(len(speeds)))
            self.log('Local step: {} | steps/s: {:.1f} +/- {:.1f} | Loss: {:.3f}'.format(step, speed_mean,
                                                                                         speed_uncertainty, loss))


def load_checkpoint(saver, sess, ckpt_dir):
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/imagenet_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        if not global_step.isdigit():
            global_step = 0
        else:
            global_step = int(global_step)
        saver.restore(sess, ckpt.model_checkpoint_path)
        util.log_fn('Successfully loaded model from %s.' % ckpt.model_checkpoint_path)
        return global_step
    else:
        raise RuntimeError('No checkpoint file found.')

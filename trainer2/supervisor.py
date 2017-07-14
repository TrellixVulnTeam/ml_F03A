#
# supervisor
# ml
#

import json
import os

import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.platform import gfile

from trainer2 import flags
from trainer2.config import create_config_proto

FLAGS = flags.get_flags()


class Supervisor(tf.train.Supervisor):

    def __init__(self, train_dir, write_summary, save_steps, *args, **kwargs):
        """
        Helper class for training supervisor (inherieted from tf.train.Supervisor)

        :param str train_dir: directory path for saving model and summary files
        :param bool write_summary: Boolean if supervisor should write summaries
        :param int save_steps: Interval between summary saves
        :param args: Additional positional arguments
        :param kwargs: Additional keyword arguments
        """
        assert train_dir is not None, 'Invalid train dir path.'
        super(Supervisor, self).__init__(*args, **kwargs)
        self.write_summary = write_summary
        self.train_dir = train_dir
        self.save_steps = save_steps
        self.graph_file = FLAGS.graph_file
        self.timeline_file = FLAGS.trace_file
        self.timeline_dict = None

    def run_session(self, server):
        """
        Return context manager for training session.

        :param server:
        :type server str or tf.train.Server
        :return: A context manager that yields a `Session` restored from the latest checkpoint or initialized from
                 scratch if not checkpoint exists.
        """
        try:
            target = server.target
        except AttributeError:
            target = ''

        return self.managed_session(
            master=target,
            start_standard_services=self.write_summary,
            config=create_config_proto())

    def write_graph_file(self, sess, job_name):
        if self.graph_file is not None and self.is_chief:
            path, filename = os.path.split(self.graph_file)
            as_text = filename.endswith('txt')
            # log_fn('Writing GraphDef as %s to %s' % ('text' if as_text else 'binary', self.graph_file))
            tf.train.write_graph(sess.graph_def, path, '{}_{}'.format(job_name, filename), as_text)

    def write_profiling_timeline(self, run_metadata):

        assert self.timeline_file is not None

        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        chrome_trace = trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True)
        ct_dict = json.loads(chrome_trace)

        if self.timeline_dict is None:
            self.timeline_dict = ct_dict
        else:
            for event in ct_dict['traceEvents']:
                if 'ts' in event:
                    self.timeline_dict['traceEvents'].append(event)

    def save_trace(self):
        timeline_path = os.path.join(self.train_dir, self.timeline_file)
        with open(timeline_path, 'w') as f:
            json.dump(self.timeline_dict, f)

    def save_model(self, sess, global_step):
        """
        Save the model checkpoint.
        """
        if self.train_dir is not None and self.is_chief:
            checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
            if not gfile.Exists(self.train_dir):
                gfile.MakeDirs(self.train_dir)
            self.saver.save(sess, checkpoint_path, global_step)

    @classmethod
    def init(cls, is_chief, global_step):
        """
        Factory method for creating Supervisor with default arguments from FLAGS.
        :param is_chief:
        :param global_step:
        :return:
        """
        if is_chief and FLAGS.write_summary and FLAGS.train_dir and FLAGS.save_summaries_steps > 0:
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir, tf.get_default_graph())
        else:
            summary_writer = None

        return cls(
            train_dir=FLAGS.train_dir,
            write_summary=FLAGS.write_summary,
            save_steps=FLAGS.save_summaries_steps,
            is_chief=is_chief,
            logdir=FLAGS.train_dir,
            saver=tf.train.Saver(tf.global_variables()),
            global_step=global_step,
            summary_op=None,
            save_model_secs=FLAGS.save_model_secs,
            summary_writer=summary_writer
        )

    def __repr__(self):
        return '{self.__class__.__name__}'.format(self=self)

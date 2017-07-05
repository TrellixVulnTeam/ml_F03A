#
# supervisor
# ml

import tensorflow as tf

from trainer2 import flags
from trainer2.util import create_config_proto
FLAGS = flags.get_flags()


class Supervisor(tf.train.Supervisor):
    """
    Helper class for training supervisor (inherieted from tf.train.Supervisor)
    """
    def __init__(self, train_dir, write_summary, save_steps, *args, **kwargs):
        super(Supervisor, self).__init__(*args, **kwargs)
        self.write_summary = write_summary
        self.train_dir = train_dir
        self.save_steps = save_steps

    def run_session(self, server):
        target = server.target or ''
        return self.managed_session(
            master=target,
            start_standard_services=self.write_summary,
            config=create_config_proto())

    @classmethod
    def init(cls, is_chief, global_step):

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

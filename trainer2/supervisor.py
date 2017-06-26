#
# supervisor
# ml

"""

"""
import tensorflow as tf


class Supervisor(tf.train.Supervisor):
    def __init__(self, *args, **kwargs):
        super(Supervisor, self).__init__(*args, **kwargs)

    def run_session(self):
        print(self.__class__.__name__)
        # with self.managed_session() as sess:

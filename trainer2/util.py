import tensorflow as tf

from trainer2 import flags

FLAGS = flags.get_flags()


def log_fn(string, debug_level=2):
    assert isinstance(debug_level, int), 'Debug level should be of type int, not {}.'.format(type(debug_level))
    if FLAGS.debug_level >= debug_level:
        tf.logging.info(string)

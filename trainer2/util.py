import tensorflow as tf

from trainer2 import flags

FLAGS = flags.get_flags()


def log_fn(string, debug_level=2):
    if FLAGS.debug_level >= debug_level:
        tf.logging.info(string)

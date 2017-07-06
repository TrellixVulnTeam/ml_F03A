import tensorflow as tf

try:
    from trainer2 import flags
except ImportError:
    import flags

FLAGS = flags.get_flags()


def log_fn(string, debug_level=2):
    if FLAGS.debug_level >= debug_level:
        tf.logging.info(string)

import tensorflow as tf

from trainer2 import flags
FLAGS = flags.get_flags()


def create_config_proto():
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.intra_op_parallelism_threads = FLAGS.num_intra_threads
    config.inter_op_parallelism_threads = FLAGS.num_inter_threads
    # config.gpu_options.force_gpu_compatible = FLAGS.force_gpu_compatible
    # config.gpu_options.force_gpu_compatible = True
    return config


def log_fn(string):
    tf.logging.info(string)
    # print(string)

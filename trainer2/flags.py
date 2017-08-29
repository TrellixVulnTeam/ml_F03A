import tensorflow as tf

#############
# EXEC MODE #
#############

tf.flags.DEFINE_boolean(
    'run_training', True,
    'WhetherRun training or evaluation.')

tf.flags.DEFINE_integer(
    'debug_level', 1,
    'Level of logging and summary output. [0 => 5]. 0 For minimum output, 5 for max.')

################
# DIRECTORIES. #
################

tf.flags.DEFINE_string(
    'data_dir', None,
    # 'data_dir', '../data/data/train',
    'Path to dataset in TFRecord format. If not specified, synthetic data will be used.')

tf.flags.DEFINE_string('train_dir', 'train_data', 'Path to session checkpoints.')

tf.flags.DEFINE_string('eval_dir', 'eval', 'Directory where to write eval event logs.')

tf.flags.DEFINE_string('pretrain_dir', None, 'Path to pretrained session checkpoints.')
# TUTORIAL: https://medium.com/towards-data-science/howto-profile-tensorflow-1a49fb18073d
tf.flags.DEFINE_string('trace_file', 'trace.json', 'Enable TensorFlow tracing and write trace to this file.')

tf.flags.DEFINE_string(
    'graph_file', None,
    'Write the model\'s graph definition to this file. Defaults to binary format unless filename ends in txt.')


#############
# BATCH     #
#############

tf.flags.DEFINE_integer(
    'batch_size', 64,
    'batch size per compute device')

tf.flags.DEFINE_integer(
    'num_batches', 8000,
    'Number of batches (per worker) to run.')

tf.flags.DEFINE_integer(
    'num_warmup_batches', None,
    'number of batches to run before timing')

tf.flags.DEFINE_integer(
    'autotune_threshold', None,
    'The autotune threshold for the models')

#############
# DEVICES   #
#############

tf.flags.DEFINE_string(
    'local_parameter_device', 'cpu',
    'Device to use as local parameter server: cpu or gpu.'
)

tf.flags.DEFINE_string(
    'device', 'gpu',
    'Device to use for computation: cpu or gpu'
)

#############
# TRAINING CONFIG
#############

tf.flags.DEFINE_string(
    'activation', 'relu',
    'Optimizer to use: relu or elu')

tf.flags.DEFINE_float(
    'learning_rate', 0.0027498,
    'Initial learning rate for training.')

tf.flags.DEFINE_string(
    'optimizer', 'adam',
    'Optimizer to use: momentum, adam, sgd or rmsprop')

tf.flags.DEFINE_float(
    'num_epochs_per_decay', 0,
    'Number of epochs per learning rate decay (using staircase version)')

tf.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.98999,
    'The exponential learning rate decay factor.')

tf.flags.DEFINE_float(
    'momentum', 0.9,
    'Momentum coefficient for training (with momentum or rmsprop).')

tf.flags.DEFINE_float(
    'rmsprop_decay', 0.9,
    'Decay term for RMSProp.')

tf.flags.DEFINE_float(
    'gradient_clip', None,
    'Gradient clipping magnitude. Disabled by default.')

tf.flags.DEFINE_float(
    'weight_decay', 0.00004,
    'Weight decay factor for training.')

tf.flags.DEFINE_float(
    'epsilon', 1e-8,
    'Weight decay factor for training.')

#############
# OUTPUT / SUMMARIES
#############

tf.flags.DEFINE_integer(
    'display_every', 100,
    'Number of local steps after which progress is printed out')

tf.flags.DEFINE_bool(
    'write_summary', True,
    'Verbosity level for summary ops. Pass 0 to disable both summaries and checkpoints.')

tf.flags.DEFINE_integer(
    'save_summaries_steps', 100,
    'How often to save summaries for trained models. Pass 0 to disable summaries.')

tf.flags.DEFINE_integer(
    'save_model_secs', 600,
    'How often to save trained models. Pass 0 to disable checkpoints')

#############
# TRAINING DATA
#############

tf.flags.DEFINE_string('data_format', 'NHWC', 'Data layout to use: NHWC (TF native) or NCHW (cuDNN native).')

# PREPROCESSING

tf.flags.DEFINE_string(
    'resize_method', 'bilinear',
    'Method for resizing input images: crop,nearest,bilinear,bicubic or area.'
    'The crop mode requires source images to be at least as large as the '
    'network input size, while the other modes support any sizes and apply'
    ' random bbox distortions before resizing (even with --nodistortions).'
)

tf.flags.DEFINE_boolean(
    'distortions', True,
    'Enable/disable distortions during image preprocessing. These  include bbox and color distortions.')

# Performance tuning flags.

tf.flags.DEFINE_boolean('winograd_nonfused', True, 'Enable/disable using the Winograd non-fused algorithms.')

tf.flags.DEFINE_boolean('sync_on_finish', False, 'Enable/disable whether the devices are synced after each step.')

# The method for managing variables:
#   parameter_server: variables are stored on a parameter server that holds
#       the master copy of the variable.  In local execution, a local device
#       acts as the parameter server for each variable; in distributed
#       execution, the parameter servers are separate processes in the cluster.
#       For each step, each tower gets a copy of the variables from the
#       parameter server, and sends its gradients to the param server.
#   distributed_replicated: Distributed training only. Each GPU has a copy of
#       the variables, and updates its copy after the parameter servers are all
#       updated with the gradients from all servers. Only works with
#       cross_replica_sync=true.
tf.flags.DEFINE_string(
    'manager_type', 'ps',
    'The method for managing variables: ps, local or dr'
)

tf.flags.DEFINE_boolean('use_nccl', True, 'Whether to use nccl all-reduce primitives where possible')

# Use 'grpc' normally, and 'grpc+mpi' on ARCHER
tf.flags.DEFINE_string('server_protocol', 'grpc', 'protocol for servers')

tf.flags.DEFINE_boolean('sync_training', True, 'Enable/disable devices syncronisation after each step.')


def get_flags():
    return tf.flags.FLAGS

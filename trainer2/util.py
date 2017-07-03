import os
import json

import tensorflow as tf

from trainer2 import flags
FLAGS = flags.get_flags()
PS_DEVICE_STR = '/{}:0'.format(FLAGS.local_parameter_device)

# INTER/INTRA threads: System will set approperiate number of threads for OP_PARALLELISM_THREADS = 0
OP_PARALLELISM_THREADS = 0


class Config(object):
    def __init__(self,
                 job_name, task_index, is_chief, ps_tasks, worker_tasks, cluster=None, server=None, worker_prefix='',
                 ps_device=PS_DEVICE_STR, sync_queue_devices=None
                 ):
        self.job_name = job_name
        self.task_index = task_index
        self.is_chief = is_chief
        self.ps_tasks = ps_tasks
        self.worker_tasks = worker_tasks
        self.cluster = cluster
        self.server = server
        self.worker_prefix = worker_prefix
        self.ps_device = ps_device
        self.sync_queue_devices = sync_queue_devices
        self.local_parameter_device = FLAGS.local_parameter_device


def create_config_proto():
    config = tf.ConfigProto()
    # config.allow_soft_placement = True
    # config.log_device_placement = True
    config.intra_op_parallelism_threads = OP_PARALLELISM_THREADS
    config.inter_op_parallelism_threads = OP_PARALLELISM_THREADS
    config.gpu_options.force_gpu_compatible = True
    return config


def get_config():
    if FLAGS.winograd_nonfused:
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    else:
        os.environ.pop('TF_ENABLE_WINOGRAD_NONFUSED', None)
    if FLAGS.autotune_threshold:
        os.environ['TF_AUTOTUNE_THRESHOLD'] = str(FLAGS.autotune_threshold)
    os.environ['TF_SYNC_ON_FINISH'] = str(int(FLAGS.sync_on_finish))

    # env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    #
    # # Print the job data as provided by the service.
    # tf.logging.info('Original job data: %s', env.get('job', {}))
    # task_data = env.get('task', None) or {'type': 'master', 'index': 0}
    #
    # cluster_data = env.get('cluster', None)
    # cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
    # ps_tasks = cluster_data.get('ps') if cluster_data else ['']
    # worker_tasks = cluster_data.get('worker') if cluster_data else ['']
    # server = tf.train.Server(
    #     server_or_cluster_def=cluster,
    #     job_name=task_data.type,
    #     task_index=task_data.index,
    #     config=create_config_proto(),
    #     protocol=FLAGS.server_protocol
    # ) if cluster else None
    #
    #
    # return Config(
    #     job_name=task_data.type,
    #     task_index=task_data.index,
    #     is_chief=task_data.type == 'master',
    #     ps_tasks=ps_tasks,
    #     worker_tasks=worker_tasks,
    #     cluster=cluster,
    #     server=server,
    #
    # )

    tf_config = os.environ.get('TF_CONFIG')

    # If TF_CONFIG is not available run local
    if not tf_config:
        # return Config(job_name='', task_index=0, is_chief=True, ps_tasks=[''], worker_tasks=[''],
        #               sync_queue_devices=[PS_DEVICE_STR])
        raise ValueError('Runtime is missing TF_CONFIG')

    tf_config_json = json.loads(tf_config)

    cluster = tf_config_json.get('cluster')
    job_name = tf_config_json.get('task', {}).get('type')
    task_index = tf_config_json.get('task', {}).get('index')

    ps_tasks = cluster.get('ps')
    worker_tasks = cluster.get('worker') + cluster.get('master')

    # For distributed mode only
    assert ps_tasks is not None
    assert worker_tasks is not None

    # If cluster information is empty run local
    if job_name is None or task_index is None or ps_tasks is None or worker_tasks is None:
        # return Config(job_name='', task_index=0, is_chief=True, ps_tasks=[''], worker_tasks=[''],
        #               sync_queue_devices=[PS_DEVICE_STR])
        raise ValueError('Runtime is missing TF_CONFIG')
    else:
        cluster_spec = tf.train.ClusterSpec(cluster)
        server_spec = tf.train.Server(
            server_or_cluster_def=cluster_spec,
            job_name=job_name,
            task_index=task_index,
            config=create_config_proto(),
            protocol=FLAGS.server_protocol
        )
        worker_prefix = '/job:{}/task:{}'.format(job_name, task_index)
        ps_device = tf.train.replica_device_setter(worker_device=worker_prefix + '/cpu:0', cluster=cluster_spec)
        sync_queue_devices = ['/job:ps/task:%s/cpu:0' % i for i in range(len(ps_tasks))]
        return Config(
            job_name=job_name,
            task_index=task_index,
            # is_chief=(job_name == 'worker' and task_index == 0),
            is_chief=(job_name == 'master'),
            ps_tasks=ps_tasks,
            worker_tasks=worker_tasks,
            cluster=cluster_spec,
            server=server_spec,
            worker_prefix=worker_prefix,
            ps_device=ps_device,
            sync_queue_devices=sync_queue_devices
        )


def log_fn(string):
    tf.logging.info(string)
    # print(string)

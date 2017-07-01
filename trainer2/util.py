import os
import json

import tensorflow as tf

from trainer2 import flags
FLAGS = flags.get_flags()
PS_DEVICE_STR = '/{}:0'.format(FLAGS.local_parameter_device)


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


def create_config_proto():
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.intra_op_parallelism_threads = FLAGS.num_intra_threads
    config.inter_op_parallelism_threads = FLAGS.num_inter_threads
    # config.gpu_options.force_gpu_compatible = FLAGS.force_gpu_compatible
    # config.gpu_options.force_gpu_compatible = True
    return config


def get_config():
    tf_config = os.environ.get('TF_CONFIG')

    # If TF_CONFIG is not available run local
    if not tf_config:
        return Config(job_name='', task_index=0, is_chief=True, ps_tasks=[''], worker_tasks=[''], sync_queue_devices=[PS_DEVICE_STR])

    tf_config_json = json.loads(tf_config)

    cluster = tf_config_json.get('cluster')
    job_name = tf_config_json.get('task', {}).get('type')
    task_index = tf_config_json.get('task', {}).get('index')

    # If cluster information is empty run local
    if job_name is None or task_index is None:
        return Config(job_name='', task_index=0, ps_tasks=[''], worker_tasks=[''])
    else:
        ps_tasks = cluster.get('ps') or ['']
        worker_tasks = cluster.get('worker') or ['']
        cluster_spec = tf.train.ClusterSpec(cluster)
        server_spec = tf.train.Server(
            server_or_cluster_def=cluster_spec,
            job_name=job_name,
            task_index=task_index,
            config=create_config_proto(),
            protocol=FLAGS.server_protocol
        )
        worker_prefix = '/job:worker/task:{}'.format(task_index)
        ps_device = tf.train.replica_device_setter(worker_device=worker_prefix + '/cpu:0', cluster=cluster_spec)
        sync_queue_devices = ['/job:ps/task:%s/cpu:0' % i for i in range(len(ps_tasks))]
        return Config(
            job_name=job_name,
            task_index=task_index,
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

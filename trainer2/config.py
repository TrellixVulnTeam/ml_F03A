#
# config
# ml
#
import json
import os

import tensorflow as tf

try:
    from trainer2 import flags
except ImportError:
    import flags

FLAGS = flags.get_flags()
PS_DEVICE_STR = '/{}:0'.format(FLAGS.local_parameter_device)

# INTER/INTRA threads: System will set approperiate number of threads for OP_PARALLELISM_THREADS = 0
OP_PARALLELISM_THREADS = 0


class Config(object):
    def __init__(self, job_name, task_index, is_chief, ps_tasks=None, worker_tasks=None, cluster=None, server=None,
                 worker_prefix='', ps_device=PS_DEVICE_STR, sync_queue_devices=None, num_cpus=1, num_gpus=0):
        """

        :param str job_name: Name of job ('worker', 'master' or 'ps' / '' for local)
        :param int task_index: Task index
        :param bool is_chief: Is current task chief
        :param ps_tasks:
        :param worker_tasks:
        :param cluster:
        :param server:
        :param worker_prefix:
        :param ps_device:
        :param sync_queue_devices:
        """
        if is_chief:
            assert task_index == 0, 'Invalid master configuration.'
            assert job_name in ['master', '']

        assert job_name in ['ps', 'worker', 'master', ''], 'Invalid job name'
        if job_name in ['ps', 'worker']:
            assert isinstance(cluster, tf.train.ClusterSpec), 'Invalid ClusterSpec for distributed training.'
            assert isinstance(server, tf.train.Server), 'Invalid Server instance for distributed training.'

        assert task_index >= 0, 'Task index cannot be negative.'

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
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus

        assert num_gpus + num_cpus > 0

        self.cpu_device = '%s/cpu:0' % worker_prefix
        self.raw_devices = ['%s/%s:%i' % (worker_prefix, FLAGS.device, i)
                            for i in range(num_gpus if num_gpus > 0 else 1)]

        assert len(self.raw_devices) > 0

        self.global_step_device = ps_device if job_name else self.cpu_device
        self.sync_queue_counter = 0

    def create_sync_queue(self, name_prefix, enqueue_after):
        """
        Adds ops to enqueue on all worker queues.
        :param str name_prefix: prefixed for the shared_name of ops.
        :param list enqueue_after: control dependency from ops.
        :return: an op that should be used as control dependency before starting next step.
        """
        assert isinstance(self.cluster, tf.train.ClusterSpec), 'The needed clusterSpec is not provided.'
        assert self.job_name != '', 'Illegal job name/type for synch queues and barriers.'

        self.sync_queue_counter += 1
        # Handle case where master is only worker
        num_workers = self.cluster.num_tasks('worker') + 1
        # try:
        #     num_workers = self.cluster.num_tasks('worker') + 1
        # except ValueError:
        #     num_workers = 1

        with tf.device(self.sync_queue_devices[self.sync_queue_counter % len(self.sync_queue_devices)]):
            sync_queues = [
                tf.FIFOQueue(num_workers, [tf.bool], shapes=[[]], shared_name='%s%s' % (name_prefix, i))
                for i in range(num_workers)]
            queue_ops = []

            # For each other worker, add an entry in a queue, signaling that it can finish this step.
            token = tf.constant(False)
            with tf.control_dependencies(enqueue_after):
                for i, q in enumerate(sync_queues):
                    if self.job_name == 'master' and i == 0:
                        queue_ops.append(tf.no_op())
                    elif self.job_name == 'worker' and i == self.task_index + 1:
                        queue_ops.append(tf.no_op())
                    else:
                        queue_ops.append(q.enqueue(token))

            # Drain tokens off queue for this worker, one for each other worker.
            temp_index = self.task_index if self.job_name == 'master' else self.task_index + 1
            queue_ops.append(sync_queues[temp_index].dequeue_many(len(sync_queues) - 1))

        return tf.group(*queue_ops)

    @classmethod
    def local_config(cls):
        """
        Convenience method for creating local config object.
        """
        return cls(job_name='', task_index=0, is_chief=True, ps_tasks=[''], worker_tasks=[''],
                   sync_queue_devices=[PS_DEVICE_STR])

    @classmethod
    def ps_server(cls, task_index, cluster, server):
        """
        Convenience method for creating ps server config object.
        """
        assert isinstance(server, tf.train.Server), 'Config server must be of type tf.train.Server.'
        return cls(job_name='ps', task_index=task_index, is_chief=False,
                   cluster=cluster, server=server)


def create_config_proto():
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    # config.log_device_placement = True
    config.intra_op_parallelism_threads = OP_PARALLELISM_THREADS
    config.inter_op_parallelism_threads = OP_PARALLELISM_THREADS
    try:
        # Only avilable for tf.version > 1.2.0
        config.gpu_options.force_gpu_compatible = True
    except AttributeError:
        pass

    return config


def get_cloud_ml_device_count(job_data, job_name):
    # 'name': (cpus, gpus)
    ml_macines = {
        'standard': (1, 0),
        'standard_gpu': (1, 1),
        'complex_model_m_gpu': (1, 4)
    }

    machine_type = job_data['{}_type'.format(job_name if job_name in ['master', 'worker'] else 'parameter_server')]
    try:
        return ml_macines[machine_type]
    except KeyError:
        tf.logging.error('Unknown machine instance {} - assuming single CPU.'.format(machine_type))
        return 1, 0


def config_factory():
    load_env_flags()

    env = json.loads(os.environ.get('TF_CONFIG', '{}'))

    # Get cluster and current task info:
    job_data = env.get('job', None)
    cluster_data = env.get('cluster', None)
    task_data = env.get('task', None)

    FLAGS.train_dir = make_trial_path(task_data, FLAGS.train_dir)

    # Print the job data as provided by the service.
    # tf.logging.info('Original job data: %s', job_data)
    try:
        ps_tasks = cluster_data.get('ps')
        worker_tasks = cluster_data.get('worker', []) + cluster_data.get('master')
        job_name = task_data.get('type')
        task_index = task_data.get('index')
        num_cpus, num_gpus = get_cloud_ml_device_count(job_data, job_name)
    except AttributeError:
        return Config.local_config()
    else:
        if cluster_data.get('master') and not ps_tasks:
            return Config.local_config()

    # Create cluster spec and start server:
    cluster = tf.train.ClusterSpec(cluster_data)
    server = tf.train.Server(cluster, job_name, task_index, FLAGS.server_protocol, create_config_proto())

    if job_name == 'ps':  # Return config for parameter server if applicable
        return Config.ps_server(task_index=task_index, cluster=cluster, server=server)

    # Set device strings for training / variables / ops placement
    worker_prefix = '/job:{}/task:{}'.format(job_name, task_index)
    ps_device = tf.train.replica_device_setter(ps_device='/job:ps', worker_device=worker_prefix, cluster=cluster)
    sync_queue_devices = ['/job:ps/task:%s/cpu:0' % i for i in range(len(ps_tasks))]

    # Is current task Session chief:
    # is_chief = job_name == 'worker' and task_index == 0
    is_chief = job_name == 'master'

    return Config(job_name, task_index, is_chief, ps_tasks, worker_tasks, cluster, server, worker_prefix, ps_device,
                  sync_queue_devices, num_cpus, num_gpus)


def make_trial_path(task_data, output_path):
    """
    For a given static output path, returns a path with the hyperparameter tuning
    trial number appended: [output_path => output_path/{trial_int}]

    :param task_data: Task data from TF_CONFIG environment variable.
    :param str output_path:
    :return: str New output path
    """
    assert isinstance(output_path, str), 'Output path must be string.'

    if task_data:
        trial = task_data.get('trial', '')
        if trial:
            # assert isinstance(trial, str), 'Trial index-string must be if type str, not {}.'.format(type(trial))
            return os.path.join(output_path, trial)
    return output_path


def load_env_flags():

    assert isinstance(FLAGS.winograd_nonfused, bool)
    assert isinstance(FLAGS.sync_on_finish, bool)
    assert isinstance(FLAGS.autotune_threshold, (int, type(None)))

    if FLAGS.winograd_nonfused:
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    else:
        os.environ.pop('TF_ENABLE_WINOGRAD_NONFUSED', None)

    if FLAGS.autotune_threshold:
        os.environ['TF_AUTOTUNE_THRESHOLD'] = str(FLAGS.autotune_threshold)

    os.environ['TF_SYNC_ON_FINISH'] = str(int(FLAGS.sync_on_finish))

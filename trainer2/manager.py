# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Defines VariableMgr and subclasses used to manage variables.

"""

from __future__ import print_function

import operator

import tensorflow as tf
from tensorflow.contrib import nccl

from trainer2.flags import get_flags

# from tensorflow.python.framework import ops
# from tensorflow.python.ops import data_flow_ops

PS_SHADOW_VAR_PREFIX = 'ps_var'
FLAGS = get_flags()


def get_manager(manager_type, config):

    assert manager_type in ['ps', 'local', 'dr'], 'Provided manager_type is invalid.'

    if manager_type == 'local':
        return LocalManager(config)

    assert config.job_name is not ''
    if manager_type == 'ps':
        return PSManager(config)
    elif manager_type == 'dr':
        return DRManager(config)


# To be used with custom_getter on tf.get_variable.
class OverrideCachingDevice(object):
    def __init__(self, devices, device_for_small_variables, small_variable_size_threshold):
        self.devices = devices
        self.sizes = [0] * len(self.devices)
        self.device_for_small_variables = device_for_small_variables
        self.small_variable_size_threshold = small_variable_size_threshold

    def __call__(self, getter, *args, **kwargs):
        size = tf.TensorShape(kwargs['shape']).num_elements()
        if size < self.small_variable_size_threshold:
            device_name = self.device_for_small_variables
        else:
            device_index, _ = min(enumerate(self.sizes), key=operator.itemgetter(1))
            device_name = self.devices[device_index]
            self.sizes[device_index] += size

        kwargs['caching_device'] = device_name
        var = getter(*args, **kwargs)
        return var


# To be used with custom_getter on tf.get_variable. Ensures the created variable
# is in LOCAL_VARIABLES and not GLOBAL_VARIBLES collection.
class OverrideToLocalVariableIfNotPsVar(object):
    # args and kwargs come from the custom_getter interface for Tensorflow
    # variables, and matches tf.get_variable's signature, with the addition of
    # 'getter' at the beginning.
    def __call__(self, getter, name, *args, **kwargs):
        if name.startswith(PS_SHADOW_VAR_PREFIX):
            return getter(*args, **kwargs)

        if 'collections' in kwargs:
            collections = kwargs['collections']
        if not collections:
            collections = set([tf.GraphKeys.GLOBAL_VARIABLES])
        else:
            collections = set(collections.copy())

        collections.remove(tf.GraphKeys.GLOBAL_VARIABLES)
        collections.add(tf.GraphKeys.LOCAL_VARIABLES)
        kwargs['collections'] = list(collections)
        return getter(name, *args, **kwargs)


class ParamServerDeviceSetter(object):
    """Helper class to assign variables on the least loaded ps-device."""

    def __init__(self, worker_device, ps_devices):
        """Initializer for ParamServerDevicSetter.

        Args:
          worker_device: the device to use for computer ops.
          ps_devices: a list of device to use for Variable ops. Each variable is
          assigned to the least loaded device.
        """
        self.ps_devices = ps_devices
        self.worker_device = worker_device
        self.ps_sizes = [0] * len(self.ps_devices)

    def __call__(self, op):
        if op.device:
            return op.device
        if op.type not in ['Variable', 'VariableV2']:
            return self.worker_device

        device_index, _ = min(enumerate(self.ps_sizes), key=operator.itemgetter(1))
        device_name = self.ps_devices[device_index]
        var_size = op.outputs[0].get_shape().num_elements()
        self.ps_sizes[device_index] += var_size

        return device_name


class VariableMgr(object):
    """Abstract superclass for class used by Trainer to control variables.

      Functions on this class are used to control how variables are created and
      managed, and how gradients are computed and applied.
    """

    def __init__(self, trainer_config):
        self.config = trainer_config
        self.staging_delta_ops = []
        self.devices = self.get_devices()

    def each_tower_has_variables(self):
        """Returns True if each GPU tower of the model has separate variables."""
        assert False, 'Must be implemented in subclass'

    def create_outer_variable_scope(self, device_num):
        """Create the tf.variable_scope around all model graph operations."""
        del device_num  # unused by this implementation
        assert False, 'Must be implemented in subclass'

    def preprocess_device_grads(self, device_grads):
        """Preprocess the device gradients prior to applying them.

        Args:
          device_grads: a list of gradients each of which calculated by a device.

        Returns: a tuple of (apply_gradients_devices, gradient_state), where
          gradients will then be applied to each entry in apply_gradients_devices,
          and gradient is passed to later calls to get_gradients_to_apply and
          append_apply_gradients_ops.
        """
        del device_grads  # unused by this implementation
        assert False, 'Must be implemented in subclass'

    def get_gradients_to_apply(self, device_num, gradient_state):
        """Returns the [(gradient, variable] to apply for device_num.

        Args:
          device_num: indexes ino the apply_gradients_devices returned by an earlier
                      call to preprocess_device_grads.
          gradient_state: from previous call to apply_gradients_devices.
        """
        del device_num, gradient_state  # unused by this implementation
        assert False, 'Must be implemented in subclass'

    def append_apply_gradients_ops(self, gradient_state, opt, grads, training_ops):
        """Adds training ops for grads to 'training_ops'.

        Args:
          gradient_state: from previous call to apply_gradients_devices.
          opt: the underlying optimizer
          grads: [(grad, var)] to apply
          training_ops: list to which to add ops
        """
        del gradient_state  # unused by this implementation
        apply_gradients_op = opt.apply_gradients(grads)
        training_ops.append(apply_gradients_op)

    def retain_tower_updates(self, device_num):
        """Return if only updates for the first GPU tower should be applied."""
        return device_num == 0 and not self.each_tower_has_variables()

    def get_post_init_ops(self):
        """Returns ops that should run post-initialization."""
        return None

    def get_devices(self):
        """Returns devices to use for computation; includes replica selection."""
        assert False, 'Must be implemented in subclass'

    def trainable_variables_on_device(self, device_num, writable=False):
        """Return the set of trainable variables on device.

        Args:
          device_num: the index to the device.
          writable: whether to get a reference to the underlying variable.

        Returns:
          The set of trainable vairalbes on the specified device.
        """
        del writable
        if self.each_tower_has_variables():
            params = [v for v in tf.trainable_variables() if v.name.startswith('v%s/' % device_num)]
        else:
            params = tf.trainable_variables()
        return params

    def __repr__(self):
        return '{self.__class__.__name__}'.format(self=self)


class LocalManager(VariableMgr):
    """VariableMgr that implements the --parameter_server mode for local jobs.

       Variables are stored on a parameter server.  For each step, each tower gets
       a copy of the variables from the parameter server, and sends its gradients to the param server.
    """

    def each_tower_has_variables(self):
        return False

    def create_outer_variable_scope(self, device_num):
        return tf.variable_scope('v', reuse=bool(device_num))

    def preprocess_device_grads(self, device_grads):
        return [self.config.ps_device], device_grads

    def get_gradients_to_apply(self, device_num, gradient_state):
        assert device_num == 0
        device_grads = gradient_state
        return aggregate_gradients_using_copy_with_variable_colocation(
            device_grads, use_mean=True)

    def get_devices(self):
        op_devices = self.config.op_devices
        if self.config.local_parameter_device == 'gpu':
            return [ParamServerDeviceSetter(d, op_devices) for d in op_devices]
        else:
            return [tf.train.replica_device_setter(
                worker_device=d, ps_device=self.config.ps_device,
                ps_tasks=1) for d in op_devices]


class PSManager(VariableMgr):
    """Implements --variable_update=parameter_server mode for distributed jobs.

       Variables are stored on a parameter server.  For each step, each tower gets
       a copy of the variables from the parameter server, and sends its gradients
       to the param server.
    """

    def each_tower_has_variables(self):
        return False

    def create_outer_variable_scope(self, device_num):
        if self.config.local_parameter_device == 'gpu':
            caching_devices = self.config.op_devices
        else:
            caching_devices = [self.config.cpu_device]
        custom_getter = OverrideCachingDevice(caching_devices, self.config.cpu_device, 1024 * 64)
        return tf.variable_scope('v', reuse=bool(device_num), custom_getter=custom_getter)

    def preprocess_device_grads(self, device_grads):
        # Returns (gradient_devices, gradient_state)
        return [self.config.ps_device], device_grads

    def get_gradients_to_apply(self, device_num, gradient_state):
        assert device_num == 0
        return aggregate_gradients_using_copy(gradient_state, use_mean=True)

    def get_devices(self):
        ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(
            len(self.config.ps_tasks), tf.contrib.training.byte_size_load_fn)
        return [tf.train.replica_device_setter(worker_device=d, cluster=self.config.cluster,
                ps_strategy=ps_strategy) for d in self.config.op_devices]


class DRManager(VariableMgr):
    """VariableMgr that implements the --distributed_replicated mode.

       Each GPU has a copy of the variables, and updates its copy after the
       parameter servers are all updated with the gradients from all servers. Only
       works with cross_replica_sync=true. Unlike 'replicated', does not use nccl
       all-reduce for replicating within a server.
    """

    def each_tower_has_variables(self):
        return True

    def create_outer_variable_scope(self, device_num):
        return tf.variable_scope('v%s' % device_num, custom_getter=OverrideToLocalVariableIfNotPsVar())

    def preprocess_device_grads(self, device_grads):
        return [self.config.ps_device], device_grads

    def get_gradients_to_apply(self, device_num, gradient_state):
        device_grads = gradient_state  # From 2nd result of preprocess_device_grads.

        avg_grads = aggregate_gradients_using_copy_with_device_selection(self.config, device_grads, use_mean=True)

        # Make shadow variable for each original trainable variable.
        for i, (g, v) in enumerate(avg_grads):
            my_name = PS_SHADOW_VAR_PREFIX + '/' + v.name
            if my_name.endswith(':0'):
                my_name = my_name[:-2]
            new_v = tf.get_variable(my_name, dtype=v.dtype.base_dtype,
                                    initializer=v.initial_value,
                                    trainable=True)
            avg_grads[i] = (g, new_v)
        return avg_grads

    def append_apply_gradients_ops(self, gradient_state, opt, grads, training_ops):
        device_grads = gradient_state  # From 2nd result of preprocess_device_grads.

        # For each variable, apply the combined gradients for this server on
        # the parameter server, and then wait for all other servers to do
        # this.
        for i, (g, v) in enumerate(grads):
            apply_gradient_op = opt.apply_gradients([(g, v)])
            barrier = self.config.create_sync_queue(
                'replicate_variable_%s' % i, [apply_gradient_op])
            with tf.control_dependencies([barrier]):
                with tf.device(self.config.cpu_device):
                    updated_value = v.read_value()
                    for my_d in range(len(self.config.devices)):
                        training_ops.append(device_grads[my_d][i][1].assign(updated_value))

    def get_post_init_ops(self):
        # Copy initialized variables for variables on the parameter server
        # to the local copy of the variable.
        def strip_port(s):
            if s.endswith(':0'):
                return s[:-2]
            return s

        local_vars = tf.local_variables()
        local_var_by_name = dict([(strip_port(v.name), v) for v in local_vars])
        post_init_ops = []
        for v in tf.global_variables():
            if v.name.startswith(PS_SHADOW_VAR_PREFIX + '/v0/'):
                prefix = strip_port(
                    v.name[len(PS_SHADOW_VAR_PREFIX + '/v0'):])
                for i in range(self.config.num_gpus):
                    name = 'v%s%s' % (i, prefix)
                    if name in local_var_by_name:
                        copy_to = local_var_by_name[name]
                        post_init_ops.append(copy_to.assign(v.read_value()))
        return tf.group(*post_init_ops)

    def get_devices(self):
        return self.config.op_devices


def sum_grad_and_var_all_reduce(grad_and_vars, devices):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))

    scaled_grads = [g for _, (g, _) in zip(devices, grad_and_vars)]
    summed_grads = nccl.all_sum(scaled_grads)

    result = []
    for d, (_, v), g in zip(devices, grad_and_vars, summed_grads):
        with tf.device(d):
            result.append((g, v))
    return result


def sum_gradients_all_reduce(tower_grads, devices):
    new_tower_grads = []
    for grad_and_vars in zip(*tower_grads):
        new_tower_grads.append(sum_grad_and_var_all_reduce(grad_and_vars, devices))
    return list(zip(*new_tower_grads))


def aggregate_gradients_using_copy_with_device_selection(config, tower_grads, use_mean):
    """Aggregate gradients, controlling device for the aggregation.

    Args:
      config: Trainer/Manager config.
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
      use_mean: if True, mean is taken, else sum of gradients is taken.
    Returns:
      List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    if config.local_parameter_device == 'gpu':
        avail_devices = config.op_devices
    else:
        avail_devices = [config.ps_device]
    agg_grads = []
    for i, single_grads in enumerate(zip(*tower_grads)):
        with tf.device(avail_devices[i % len(avail_devices)]):
            agg_grads.extend(aggregate_gradients_using_copy(zip(single_grads), use_mean))
    return agg_grads


def aggregate_gradients_using_copy_with_variable_colocation(tower_grads, use_mean):
    """Aggregate gradients, colocating computation with the gradient's variable.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
      use_mean: if True, mean is taken, else sum of gradients is taken.
    Returns:
      List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    agg_grads = []
    for _, single_grads in enumerate(zip(*tower_grads)):
        var = single_grads[0][1]

        for __, v in single_grads:
            assert v == var

        with tf.device(var.device):
            agg_grads.extend(aggregate_gradients_using_copy(zip(single_grads), use_mean))
    return agg_grads


def aggregate_gradients_using_copy(tower_grads, use_mean):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
      use_mean: if True, mean is taken, else sum of gradients is taken.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    agg_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        # grads = []
        grads = [g for g, _ in grad_and_vars]
        grad = tf.add_n(grads)

        if use_mean and len(grads) > 1:
            grad = tf.multiply(grad, 1.0 / len(grads))

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        agg_grads.append(grad_and_var)
    return agg_grads

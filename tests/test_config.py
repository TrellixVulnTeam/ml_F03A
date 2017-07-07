#
# test_config
# ml
#

"""

"""
import os
import json
from unittest.mock import patch

import tensorflow as tf

from trainer2 import config
from trainer2.flags import get_flags
FLAGS = get_flags()
SAMLPLE_TF_CONFIG = {
    'cluster': {
        'master': [
            'master-e0ca1f7b4d-0:2222'
        ]
    },
    'task': {
        'type': 'master',
        'index': 0
    },
    'job': {
        'scale_tier': 'CUSTOM',
        'master_type': 'standard',
        'worker_type': 'standard_gpu',
        'parameter_server_type': 'standard'
    }
}


class SuperTestConfig(tf.test.TestCase):
    def setUp(self):
        self.config = config.Config(job_name='', task_index=0, is_chief=True, ps_tasks=[''], worker_tasks=[''],
                                    sync_queue_devices=[config.PS_DEVICE_STR])
        os.environ['TF_CONFIG'] = json.dumps(SAMLPLE_TF_CONFIG)

    def tearDown(self):
        del os.environ['TF_CONFIG']
        self.config = None


class TestConfig(SuperTestConfig):

    def test_config(self):
        # and invalid master config
        self.assertRaises(AssertionError, config.Config, 'master', 1, True)
        # Negative task index
        self.assertRaises(AssertionError, config.Config, 'worker', -1, False)
        # invalid job name
        self.assertRaises(AssertionError, config.Config, 'name', 2, False)

    def test_create_sync_queue(self):

        # Illegal arguments
        self.assertRaises(TypeError, self.config.create_sync_queue)

        # Illegal Job Name
        self.assertRaises(AssertionError, self.config.create_sync_queue, 'prefix', [])

        self.config.job_name = 'master'
        self.assertRaises(AssertionError, self.config.create_sync_queue, 'prefix', [])

    def test_local_config(self):
        self.assertEqual(self.config.job_name, config.Config.local_config().job_name)

    def test_config_factory(self):
        tf_config = config.config_factory()
        self.assertEqual(tf_config.num_cpus, 1)
        self.assertEqual(tf_config.num_gpus, 0)
        self.assertEqual(tf_config.job_name, '')

    def test_env_flags(self):
        with patch.dict(os.environ):
            FLAGS.winograd_nonfused = True
            FLAGS.autotune_threshold = 5
            FLAGS.sync_on_finish = False
            config.load_env_flags()
            self.assertEqual(os.environ['TF_ENABLE_WINOGRAD_NONFUSED'], '1')
            self.assertEqual(os.environ['TF_AUTOTUNE_THRESHOLD'], '5')
            self.assertTrue(os.environ['TF_SYNC_ON_FINISH'], '0')

        with patch.dict(os.environ):
            FLAGS.sync_on_finish = 77
            self.assertRaises(AssertionError, config.load_env_flags)

    def test_ps_server(self):
        # Illegal server spec
        self.assertRaises(AssertionError, config.Config.ps_server, 1, None, None)

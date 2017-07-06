#
# test_config
# ml
#

"""

"""
import os
from unittest.mock import patch

import tensorflow as tf

from trainer2 import config
from trainer2.flags import get_flags
FLAGS = get_flags()


class SuperTestConfig(tf.test.TestCase):
    def setUp(self):
        self.config = config.Config(job_name='', task_index=0, is_chief=True, ps_tasks=[''], worker_tasks=[''],
                                    sync_queue_devices=[config.PS_DEVICE_STR])

    def tearDown(self):
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

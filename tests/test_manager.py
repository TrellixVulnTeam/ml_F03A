#
# test_manager
# ml.test
#

"""

"""
from unittest import TestCase

from trainer2 import manager
from trainer2.config import Config
from trainer2.flags import get_flags

FLAGS = get_flags()


class SuperTestManager(TestCase):
    def setUp(self):
        FLAGS.manager_type = 'local'
        self.config = Config.local_config()

    def tearDown(self):
        self.config = None


class TestManager(SuperTestManager):
    def test_override_local_var(self):
        # self.assertRaises(AssertionError, manager.OverrideToLocalVariableIfNotPsVar(None, 'new-name'))
        pass

    def test_get_manager(self):
        self.assertIsInstance(manager.get_manager('local', config=self.config), manager.LocalManager)

        self.config.job_name = 'master'
        self.assertIsInstance(manager.get_manager('ps', config=self.config), manager.PSManager)
        self.assertIsInstance(manager.get_manager('dr', config=self.config), manager.DRManager)

        self.assertRaises(AssertionError, manager.get_manager, 'incorrect', self.config)

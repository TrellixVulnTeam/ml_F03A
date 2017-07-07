#
# test_log_fn
# ml
#

"""

"""
from unittest import TestCase
from trainer2.util import log_fn


class TestLog_fn(TestCase):
    def test_log_fn(self):
        self.assertRaises(AssertionError, log_fn, 'log this message', 'string_not_int')

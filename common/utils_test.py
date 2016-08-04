'''
Created on Aug 4, 2016

@author: lxh5147
'''
import unittest
from utils import check_and_throw_if_fail

class UtilsTest(unittest.TestCase):

    def test_check_and_throw_if_fail(self):
        check_and_throw_if_fail(True, "nothing happens")
        try:
            check_and_throw_if_fail(False, "an exception was thrown")
            self.assertEqual(True, False, "should not happen")
        except Exception as e:
            self.assertEqual(e.message, "an exception was thrown", "exception captured")

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

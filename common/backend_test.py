'''
Created on Aug 4, 2016

@author: lxh5147
'''
import unittest
from backend import get_shape
from keras.layers import Input
import keras.backend as K
import numpy as np
class BackendTest(unittest.TestCase):


    def test_shape(self):
        # Input is a keras tensor
        x = Input((3,))
        self.assertEqual(get_shape(x), (None, 3), 'get_shape')
        x = K.variable((None, 3))
        # Placeholder is keras tensor
        x = K.placeholder(shape = (None, 3), ndim = 2)
        self.assertEqual(get_shape(x), (None, 3), 'get_shape')
        # Variable is not a keras tensor
        x = K.variable(np.random.random((3, 2)))
        try:
            get_shape(x)
            self.assertTrue(False, 'should not be executed')
        except Exception as e:
            self.assertTrue(e.message.startswith('You tried to call get_shape on'), 'get_shape')
            self.assertTrue(True, 'get_shape for non keras tensor')

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

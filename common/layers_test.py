'''
Created on Aug 26, 2016

@author: lxh5147
'''
import unittest
from layers import BiDirectionalLayer
import numpy as np
from keras.layers import Input
import keras.backend as K

class LayersTest(unittest.TestCase):
    def test_BiDirectionalLayer(self):
        layer = BiDirectionalLayer(time_step_axis = 1)
        # work when time steps = None
        left_to_right = Input((None, 3))
        right_to_left = Input((None, 2))
        output = layer([left_to_right, right_to_left])
        f = K.function(inputs = [left_to_right, right_to_left ], outputs = [output])
        left_to_right_val = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
        right_to_left_val = [[[0.1, 0.2], [0.4, 0.5]], [[0.7, 0.8], [1.0, 1.1]]]
        output_val_ref = [[[1, 2, 3, 0.4, 0.5], [4, 5, 6, 0.1, 0.2]], [[7, 8, 9, 1.0, 1.1], [10, 11, 12, 0.7, 0.8]]]
        output_val = f([left_to_right_val, right_to_left_val])[0]
        self.assertTrue(np.sum(np.abs(output_val - output_val_ref)) < 0.0001, 'output_val')

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

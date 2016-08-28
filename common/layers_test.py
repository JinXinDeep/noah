'''
Created on Aug 26, 2016

@author: lxh5147
'''
import unittest
from layers import BiDirectionalLayer, MLPClassifierLayer
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
        # check keras shape
        self.assertEqual(output._keras_shape, (None, None, 5), "_keras_shape")
        # check with call
        f = K.function(inputs = [left_to_right, right_to_left ], outputs = [output])
        left_to_right_val = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
        right_to_left_val = [[[0.1, 0.2], [0.4, 0.5]], [[0.7, 0.8], [1.0, 1.1]]]
        output_val_ref = [[[1, 2, 3, 0.4, 0.5], [4, 5, 6, 0.1, 0.2]], [[7, 8, 9, 1.0, 1.1], [10, 11, 12, 0.7, 0.8]]]
        output_val = f([left_to_right_val, right_to_left_val])[0]
        self.assertTrue(np.sum(np.abs(output_val - output_val_ref)) < 0.0001, 'output_val')

    def test_MLPClassifierLayer(self):
        output_dim = 4
        hidden_unit_numbers = [2, 3, 4]
        hidden_unit_activation_functions = ['relu', 'relu', 'relu']
        layer = MLPClassifierLayer(output_dim, hidden_unit_numbers, hidden_unit_activation_functions,
                 output_activation_function = 'softmax', use_sequence_input = True)
        # TODO: when in keras 1.0.8, time steps can be None
        input_tensor = Input((None, 2))
        output_tensor = layer(input_tensor)
        self.assertEqual(output_tensor._keras_shape, (None, None, 4), "_keras_shape")

        f = K.function(inputs = [input_tensor], outputs = [output_tensor])
        input_tensor_value = [[[1, 2], [3, 4], [5, 6]], [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]]  # 2 samples, 3 time steps
        output_tensor_value = f([input_tensor_value])[0]
        self.assertEqual(output_tensor_value.shape, (2, 3, 4), "output_tensor_value")
        # TODO: verify the correctness of the value
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

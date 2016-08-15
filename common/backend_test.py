'''
Created on Aug 4, 2016

@author: lxh5147
'''
import unittest
from backend import get_shape, get_length_without_padding, unpack, reverse, reshape, inner_product, top_k
from keras.layers import Input
import keras.backend as K
import numpy as np
class BackendTest(unittest.TestCase):


    def test_get_shape(self):
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

    def test_get_length_without_padding(self):
        x = Input((2, 3))
        length_without_padding = get_length_without_padding(x)
        f = K.function(inputs = [x], outputs = [length_without_padding])
        x_val = [[[1, 2, 3], [0, 0, 0]], [[1, 2, 3], [1, 0, 0]]]
        len_val = f([x_val])[0]
        self.assertTrue(np.array_equal(len_val, [1, 2]), "len_val")

    def test_unpack(self):
        x = Input((3,))
        # num must match the shape of data input; otherwise will get exception
        x_list = unpack(x, 2)
        f = K.function(inputs = [x], outputs = x_list)
        x_val = [[3, 2, 2], [4, 2, 2]]
        x_list_val = f([x_val])
        self.assertEqual(len(x_list_val), 2, "x_list_val")
        self.assertTrue(np.array_equal(x_list_val[0], [3, 2, 2]), "x_list_val")
        self.assertTrue(np.array_equal(x_list_val[1], [4, 2, 2]), "x_list_val")

    def test_reverse(self):
        x = Input((3,))
        reversed_x = reverse(x, 2)
        f = K.function(inputs = [x], outputs = [reversed_x])
        x_val = [[3, 2, 2], [4, 2, 2]]
        reversed_x_val = f([x_val])[0]
        self.assertTrue(np.array_equal(reversed_x_val, [ [4, 2, 2], [3, 2, 2]]), "reversed_x_val")

    def test_reshape(self):
        x = Input((4,))
        y = reshape(x, shape = (-1, 2, 2))  # - means all the remaining
        f = K.function(inputs = [x], outputs = [y])
        x_val = [[3, 2, 2, 4], [4, 2, 2, 4]]
        y_val = f([x_val])[0]
        self.assertTrue(np.array_equal(y_val, [[ [3, 2], [2, 4]], [[4, 2], [2, 4]]]), "y_val")

    def test_inner_product(self):
        x = K.variable([2, 3, 4])
        y = K.variable([3, 4, 5])
        inner_prod = inner_product(x, y)
        f = K.function(inputs = [x, y], outputs = [inner_prod])
        inner_prod_val = f ([K.get_value(x), K.get_value(y)])[0]
        self.assertEqual(6 + 12 + 20, inner_prod_val, "inner_prod_val")
        # more complicated inner product
        x = Input((3,))
        y = K.variable([1, 2, 3])
        inner_prod = inner_product(x, y)
        f = K.function(inputs = [x, y], outputs = [inner_prod])
        inner_prod_val = f ([[[1, 2, 1], [2, 1, 2]], K.get_value(y)])[0]
        self.assertTrue(np.array_equal(inner_prod_val, [1 + 4 + 3, 2 + 2 + 6]), "inner_prod_val")

    def test_top_k(self):
        x = K.variable([2, 3, 4, 5, 1, 6])
        x_top_k, indices_top_k = top_k(x, k = 2)
        f = K.function(inputs = [x], outputs = [x_top_k, indices_top_k])
        x_top_k_val, indices_top_k_val = f ([K.get_value(x)])
        self.assertTrue(np.array_equal(x_top_k_val, [6, 5]), "x_top_k_val")
        self.assertTrue(np.array_equal(indices_top_k_val, [5, 3]), "indices_top_k_val")
        # more complicated
        x = K.variable([[2, 3, 4], [5, 1, 6]])
        x_top_k, indices_top_k = top_k(x, k = 2)
        f = K.function(inputs = [x], outputs = [x_top_k, indices_top_k])
        x_top_k_val, indices_top_k_val = f ([K.get_value(x)])
        self.assertTrue(np.array_equal(x_top_k_val, [[4, 3], [6, 5]]), "x_top_k_val")
        self.assertTrue(np.array_equal(indices_top_k_val, [[2, 1], [2, 0]]), "indices_top_k_val")

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

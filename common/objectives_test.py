'''
Created on Aug 4, 2016

@author: lxh5147
'''
import unittest
import keras.backend as K
from math import log
from objectives import categorical_crossentropy_ex
from keras.layers import Input
class ObjectivesTest(unittest.TestCase):

    def test_categorical_crossentropy_ex(self):
        # applicable to one dimension tensor
        y_true_val = [0, 0, 1, 0, 0]
        y_pred_val = [.1, .2, .5, .1, .1]
        y_true = K.variable(y_true_val)
        y_pred = K.variable(y_pred_val)
        result = categorical_crossentropy_ex(y_true, y_pred)
        result_val = K.function([], [result])([])[0]
        self.assertEqual(-round(log(.5), 6), round(result_val, 6) , "result_val")
        # applicable to multiple dimension tensors
        y_true = Input((None, 2))  # shape = (none,3,2)
        y_pred = Input((None, 2))
        result = categorical_crossentropy_ex(y_true, y_pred)
        func = K.function([y_true, y_pred], [result])
        y_true_val = [[[0, 1], [1, 0], [0, 1]], [[1, 0], [0, 1], [0, 1]]]
        y_pred_val = [[[.2, .8], [.3, .7], [.4, .6]], [[.55, .45], [.6, .4], [.7, .3]]]
        result_val = func([y_true_val, y_pred_val])[0]
        self.assertEqual(-round(log(.8) + log(.3) + log(.6) + log(.55) + log(.4) + log(.3), 6), round(result_val, 6), "result_val")

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

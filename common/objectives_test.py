'''
Created on Aug 4, 2016

@author: lxh5147
'''
import unittest
import keras.backend as K
from math import log
from objectives import categorical_crossentropy_ex

class ObjectivesTest(unittest.TestCase):

    def test_categorical_crossentropy_ex(self):
        y_true_val = [0, 0, 1, 0, 0]
        y_pred_val = [.1, .2, .5, .1, .1]
        y_true = K.variable(y_true_val)
        y_pred = K.variable(y_pred_val)
        log_likelihood = categorical_crossentropy_ex(y_true, y_pred)
        log_likelihood_val = K.function([], [log_likelihood])([])[0]
        self.assertEqual(-round(log(.5), 6), round(log_likelihood_val, 6) , "log_likelihood_val")

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

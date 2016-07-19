'''
Created on Jul 19, 2016

@author: lxh5147
'''
import keras.backend as K
from .utils import check_and_throw_if_fail
from .backend import  shape
def categorical_crossentropy_ex(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes.
    '''
    check_and_throw_if_fail(shape(y_true) == shape(y_pred), "y_true")
    return K.sum(K.categorical_crossentropy(y_pred, y_true))


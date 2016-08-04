'''
Created on Jul 19, 2016

@author: lxh5147
'''
import keras.backend as K

def categorical_crossentropy_ex(y_true, y_pred):
    '''
    Expects a binary class matrix instead of a vector of scalar classes.
    '''
    return K.sum(K.categorical_crossentropy(y_pred, y_true))


'''
Created on Jul 19, 2016

@author: lxh5147
'''
import keras.backend as K

def categorical_crossentropy_ex(y_true, y_pred, from_logits=False):
    '''
    cross entropy (i.e., - log-likelihood) of the ground truth based on current prediction.

    # Parameters
    ----------
    y_true : binary tensor
    y_pred: a tensor of float, must have the same shape of y_true
        each element of y_pred is a float between [0,1], and the sum of y_pred along the last dimension is a ONE tensor
    from_logits: y_pred represents the unscaled log probabilities

    # Returns
    ------
    - log-likelihood of y_true under y_pred
    '''
    return K.sum(K.categorical_crossentropy(y_pred, y_true, from_logits))


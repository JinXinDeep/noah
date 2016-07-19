'''
Created on Jul 16, 2016

@author: lxh5147
'''
from keras import backend as K
from .utils import check_and_throw_if_fail

if K._BACKEND == 'theano':
    def  unpack(x):
        return [x[i] for i in range(x.shape()[0])]

    def shape(x):
        return x.shape()

elif K._BACKEND == 'tensorflow':
    import tensorflow as tf
    def  unpack(x):
        return tf.unpack(x)
    def shape(x):
        return tuple([i.__int__() for i in x.get_shape()])

def inner_product(x, y):
    '''
    x: a tensor, with a shape ..,inner_product_dim
    y: 1 dimension tensor with shape inner_product_dim
    '''
    x_shape = shape(x)
    check_and_throw_if_fail(len(x_shape) > 0 , "x")
    check_and_throw_if_fail(len(shape(y)) == 1 , "y")
    inner_product_dim = x_shape[-1]
    check_and_throw_if_fail(shape(y)[0] == inner_product_dim , "y")
    x = K.expand_dims(x, -2)    # 1*inner_product_dim
    y = K.expand_dims(y)    # inner_product_dim*1
    output = K.dot(x, y)    # 1*1
    output = K.squeeze(output, -1)
    output = K.squeeze(output, -1)
    return output
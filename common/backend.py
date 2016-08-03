'''
Created on Jul 16, 2016

@author: lxh5147
'''
from keras import backend as K
from .utils import check_and_throw_if_fail

def shape(x):
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    else:
        raise Exception('You tried to shape on "' + x.name + '". This tensor has no information  about its expected input shape,')

if K._BACKEND == 'theano':
    from theano import tensor as T
    def unpack(x, len):
        return [x[i] for i in range(len)]
    def reverse(x):
        return x[::-1]
    def top_k(x, k):
        raise Exception('Not implemented yet!')
    def reshape(x, shape, ndim=None):
        return T.reshape(x, shape, ndim)

elif K._BACKEND == 'tensorflow':
    import tensorflow as tf
    def unpack(x, len=None):
        ls = tf.unpack(x)
        if len:
            return ls[:len]
        else
            return ls
    def reverse(x):
        x_list = tf.unpack(x)
        x_list.reverse()
        return K.pack(x_list)
    # Finds values and indices of the k largest entries for the last dimension.
    def top_k(x, k):
        return tf.nn.top_k(x, k)
    def reshape(x, shape, ndim=None):
        return tf.reshape(x, shape)

def inner_product(x, y):
    '''
    x: a tensor, with a shape ..,inner_product_dim
    y: 1 dimension tensor with shape inner_product_dim
    '''
    x = K.expand_dims(x, -2)    # 1*inner_product_dim
    y = K.expand_dims(y)    # inner_product_dim*1
    output = K.dot(x, y)    # 1*1
    output = K.squeeze(output, -1)
    output = K.squeeze(output, -1)
    return output

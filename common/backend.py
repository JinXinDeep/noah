'''
Created on Jul 16, 2016

@author: lxh5147
'''
from keras import backend as K

def get_shape(x):
    '''
    get_shape (i.e., a tuple of integers) of a keras tensor.

    # Parameters
    ----------
    x : a keras tensor, which has a property of _keras_shape

    # Returns
    ------
    a tuple of integers, which representing the get_shape of the keras tensor

    # Raises
    ------
    Exception
        if the input tensor does not has _keras_shape property
    '''
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    else:
        raise Exception('You tried to call get_shape on "' + x.name + '". This tensor has no information about its expected input get_shape.')

def get_length_without_padding(x):
    '''
    x: batch_size, time_steps, input_dim
    '''
    s = K.sum(x, axis = [0, 2])  # time_steps
    return K.sum (K.cast(K.not_equal(s, 0), 'int32'))

def trim_right_padding(x):
    '''
    x: batch_size, time_steps, input_dim
    '''
    if K.ndim(x) == 2:
        y = K.expand_dims(x)
        length_without_padding = get_length_without_padding(y)
        return x[:, :length_without_padding]
    else:
        y = x
        length_without_padding = get_length_without_padding(y)
        return x[:, :length_without_padding, :]

if K._BACKEND == 'theano':
    from theano import tensor as T
    def unpack(x, length):
        return [x[i] for i in range(length)]
    def reverse(x):
        return x[::-1]
    def top_k(x, k):
        raise Exception('Not implemented yet!')
    def reshape(x, get_shape, ndim = None):
        return T.reshape(x, get_shape, ndim)

elif K._BACKEND == 'tensorflow':
    import tensorflow as tf
    def unpack(x, length = None):
        ls = tf.unpack(x)
        if length:
            return ls[:length]
        else:
            return ls
    def reverse(x):
        x_list = tf.unpack(x)
        x_list.reverse()
        return K.pack(x_list)
    # Finds values and indices of the k largest entries for the last dimension.
    def top_k(x, k):
        return tf.nn.top_k(x, k)
    def reshape(x, get_shape, ndim = None):
        return tf.reshape(x, get_shape)

def inner_product(x, y):
    '''
    x: a tensor, with a get_shape ..,inner_product_dim
    y: 1 dimension tensor with get_shape inner_product_dim
    '''
    x = K.expand_dims(x, -2)  # 1*inner_product_dim
    y = K.expand_dims(y)  # inner_product_dim*1
    output = K.dot(x, y)  # 1*1
    output = K.squeeze(output, -1)
    output = K.squeeze(output, -1)
    return output

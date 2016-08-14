'''
Created on Jul 16, 2016

@author: lxh5147
'''
from keras import backend as K

def get_shape(x):
    '''Gets shape (i.e., a tuple of integers) of a keras tensor.

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
    '''Gets length without padding (right) of a input tensor.

    # Parameters
    ----------
    x : a tensor whose dimensions >=3 , with the last two dimensions of the shape time_steps, input_dim

    # Returns
    ------
    a tensor represents the length of the input tensor after removing all right padding zeros
    '''
    s = K.sum(x, axis = -1)  # ..., time_steps
    return K.sum (K.cast(K.not_equal(s, 0), 'int32'), axis = -1)  # ...

if K._BACKEND == 'theano':
    from theano import tensor as T
    def unpack(x):
        '''Gets a list of tensors by slicing a tensor along its first dimension.

        # Parameters
        ----------
        x : a tensor whose dimensions >= 1
        num : number of tensors to return

        # Returns
        ------
        a list of tensors sliced by the first dimension of the input tensor
        '''
        return x[::]
    def reverse(x):
        '''Reverses elements of a tensor along its first dimension.

        # Parameters
        ----------
        x : a tensor whose dimensions >= 1

        # Returns
        ------
        the reversed tensor with the same shape of the input
        '''
        return x[::-1]
    def top_k(x, k):
        raise Exception('Not implemented yet!')
    def reshape(x, shape, ndim = None):
        """Reshapes a tensor.

          Given `tensor`, this operation returns a tensor that has the same values
          as `tensor` with shape `shape`.

          If one component of `shape` is the special value -1, the size of that dimension
          is computed so that the total size remains constant.  In particular, a `shape`
          of `[-1]` flattens into 1-D.  At most one component of `shape` can be -1.

          If `shape` is 1-D or higher, then the operation returns a tensor with shape
          `shape` filled with the values of `tensor`. In this case, the number of elements
          implied by `shape` must be the same as the number of elements in `tensor`.

          For example:

          ```prettyprint
          # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
          # tensor 't' has shape [9]
          reshape(t, [3, 3]) ==> [[1, 2, 3],
                                  [4, 5, 6],
                                  [7, 8, 9]]

          # tensor 't' is [[[1, 1], [2, 2]],
          #                [[3, 3], [4, 4]]]
          # tensor 't' has shape [2, 2, 2]
          reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
                                  [3, 3, 4, 4]]

          # tensor 't' is [[[1, 1, 1],
          #                 [2, 2, 2]],
          #                [[3, 3, 3],
          #                 [4, 4, 4]],
          #                [[5, 5, 5],
          #                 [6, 6, 6]]]
          # tensor 't' has shape [3, 2, 3]
          # pass '[-1]' to flatten 't'
          reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]

          # -1 can also be used to infer the shape

          # -1 is inferred to be 9:
          reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                   [4, 4, 4, 5, 5, 5, 6, 6, 6]]
          # -1 is inferred to be 2:
          reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                   [4, 4, 4, 5, 5, 5, 6, 6, 6]]
          # -1 is inferred to be 3:
          reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
                                        [2, 2, 2],
                                        [3, 3, 3]],
                                       [[4, 4, 4],
                                        [5, 5, 5],
                                        [6, 6, 6]]]

          # tensor 't' is [7]
          # shape `[]` reshapes to a scalar
          reshape(t, []) ==> 7
          ```

        # Parameters
        ----------
        tensor: A `Tensor`.
        shape: A `Tensor` of type `int32`. Defines the shape of the output tensor.
        ndim: the length of the shape; if ndim = None, the length of the shape must be able to be inferred from shape

        # Returns
        ------
            A `Tensor`. Has the same type as `tensor`.
        """
        return T.reshape(x, shape, ndim)

elif K._BACKEND == 'tensorflow':
    import tensorflow as tf
    def unpack(x, num = None):
        '''Gets a list of tensors by slicing a tensor along its first dimension.

        # Parameters
        ----------
        x : a tensor whose dimensions >= 1
        num : number of tensors to return; if None, x's shape of its first dimension must be specified

        # Returns
        ------
        a list of tensors sliced by the first dimension of the input tensor
        '''
        return tf.unpack(x, num = num)

    def reverse(x, num = None):
        '''Reverses elements of a tensor along its first dimension.

        # Parameters
        ----------
        x : a tensor whose dimensions >= 1
        num : shape of the first dimension of the input tensor; if None, the shape of the first dimension of the input tensor must be specified

        # Returns
        ------
        the reversed tensor with the same shape of the input
        '''
        x_list = tf.unpack(x, num)
        x_list.reverse()
        return K.pack(x_list)

    # Finds values and indices of the k largest entries for the last dimension.
    def top_k(x, k):
        return tf.nn.top_k(x, k)

    def reshape(x, shape):
        """Reshapes a tensor.

          Given `tensor`, this operation returns a tensor that has the same values
          as `tensor` with shape `shape`.

          If one component of `shape` is the special value -1, the size of that dimension
          is computed so that the total size remains constant.  In particular, a `shape`
          of `[-1]` flattens into 1-D.  At most one component of `shape` can be -1.

          If `shape` is 1-D or higher, then the operation returns a tensor with shape
          `shape` filled with the values of `tensor`. In this case, the number of elements
          implied by `shape` must be the same as the number of elements in `tensor`.

          For example:

          ```prettyprint
          # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
          # tensor 't' has shape [9]
          reshape(t, [3, 3]) ==> [[1, 2, 3],
                                  [4, 5, 6],
                                  [7, 8, 9]]

          # tensor 't' is [[[1, 1], [2, 2]],
          #                [[3, 3], [4, 4]]]
          # tensor 't' has shape [2, 2, 2]
          reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
                                  [3, 3, 4, 4]]

          # tensor 't' is [[[1, 1, 1],
          #                 [2, 2, 2]],
          #                [[3, 3, 3],
          #                 [4, 4, 4]],
          #                [[5, 5, 5],
          #                 [6, 6, 6]]]
          # tensor 't' has shape [3, 2, 3]
          # pass '[-1]' to flatten 't'
          reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]

          # -1 can also be used to infer the shape

          # -1 is inferred to be 9:
          reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                   [4, 4, 4, 5, 5, 5, 6, 6, 6]]
          # -1 is inferred to be 2:
          reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                   [4, 4, 4, 5, 5, 5, 6, 6, 6]]
          # -1 is inferred to be 3:
          reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
                                        [2, 2, 2],
                                        [3, 3, 3]],
                                       [[4, 4, 4],
                                        [5, 5, 5],
                                        [6, 6, 6]]]

          # tensor 't' is [7]
          # shape `[]` reshapes to a scalar
          reshape(t, []) ==> 7
          ```

        # Parameters
        ----------
        tensor: A `Tensor`.
        shape: A `Tensor` of type `int32`. Defines the shape of the output tensor.

        # Returns
        ------
            A `Tensor`. Has the same type as `tensor`.
        """
        return tf.reshape(x, shape)

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

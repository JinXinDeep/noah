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

def get_time_step_length_without_padding(x, time_step_dim = -2, padding = 0):
    '''Gets time steps without padding (right) of a input tensor.

    # Parameters
    ----------
    x : a tensor whose dimensions >=3
    time_step_dim: the time step dimension of x
    padding : a scalar tensor that represents the padding
    # Returns
    ------
    a tensor represents the length of the input tensor after removing all right padding zeros
    '''
    ndim = K.ndim(x)
    time_step_dim = time_step_dim % ndim
    x = K.cast(K.not_equal(x, padding), 'int32')  # binary tensor
    axis = [i for i in range(ndim) if i != time_step_dim]
    s = K.sum(x, axis)
    s = K.cast(K.not_equal(s, 0), 'int32')
    return K.sum(s)

def inner_product(x, y):
    '''Gets the inner product between a tensor and a vector. The last dimension of that tensor must have the same shape as the vector.

    # Parameters
    ----------
    x : a tensor whose dimensions >=2, of a shape .., vector_dim
    y : a vector (one dimension vector of shape vector_dim

    # Returns
    ------
    a tensor with ndim-1 dimensions, where ndim is the number of dimensions of the input tensor
    '''
    x = K.expand_dims(x, -2)  # ..., 1, vector_dim
    y = K.expand_dims(y)  # vector_dim,1
    output = K.dot(x, y)  # ..., 1*1
    output = K.squeeze(output, -1)
    output = K.squeeze(output, -1)
    return output

def beam_search(initial_input, initial_state, constant_context, embedding, step_func, beam_size = 1, max_length = 20):
    number_of_samples = K.shape(initial_input)[0]
    state_dim = K.shape(initial_state)[-1]
    current_input = K.repeat_elements(initial_input, beam_size, 0)  # shape: nb_samples*beam_size, input_dim
    current_state = K.repeat_elements(initial_state, beam_size, 0)  # shape: nb_samples*beam_size, state_dim
    output_score = K.sum(K.zeros_like(current_state), -1)  # shape: nb_samples*beam_size

    output_score_list = []  # nb_samples, beam_size
    output_label_id_list = []
    prev_output_index_list = []  # the index of candidate from which current label id is generated

    # TODO: support the case that max_length is a scalar tensor
    for _ in xrange(max_length):
        _step_score, _state = step_func(current_input, current_state, constant_context)  # nb_samples*beam_size , output_dim
        output_dim = K.shape(_step_score)[-1]
        # accumulate score
        _score = K.expand_dims(output_score) + K.log(_step_score)  # nb_samples*beam_size, output_dim
        # select top output labels for each sample
        _score = K.reshape(_score, shape = K.pack([number_of_samples, beam_size * output_dim ]))  # nb_samples, beam_size* output_dim
        _top_score , _top_indice = top_k (_score, beam_size)  # nb_samples, beam_size
        # update accumulated output score
        output_score_list.append (_top_score)
        output_score = K.reshape(_top_score, shape = (-1,))  # nb_samples * beam_size
        # update output label and previous output index
        # _top_indice = beam_id * output_dim + output_label_id
        prev_output_index = _top_indice // output_dim
        prev_output_index_list.append(prev_output_index)
        output_label_id = _top_indice % output_dim
        output_label_id_list.append (output_label_id)
        # update current input and current_state
        current_input = embedding (K.reshape(output_label_id, shape = (-1,)))  # nb_samples* beam_siz, input_dim
        # _state : nb_samples*beam_size, state_dim
        current_state = K.reshape (K.gather(_state, prev_output_index), shape = K.pack([number_of_samples * beam_size , state_dim ]))  # nb_samples, beam_size, state_dim

    return output_label_id_list, prev_output_index_list, output_score_list

def get_k_best_from_lattice(lattice, k = 1, eos = None):
    # from back to front
    for _ in lattice:_.reverse()
    output_label_id_list, prev_output_index_list, output_score_list = lattice

    path_list = []
    for output_score, output_label_id, prev_output_index in zip(output_score_list, output_label_id_list, prev_output_index_list):
        _score, _indice = top_k (output_score, k)  # shape: nb_samples, k
        path_list.append (gather_by_sample(output_label_id, output_indice))  # nb_sample, k
        score = gather_by_sample(output_score, output_indice)
        if eos:
            cond = K.equal(path_list[-1], eos)
            path_score = K.reshape(choose_by_cond(cond, score, path_score), shape = (-1, k))
        output_indice = gather_by_sample(prev_output_index_list, output_indice)
    if eos:
        path_score, output_indice = top_k(path_score, k)  # sort the top k path by default, nb_samples, k
        path_list = [gather_by_sample(path, output_indice) for path in path_list]
    path_list = K.permute_dimensions(K.pack(path_list), (1, 2, 0))  # time_steps, nb_samples, k -> nb_samples, k, time_steps
    return path_list, path_score

def choose_by_cond(cond, _1, _2):
    '''Performs element wise choose from _1 or _2 based on condition cond. At a give position, if the element in cond is 1, select the element from _1 otherwise from _2 from the same position.

    # Parameters
    ----------
    cond : a binary tensor
    _1 : first tensor with the same shape of cond
    _2: second tensor with the shape of cond and the same data type of _1

    # Returns
    ------
    a tensor with the shape of cond and same data type of _1
    '''
    r = []
    original_shape = K.shape(cond)
    _1 = K.reshape(_1, shape = (-1,))
    _2 = K.reshape(_2, shape = (-1,))
    cond = K.reshape(cond, shape = (-1,))
    for _c, _1, _2 in zip (unpack(cond), unpack(_1), unpack(_2)):
        r.append(K.switch(_c, _1, _2))
    output = K.pack(r)
    return K.reshape(output, original_shape)

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
    def top_k(x, k = 1):
        """Finds values and indices of the `k` largest entries for the last dimension sorted by value in descent.

        If the input is a vector (rank-1), finds the `k` largest entries in the vector
        and outputs their values and indices as vectors.  Thus `values[j]` is the
        `j`-th largest entry in `input`, and its index is `indices[j]`.

        For matrices (resp. higher rank input), computes the top `k` entries in each
        row (resp. vector along the last dimension).  Thus,

            values.shape = indices.shape = input.shape[:-1] + [k]

        If two elements are equal, the lower-index element appears first.

        # Parameters
        ----------
        input: 1-D or higher `Tensor` with last dimension at least `k`.
        k: 0-D `int32` `Tensor`.  Number of top elements to look for along the last dimension (along each row for matrices).

        # Returns:
        ----------
        values: The `k` largest elements along each last dimensional slice.
        indices: The indices of `values` within the last dimension of `input`.
        """
        x_sorted = T.sort(x)
        x_sort_arg = T.argsort(x)
        ndim = x.ndim
        if ndim == 1:
            x_sorted = x_sorted[-k:]
            x_sorted = x_sorted[::-1]
            x_sort_arg = x_sort_arg[-k:]
            x_sort_arg = x_sort_arg[::-1]
            return x_sorted, x_sort_arg
        else:
            new_shape = T.stack(*([x.shape[i] for i in range(ndim - 1)] + [k]))
            x_sorted = T.reshape(x_sorted, newshape = (-1, x.shape[-1]))[:, -k:]
            x_sorted = x_sorted[:, ::-1]
            x_sorted = T.reshape(x_sorted, new_shape, ndim = ndim)
            x_sort_arg = T.reshape(x_sort_arg, newshape = (-1, x.shape[-1]))[:, -k:]
            x_sort_arg = x_sort_arg[:, ::-1]
            x_sort_arg = T.reshape(x_sort_arg, new_shape, ndim = ndim)
            return x_sorted, x_sort_arg

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
    def top_k(x, k = 1, sorted_by_value_descent = True):
        """Finds values and indices of the `k` largest entries for the last dimension.

        If the input is a vector (rank-1), finds the `k` largest entries in the vector
        and outputs their values and indices as vectors.  Thus `values[j]` is the
        `j`-th largest entry in `input`, and its index is `indices[j]`.

        For matrices (resp. higher rank input), computes the top `k` entries in each
        row (resp. vector along the last dimension).  Thus,

            values.shape = indices.shape = input.shape[:-1] + [k]

        If two elements are equal, the lower-index element appears first.

        # Parameters
        ----------
        input: 1-D or higher `Tensor` with last dimension at least `k`.
        k: 0-D `int32` `Tensor`.  Number of top elements to look for along the last dimension (along each row for matrices).
        sorted_by_value_descent: If true the resulting `k` elements will be sorted_by_value_descent by the values in descending order.

        # Returns:
        ----------
        values: The `k` largest elements along each last dimensional slice.
        indices: The indices of `values` within the last dimension of `input`.
        """
        return tf.nn.top_k(x, k)

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
        ndim : not used.

        # Returns
        ------
            A `Tensor`. Has the same type as `tensor`.
        """
        return tf.reshape(x, shape)

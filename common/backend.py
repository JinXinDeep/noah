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
    output = dot(x, y)  # ..., 1*1
    output = K.squeeze(output, -1)
    output = K.squeeze(output, -1)
    return output

def _beam_search_one_step(_step_score, _state, output_score, number_of_samples, beam_size, state_dim, output_score_list, prev_output_index_list, output_label_id_list, embedding, _tensors_to_debug = None):
    output_dim = K.shape(_step_score)[1]  # nb_samples*beam_size, output_dim
    # accumulate score
    _score = K.expand_dims(output_score) + K.log(_step_score)  # nb_samples*beam_size, output_dim
    # select top output labels for each sample
    _score = K.reshape(_score, shape = K.pack([number_of_samples, beam_size * output_dim ]))  # nb_samples, beam_size* output_dim
    _top_score , _top_indice = top_k (_score, beam_size)  # -1, beam_size
    # update accumulated output score
    output_score_list.append (_top_score)
    output_score = K.reshape(_top_score, shape = (-1,))  # nb_samples * beam_size

    # update output label and previous output index
    # _top_indice = beam_id * output_dim + output_label_id
    prev_output_index = _top_indice // output_dim
    prev_output_index_list.append(prev_output_index)
    output_label_id = _top_indice - prev_output_index * output_dim
    output_label_id_list.append (output_label_id)
    # update current input and current_state
    current_input = embedding (K.reshape(output_label_id, shape = (-1,)))  # nb_samples* beam_siz, input_dim
    # _state : nb_samples*beam_size, state_dim
    # first reshape _state to nb_samples, beam_size, state_dim
    # then gather by sample to get a tensor with the shape: nb_samples, beam_size, state_dim
    # finally reshape to nb_samples*beam_size, state_dim
    # note that prev_output_index has a shape of -1, beam_size, so should be reshape to nb_samples, beam_size before calling gather_by_sample
    current_state = K.reshape (gather_by_sample(K.reshape(_state, shape = K.pack([number_of_samples , beam_size , state_dim ])), K.reshape(prev_output_index, shape = K.pack([number_of_samples, beam_size]))), shape = K.pack([number_of_samples * beam_size , state_dim ]))
    if _tensors_to_debug is not None:
        _tensors_to_debug += [_score, _top_score, _top_indice]
    return output_score, current_input, current_state
# output, current_state = self.step(current_input, current_state, context)
def beam_search(initial_input, initial_state, constant_context, embedding, step_func, beam_size = 1, max_length = 20):
    '''Returns a lattice with time steps = max_length and beam size = beam_size; each node of the lattice at time step t has a parent node at time step t-1, an accumulated score, and a label as its output.

    # Parameters
    ----------
    initial_input : a tensor with a shape of nb_samples, representing the initial input used by the step function
    initial_state: a tensor with a shape of nb_samples,state_dim, representing the initial state used by the step function
    constant_context: a tensor with a shape of nb_samples,context_dim, representing the context tensor used by the step function
    embedding: an embedding layer that maps input/output labels to their embedding
    step_func: in a form like step_func(current_input, current_state, constant_context), which returns a score tensor and a tensor representing the updated state
    beam_size: beam size
    max_length: max time steps to expand

    # Returns
    ------
    output_label_id_tensor: a tensor with a shape of max_length, nb_samples, beam_size of type int32, representing labels of nodes
    prev_output_index_tensor: a tensor with a shape of max_length, nb_samples, beam_size of type int32, representing parent's indexes (in the range of 0..beam_size-1) of nodes
    output_score_tensor: a tensor with a shape of max_length, nb_samples, beam_size of type float32, representing accumulated scores of nodes
    '''
    number_of_samples = K.shape(initial_input)[0]
    state_dim = K.shape(initial_state)[-1]
    current_input = K.repeat_elements(initial_input, beam_size, 0)  # shape: nb_samples*beam_size, input_dim
    current_state = K.repeat_elements(initial_state, beam_size, 0)  # shape: nb_samples*beam_size, state_dim
    output_score = K.sum(K.zeros_like(current_state), -1)  # shape: nb_samples*beam_size

    output_score_list = []  # nb_samples, beam_size
    output_label_id_list = []
    prev_output_index_list = []  # the index of candidate from which current label id is generated

    for _ in xrange(max_length):
        _step_score, _state = step_func(current_input, current_state, constant_context)  # nb_samples*beam_size , output_dim
        output_score, current_input, current_state = _beam_search_one_step(_step_score, _state, output_score, number_of_samples, beam_size, state_dim, output_score_list, prev_output_index_list, output_label_id_list, embedding)

    return K.pack(output_label_id_list), K.pack(prev_output_index_list), K.pack(output_score_list)

def get_k_best_from_lattice(lattice, k = 1, eos = None, _tensors_to_debug = None):
    '''Selects top k best path from a lattice in a descending order by their scores

    # Parameters
    ----------
    lattice : a triple consisting of output_label_id_tensor, prev_output_index_tensor and output_score_tensor. This lattice is generated by calling beam_search.
    k: the number of path to select from that lattice
    eos: if not None, it is the id of the label that represents the end of sequence

    # Returns
    ------
    sequence: a tensor of type int32 with a shape of nb_samples, k, time_stpes, representing the top-k best sequences
    sequence_score: a tensor of type float32 with a shape of nb_samples, k, representing the scores of the top-k best sequences
    '''
    lattice = [unpack(_) for _ in  lattice]
    for l in lattice: l.reverse()
    output_label_id_list, prev_output_index_list, output_score_list = lattice
    sequence_score, output_indice = top_k (output_score_list[0], k)  # shape: nb_samples,k
    if _tensors_to_debug is not None:
        _tensors_to_debug.append(sequence_score)
        _tensors_to_debug.append(output_indice)

    nb_samples = K.shape(sequence_score)[0]
    # fill sequence and update sequence_score
    sequence = []
    for cur_output_score, output_label_id, prev_output_index in zip(output_score_list, output_label_id_list, prev_output_index_list):
        sequence_score_candidate = K.reshape(gather_by_sample(cur_output_score, output_indice), shape = K.pack([nb_samples, k]))
        sequence.append (K.reshape(gather_by_sample(output_label_id, output_indice), shape = K.pack([nb_samples, k])))  # shape: -1,  k, nb_samples could be -1
        if eos is not None and len(sequence) > 1:
            cond = K.equal(sequence[-1], eos)
            sequence_score = choose_by_cond(cond, sequence_score_candidate, sequence_score)
            if _tensors_to_debug is not None:
                _tensors_to_debug.append(cond)
                _tensors_to_debug.append(sequence_score_candidate)
                _tensors_to_debug.append(sequence_score)
        output_indice = gather_by_sample(prev_output_index, output_indice)
        if _tensors_to_debug is not None:
            _tensors_to_debug.append(output_indice)

    if eos is not None and len(sequence) > 1:
        sequence_score, output_indice = top_k(sequence_score, k)
        sequence = [gather_by_sample(_, output_indice) for _ in sequence]

    # reverse the sequence so we get sequence from time step 0, 1, ...,
    sequence.reverse()
    sequence = K.permute_dimensions(K.pack(sequence), (1, 2, 0))  # time_steps, nb_samples, k -> nb_samples, k, time_steps
    return sequence, sequence_score

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
    cond = K.cast(cond, _1.dtype)
    return _1 * cond + _2 * (1 - cond)

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

    def gather_by_sample(x, indices):
        '''Performs gather operation along the first dimension, i.e., ret[i] = gather( x[i], indices[i]).
        For example, when x is a matrix, and indices is a vector, it selects one element for each row from x.
        Note that this is different from gather, which selects |indices| ndim-1 sub tensors (i.e., x[i], where i = indices[:::]) from x

        # Parameters
        ----------
        x : a tensor with a shape nb_samples, ...; its number of dimensions >= 2
        indices : a tensor of type int with a shape nb_sample,...; its number of dimensions <= # of dimensions of x - 1

        # Returns
        ------
        a tensor with the shape of nb_samples, ..., where ret[i,:::,:::]= x[i,indices[i,:::],:::]; and its number of dimensions = # dimensions of x + # dimension of indices - 2
        '''
        y_list = []
        for x_i , i in zip(unpack(x), unpack(indices)):
            y_i = K.gather(x_i, i)
            y_list.append(y_i)
        return K.pack(y_list)

    def dot(x, y):
        return T.dot(x, y)

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

    def reverse(x):
        '''Reverses elements of a tensor along its first dimension.

        # Parameters
        ----------
        x : a tensor whose dimensions >= 1
        num : shape of the first dimension of the input tensor; if None, the shape of the first dimension of the input tensor must be specified

        # Returns
        ------
        the reversed tensor with the same shape of the input
        '''
        ndim = K.ndim(x)
        dims = [True] + [False for _ in range(1, ndim)]
        return tf.reverse(x, dims)

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

    def gather_by_sample(x, indices):
        '''Performs gather operation along the first dimension, i.e., ret[i] = gather( x[i], indices[i]).
        For example, when x is a matrix, and indices is a vector, it selects one element for each row from x.
        Note that this is different from gather, which selects |indices| ndim-1 sub tensors (i.e., x[i], where i = indices[:::]) from x

        # Parameters
        ----------
        x : a tensor with a shape nb_samples, ...; its number of dimensions >= 2
        indices : a tensor of type int with a shape nb_sample,...; its number of dimensions <= # of dimensions of x - 1

        # Returns
        ------
        a tensor with the shape of nb_samples, ..., where ret[i,:::,:::]= x[i,indices[i,:::],:::]; and its number of dimensions = # dimensions of x + # dimension of indices - 2
        '''
        x_shape = K.shape(x)
        nb_samples = x_shape[0]
        ones = tf.ones(shape = K.pack([nb_samples]), dtype = 'int32')
        elems = tf.scan(lambda prev, one: prev + one , ones, initializer = tf.constant(-1, dtype = 'int32'))
        def _step(prev, i):
            x_i = K.gather(x, i)
            indices_i = K.gather(indices, i)
            return K.gather(x_i, indices_i)
        return tf.scan(_step , elems, initializer = tf.zeros(shape = x_shape[1:], dtype = x.dtype))

    # support None
    def dot(x, y):
        '''Multiplies 2 tensors.
        When attempting to multiply a ND tensor
        with a ND tensor, reproduces the Theano behavior
        (e.g. (2, 3).(4, 3, 5) = (2, 4, 5))
        '''
        ndim_x = K.ndim(x)
        ndim_y = K.ndim(y)

        if ndim_x is not None and ndim_x > 2 or ndim_y > 2:
            x_shape = tf.shape(x)
            y_shape = tf.shape(y)
            y_permute_dim = list(range(ndim_y))
            y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
            xt = tf.reshape(x, K.pack([-1, x_shape[ndim_x - 1]]))
            yt = tf.reshape(tf.transpose(y, perm = y_permute_dim), K.pack([y_shape[ndim_y - 2], -1]))
            target_shape = [x_shape[i] for i in range(ndim_x - 1)] + [y_shape[i] for i in range(ndim_y - 2)] + [y_shape[ndim_y - 1]]
            return tf.reshape(tf.matmul(xt, yt), K.pack(target_shape))
        out = tf.matmul(x, y)
        return out

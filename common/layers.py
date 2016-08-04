'''
Created on Jul 16, 2016

@author: lxh5147
'''

from keras import backend as K
from keras.engine import Layer
from .backend import reshape, reverse, inner_product, unpack, top_k, trim_right_padding
from .utils import check_and_throw_if_fail
from keras.layers import Dense, BatchNormalization
from keras.layers.wrappers import TimeDistributed

from keras import activations, initializations, regularizers
import numpy as np

'''
Helper function that performs reshape on a tensor
'''
class ReshapeLayer(Layer):
    '''
    Refer to: https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf, formula 8,9 and 10
    '''
    def __init__(self, target_shape, target_tensor_shape=None, ** kwargs):
        self.target_shape = tuple(target_shape)
        self.target_tensor_shape = target_tensor_shape
        super(ReshapeLayer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if self.target_tensor_shape:
            return reshape(x, self.target_tensor_shape, ndim=len(self.target_shape))    # required by theano
        else:
            return reshape(x, self.target_shape)

    def get_output_shape_for(self, input_shape):
        return self.target_shape

'''
Helper function that performs reshape on a tensor
'''
class BiDirectionalLayer(Layer):
    def call(self, inputs, mask=None):
        left_to_right = inputs[0]
        right_to_left = inputs[1]
        ndim = K.ndim(right_to_left)
        axes = [1, 0] + list(range(2, ndim))
        right_to_left = K.permute_dimensions(right_to_left, axes)
        right_to_left = reverse(right_to_left)
        right_to_left = K.permute_dimensions(right_to_left, axes)
        return K.concatenate([left_to_right, right_to_left], axis=-1)
    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][:-1] + (input_shapes[0][-1] + input_shapes[1][-1],)

'''
Helper function that performs reshape on a tensor
'''
class MLPClassifierLayer(Layer):
    '''
    Represents a mlp classifier, which consists of several hidden layers followed by a softmax output layer
    '''
    def __init__(self, output_dim, hidden_unit_numbers, hidden_unit_activation_functions, output_activation_function='softmax', **kwargs):
        '''
        input_sequence: input sequence, batch_size * time_steps * input_dim
        hidden_unit_numbers: number of hidden units of each hidden layer
        hidden_unit_activation_functions: activation function of hidden layers
        output_dim: output dimension
        returns a tensor of shape: batch_size*time_steps*output_dim
        '''
        check_and_throw_if_fail(output_dim > 0 , "output_dim")
        check_and_throw_if_fail(len(hidden_unit_numbers) == len(hidden_unit_activation_functions) , "hidden_unit_numbers")
        self.output_dim = output_dim
        self.hidden_unit_numbers = hidden_unit_numbers
        self.hidden_unit_activation_functions = hidden_unit_activation_functions
        self.output_activation_function = output_activation_function
        if hidden_unit_numbers:
            self.uses_learning_phase = True
        super(MLPClassifierLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.layers = []
        ndim = len(input_shape)
        for hidden_unit_number, hidden_unit_activation_function in zip(self.hidden_unit_numbers, self.hidden_unit_activation_functions):
            dense = Dense(hidden_unit_number, activation=hidden_unit_activation_function)
            if ndim == 3:
                dense = TimeDistributed(dense)
            norm = BatchNormalization(mode=2)
            self.layers.append(dense)
            self.layers.append(norm)

        dense = Dense(self.output_dim, activation=self.output_activation_function)
        if ndim == 3:
            dense = TimeDistributed(dense)
        self.layers.append(dense)

    def call(self, x, mask=None):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1] + (self.output_dim,)

class AttentionLayer(Layer):
    '''
    Refer to: http://nlp.ict.ac.cn/Admin/kindeditor/attached/file/20141011/20141011133445_31922.pdf
    TODO: should be straightforward to enhance the attention layer to support advanced attention model, such as coverage
    '''
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shapes):
        '''
        s' input_shape[0]: batch_size , input_dim
        h' input_shape[1]: batch_size , time_steps, h_input_dim
        '''
        initial_W_a = np.random.random((input_shapes[0][-1], self.output_dim))
        initial_U_a = np.random.random((input_shapes[-1][-1], self.output_dim))
        initial_v_a = np.random.random((self.output_dim,))
        self.W_a = K.variable(initial_W_a)
        self.U_a = K.variable(initial_U_a)
        self.v_a = K.variable(initial_v_a)
        self.trainable_weights = [self.W_a, self.U_a, self.v_a]
        self.U_a_h = K.dot(self.h, self.U_a)    # batch_size, time_steps, output_dim

    def call(self, inputs, mask=None):
        '''
        s: batch_size , input_dim
        h: batch_size,time_steps, h_input_dim
        '''
        s, h = inputs
        W_a_s = K.expand_dims(K.dot(s, self.W_a), 1)    # batch_size, 1, output_dim
        e = K.tanh (W_a_s + self.U_a_h)    # batch_size, time_steps, output_dim
        e = inner_product(e, self.v_a)    # shape: batch_size, time_steps
        e = K.exp (e)
        e_sum = K.sum(e, -1, keepdims=True)    # batch_size, 1
        a = e / e_sum    # batch_size, time_steps
        a = K.expand_dims(a)    # batch_size, time_steps, 1
        c = a * h    # batch_size, time_steps, h_input_dim
        c = K.sum(c, axis=1)    # batch_size, h_input_dim
        return c

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[1][0], input_shapes[1][2])

class GRUCell(Layer):
    '''
    general version of: http://arxiv.org/pdf/1409.0473v3.pdf, which supports multiple inputs
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='hard_sigmoid', consume_less='gpu',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = W_regularizer
        self.U_regularizer = U_regularizer
        self.b_regularizer = b_regularizer
        self.consume_less = consume_less
        super(GRUCell, self).__init__(**kwargs)

    def build(self, input_shapes):
        self.W = []
        input_dims = [input_shape[1] for input_shape in input_shapes[:-1]]
        if self.consume_less == 'gpu':
            for input_dim in input_dims:
                self.W.append (self.inner_init((input_dim, 3 * self.output_dim)))
            self.U = self.inner_init((self.output_dim, 3 * self.output_dim))
            self.b = K.variable(np.hstack((np.zeros(self.output_dim), np.zeros(self.output_dim), np.zeros(self.output_dim))))
            self.trainable_weights = self.W + [self.U, self.b]
        else:
            self.W_z = []
            self.W_r = []
            self.W_h = []
            self.W = []
            for input_dim in input_dims:
                self.W_z.append(self.inner_init((input_dim, self.output_dim)))
                self.W_r.append(self.inner_init((input_dim, self.output_dim)))
                self.W_h.append(self.inner_init((input_dim, self.output_dim)))
                self.W.append (K.concatenate([self.W_z[-1], self.W_r[-1], self.W_h[-1]]))

            self.U_z = self.inner_init((self.output_dim, self.output_dim))
            self.U_r = self.inner_init((self.output_dim, self.output_dim))
            self.U_h = self.inner_init((self.output_dim, self.output_dim))
            self.U = K.concatenate([self.U_z, self.U_r, self.U_h])

            self.b_z = K.zeros((self.output_dim,))
            self.b_r = K.zeros((self.output_dim,))
            self.b_h = K.zeros((self.output_dim,))
            self.b = K.concatenate([ self.b_z, self.b_r, self.b_h])

            self.trainable_weights = self.W_z + self.W_r + self.W_h + [self.U_z, self.U_r, self.U_h, self.b_z, self.b_r, self.b_h]

        self.regularizers = []
        if self.W_regularizer:
            for W in self.W:
                regularizer = regularizers.get(self.W_regularizer)
                regularizer.set_param(W)
                self.regularizers.append(regularizer)
        if self.U_regularizer:
            regularizer = regularizers.get(self.U_regularizer)
            regularizer.set_param(self.U)
            self.regularizers.append(regularizer)
        if self.b_regularizer:
            regularizer = regularizers.get(self.b_regularizer)
            regularizer.set_param(self.b)
            self.regularizers.append(regularizer)

    def call(self, inputs, mask=None):
        check_and_throw_if_fail(len(inputs) == 3 , "inputs")
        # the last one is previous state
        h_prev = inputs[-1]
        if self.consume_less == 'gpu':
            x = self.b
            for y, W in zip(inputs[:-1], self.W):
                x += K.dot(y , W)

            x_z = x[:, :self.output_dim]
            x_r = x[:, self.output_dim: 2 * self.output_dim]
            x_h = x[:, 2 * self.output_dim:]

            matrix_inner = K.dot(h_prev , self.U[:, :2 * self.output_dim])
            inner_z = matrix_inner[:, :self.output_dim]
            inner_r = matrix_inner[:, self.output_dim: 2 * self.output_dim]
            z = self.inner_activation(x_z + inner_z)
            r = self.inner_activation(x_r + inner_r)
            inner_h = K.dot(r * h_prev , self.U[:, 2 * self.output_dim:])
            hh = self.activation(x_h + inner_h)
        else:
            x_z = self.b_z
            x_r = self.b_r
            x_h = self.b_h
            for y, W_z, W_r, W_h in zip (inputs[:-1], self.W_z, self.W_r, self.W_h):
                x_z += K.dot(y, W_z)
                x_r += K.dot(y, W_r)
                x_h += K.dot(y, W_h)
            z = self.inner_activation(x_z + K.dot(h_prev , self.U_z))
            r = self.inner_activation(x_r + K.dot(h_prev, self.U_r))
            inner_h = K.dot(r * h_prev , self.U_h)
            hh = self.activation(x_h + inner_h)
        return  (1 - z) * h_prev + z * hh    # consistent with the formula in the paper

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], self.output_dim)

def RNNLayerBase(Layer):

    def __init__(self, rnn_cell, attention, output_embedding, mlp_classifier, stateful=False, **kwargs):
        # TODO: apply drop out to inputs and inner outputs
        self.rnn_cell = rnn_cell
        self.attention = attention
        self.mlp_classifier = mlp_classifier
        self.output_embedding = output_embedding
        self.stateful = stateful
        self.uses_learning_phase = mlp_classifier.uses_learning_phase or rnn_cell.uses_learning_phase or output_embedding.output_embedding or attention.uses_learning_phase
        super(RNNLayerBase, self).__init__(**kwargs)

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(x)    # (samples, timesteps)
        initial_state = K.sum(initial_state, axis=(1,))    # (samples,)
        initial_state = K.expand_dims(initial_state)    # (samples, 1)
        initial_state = K.tile(initial_state, [1, self.output_dim])    # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def get_output_shape_for(self, input_shape):
        raise NotImplementedError

    def step(self, x, states, source_context):
        h_prev = states[0]    # previous memory
        c = self.attention(h_prev, source_context)
        h = self.rnn_cell([x, c, h_prev])
        output = self.mlp_classifier(h)
        return output, [h]

    def build(self, input_shapes):
        raise NotImplementedError

    def call(self, inputs, mask=None):
        raise NotImplementedError

def RNNLayer(RNNLayerBase):
    '''
    RNN based decoder for training, using the ground truth output
    '''
    def get_output_shape_for(self, input_shapes):
        input_shape, _ = input_shapes
        return (input_shape[0], input_shape[1], self.mlp_classifier.output_dim)

    def build(self, input_shapes):
        input_shape, _ = input_shapes
        check_and_throw_if_fail(len(input_shape) == 2, "input_shape=(nb_samples, time_steps)")
        if self.stateful:
            check_and_throw_if_fail(not input_shape[0] , 'If a RNN is stateful, a complete  input_shape must be provided (including batch size).')
            self.states = [K.zeros((input_shape[0], self.rnn_cell.output_dim))]
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]

    def call(self, inputs, mask=None):
        # input shape: (nb_samples, time (padded with zeros))
        x, source_context = inputs
        x = trim_right_padding(x)    # to make computation more efficient
        if self.stateful:
            current_states = self.states
        else:
            current_states = self.get_initial_states(x)
        x = self.embedding(x)
        x = K.permute_dimensions(x, axes=[1, 0, 2])
        input_list = unpack(x)
        successive_states = []
        successive_outputs = []
        for current_input in input_list:
            # TODO: randomly use the real greedy output as the next input
            output, current_states = self.step(current_input, current_states, source_context)
            successive_outputs.append(output)
            successive_states.append(current_states)
        outputs = K.pack(successive_outputs)
        outputs = K.permute_dimensions(outputs, axes=[1, 0, 2])
        new_states = successive_states[-1]
        if self.stateful:
            self.updates = []
            for i in range(len(new_states)):
                self.updates.append((self.states[i], new_states[i]))
        return outputs


def RNNBeamSearchDecoder(RNNLayerBase):
    '''
    RNN based decoder for prediction, using beam search
    '''
    def __init__(self, max_output_length, beam_size, number_of_output_sequence=1, eos=None, **kwargs):
        check_and_throw_if_fail(max_output_length > 0, "check_and_throw_if_fail")
        check_and_throw_if_fail(beam_size > 0, "beam_size")
        check_and_throw_if_fail(number_of_output_sequence > 0, "number_of_output_sequence")
        check_and_throw_if_fail(beam_size >= number_of_output_sequence, "number_of_output_sequence")
        self.max_output_length = max_output_length
        self.beam_size = beam_size
        self.number_of_output_sequence = number_of_output_sequence
        self.eos = None
        super(RNNBeamSearchDecoder, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        # two outputs: sequence and score
        return (input_shape[0], self.number_of_output_sequence, self.max_output_length), (input_shape[0], self.number_of_output_sequence)

    def build(self, input_shapes):
        check_and_throw_if_fail(self.stateful == False, "stateful")
        input_shape = input_shapes[-1]
        check_and_throw_if_fail(len(input_shape) == 2, "input_shape=(nb_samples, source_context_input_dim")
        self.states = [None]

    @staticmethod
    def gather_per_sample(x, indices):
        y_list = []
        for xi , indice in zip(unpack(x), unpack(indices)):
            yi = K.gather(xi, indice)
            y_list.append(yi)
        return K.pack(y_list)

    def call(self, inputs, mask=None):
        check_and_throw_if_fail(self.beam_size <= self.mlp_classifier.output_dim , "beam_size")
        # input shape: (nb_samples, time (padded with zeros))
        x, source_context = inputs
        if K.ndim(x) == 2:
            x = K.squeeze(x, 1)
        # x is the initial input, i.e., bos
        current_states = self.get_initial_states(x)
        current_input = self.embedding(x)
        current_input = K.repeat_elements(current_input, self.beam_size, 0)    #  nb_samples* beam_size, input_dim
        current_states = K.repeat_elements(current_states, self.beam_size, 0)
        output_scores = K.cast (K.zeros_like(x), K.common._FLOATX)
        output_scores = K.repeat_elements(output_scores, self.beam_size, 0)    # nb_samples*beam_size
        output_scores_list = []    # nb_samples, beam_size
        output_label_id_list = []
        prev_output_index_list = []
        for _ in xrange(self.max_output_length):
            output, current_states = self.step(current_input, current_states, source_context)    # nb_samples*beam_size , output_dim
            scores = K.expand_dims(output_scores) + K.log(output)
            scores = K.reshape(scores, shape=(-1, self.beam_size, self.mlp_classifier.output_dim))    # nb_samples, beam_size,  output_dim
            top_k_k_scores, top_k_k_scores_indices = top_k (scores, self.beam_size)    # nb_samples, beam_size,  beam_size
            top_k_k_scores = K.reshape(top_k_k_scores , shape=(-1, self.beam_size * self.beam_size))    # nb_samples, beam_size* beam_size
            top_k_k_scores_indices = K.reshape(top_k_k_scores_indices , shape=(-1, self.beam_size * self.beam_size))
            # nb_samples, k, k
            top_k_scores, top_k_scores_indices = top_k (top_k_k_scores, self.beam_size)    #  nb_samples, beam_size
            scores = K.reshape(top_k_scores, shape=(-1, self.mlp_classifier.output_dim))    # nb_samples*beam_size, output_dim
            x = RNNBeamSearchDecoder.gather_per_sample(top_k_k_scores_indices, top_k_scores_indices)    # nb_samples, beam_size
            output_scores = RNNBeamSearchDecoder.gather_per_sample(scores, K.reshape(x, shape=(-1,)))    # nb_samples*beam_size
            output_scores_list.append (K.reshape(output_scores, shape=(-1, self.beam_size)))
            output_label_id_list.append(x)
            prev_output_index = top_k_scores_indices // self.beam_size    # nb_samples, beam_size
            prev_output_index_list.append (prev_output_index)
            current_input = self.embedding(x)
            current_input = K.reshape(current_input, shape=(-1, self.embedding.output_dim))
            # current states:  nb_samples*beam_size, cell_output_dim
            current_states = K.gather (current_states, prev_output_index)
        if self.eos:
            eos = self.eos + K.zeros_like(x)
            eos = K.reshape (K.repeat_elements(eos, self.number_of_output_sequence), shape=(-1, self.number_of_output_sequence))
        return RNNBeamSearchDecoder.get_k_best_from_lattice([output_scores_list, output_label_id_list, prev_output_index], self.number_of_output_sequence, eos)

    @staticmethod
    def cond_set(cond, t1, t2):
        r = []
        t1 = K.reshape(t1, shape=(-1,))
        t2 = K.reshape(t1, shape=(-1,))
        cond = K.reshape(cond, shape=(-1,))
        for _c, _1, _2 in zip (unpack(cond), unpack(t1), unpack(t2)):
            r.append(K.switch(_c, _1, _2))
        return K.pack(r)

    @staticmethod
    def get_k_best_from_lattice(lattice, k, eos):
        for l in lattice:
            l.reverse()
        output_scores_list, output_label_id_list, prev_output_index = lattice
        pathes = []
        path_scores, output_indices = top_k (output_scores_list[0], k)
        for output_scores, output_label_id, prev_output_index in zip(output_scores_list, output_label_id_list, prev_output_index):
            pathes.append (RNNBeamSearchDecoder.gather_per_sample(output_label_id, output_indices))
            scores = RNNBeamSearchDecoder.gather_per_sample(output_scores, output_indices)
            if eos:
                cond = K.equal(pathes[-1], eos)
                path_scores = K.reshape(RNNBeamSearchDecoder.cond_set(cond, scores, path_scores), shape=(-1, k))
            output_indices = RNNBeamSearchDecoder.gather_per_sample(prev_output_index, output_indices)
        if eos:
            path_scores, output_indices = top_k(path_scores, k)
            pathes = [RNNBeamSearchDecoder.gather_per_sample(path, output_indices) for path in pathes]
        pathes = K.permute_dimensions(K.pack(pathes), (1, 2, 0))
        return pathes, path_scores

'''
Created on Jul 16, 2016

@author: lxh5147
'''

from keras import backend as K
from keras.engine import Layer
from .backend import unpack, inner_product, shape
from .utils import check_and_throw_if_fail
from keras.layers import Dense, Activation, BatchNormalization
from keras.layers.wrappers import TimeDistributed

from keras import activations, initializations, regularizers
import numpy as np

class ReshapeLayer(Layer):
    '''
    Refer to: https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf, formula 8,9 and 10
    '''
    def __init__(self, target_shape, **kwargs):
        self.target_shape = target_shape
        super(ReshapeLayer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return K.reshape(x, self.target_shape)

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
        right_to_left_time_step_list = unpack(right_to_left)
        right_to_left_time_step_list.reverse()
        right_to_left = K.pack(right_to_left_time_step_list)
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
        super(MLPClassifierLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.layers = []
        ndim = len(input_shape)
        for hidden_unit_number, hidden_unit_activation_function in zip(self.hidden_unit_numbers, self.hidden_unit_activation_functions):
            dense = Dense(hidden_unit_number)
            activation = Activation(hidden_unit_activation_function)
            norm = BatchNormalization()
            if ndim == 3:
                dense = TimeDistributed(dense)
                activation = TimeDistributed(activation)
            self.layers.append(dense)
            self.layers.append(activation)
            self.layers.append(norm)

        dense = Dense(self.output_dim)
        activation = Activation(self.output_activation_function)
        if ndim == 3:
            dense = TimeDistributed(dense)
            activation = TimeDistributed(activation)

        self.layers.append(dense)
        self.layers.append(activation)

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
    def build(self, input_shapes):
        '''
        input_shape[0]: batch_size , input_dim
        input_shape[1]: batch_size , time_steps, h_input_dim
        '''
        check_and_throw_if_fail(len(input_shapes) == 2, "input_shapes")
        input_shape = input_shapes[0]
        check_and_throw_if_fail(len(input_shape) == 2, "input_shape")
        input_dim = input_shape[-1]
        initial_W_a = np.random.random((input_dim, self.output_dim))
        initial_U_a = np.random.random((shape(self.h)[-1], self.output_dim))
        initial_v_a = np.random.random((self.output_dim,))
        self.W_a = K.variable(initial_W_a)
        self.U_a = K.variable(initial_U_a)
        self.v_a = K.variable(initial_v_a)
        self.trainable_weights = [self.W_a, self.U_a, self.v_a]
        self.u_a_h = K.dot(self.h, self.U_a)

    def call(self, inputs, mask=None):
        '''
        s,h: batch_size * input_dim
        '''
        s, h = inputs
        check_and_throw_if_fail(K.ndim(s) == 2, "s")
        w_a_s = K.dot(s, self.W_a)
        timesteps = shape(self.u_a_h)[1]
        w_a_s = K.repeat(w_a_s, timesteps)
        e = K.tanh (w_a_s + self.u_a_h)
        e = inner_product(e, self.v_a)    # shape: batch_size, time_steps
        e = K.exp (e)
        e_sum = K.sum(e, -1, keepdims=True)
        a = e / e_sum
        a = K.expand_dims(a)    # to shape: batch_size, time_steps, 1
        c = a * h
        c = K.sum(c, axis=1)
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
        h_tm1 = inputs[-1]
        if self.consume_less == 'gpu':
            x = self.b
            for y, W in zip(inputs[:-1], self.W):
                x += K.dot(y , W)

            x_z = x[:, :self.output_dim]
            x_r = x[:, self.output_dim: 2 * self.output_dim]
            x_h = x[:, 2 * self.output_dim:]

            matrix_inner = K.dot(h_tm1 , self.U[:, :2 * self.output_dim])
            inner_z = matrix_inner[:, :self.output_dim]
            inner_r = matrix_inner[:, self.output_dim: 2 * self.output_dim]
            z = self.inner_activation(x_z + inner_z)
            r = self.inner_activation(x_r + inner_r)
            inner_h = K.dot(r * h_tm1 , self.U[:, 2 * self.output_dim:])
            hh = self.activation(x_h + inner_h)
        else:
            x_z = self.b_z
            x_r = self.b_r
            x_h = self.b_h
            for y, W_z, W_r, W_h in zip (inputs[:-1], self.W_z, self.W_r, self.W_h):
                x_z += K.dot(y, W_z)
                x_r += K.dot(y, W_r)
                x_h += K.dot(y, W_h)
            z = self.inner_activation(x_z + K.dot(h_tm1 , self.U_z))
            r = self.inner_activation(x_r + K.dot(h_tm1, self.U_r))
            hh = self.activation(x_h + K.dot(r * h_tm1, self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        return h

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
        super(RNNLayerBase, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        raise NotImplementedError

    def step(self, x, states, source_context):
        h_tm1 = states[0]    # previous memory
        c = self.attention(h_tm1, source_context)
        h = self.rnn_cell([x, c, h_tm1])
        output = self.mlp_classifier(h)
        return [output, h]

    def build(self, input_shapes):
        raise NotImplementedError

    def get_initial_states(self, x):
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
        check_and_throw_if_fail(len(input_shape) == 2, "input_shape=(nb_samples, time")
        if self.stateful:
            check_and_throw_if_fail(not input_shape[0] , 'If a RNN is stateful, a complete  input_shape must be provided (including batch size).')
            self.states = [K.zeros((input_shape[0], self.rnn_cell.output_dim))]
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(x)    # (samples, timesteps)
        initial_state = K.sum(initial_state, axis=(1,))    # (samples,)
        initial_state = K.expand_dims(initial_state)    # (samples, 1)
        initial_state = K.tile(initial_state, [1, self.output_dim])    # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def call(self, inputs, mask=None):
        # input shape: (nb_samples, time (padded with zeros))
        x, source_context = inputs
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


def RNNDecoderWithBeamSearch(RNNLayerBase):
    '''
    RNN based decoder for prediction, using beam search
    '''
    def __init__(self, max_output_length, beam_size, number_of_output_sequence=1, **kwargs):
        check_and_throw_if_fail(max_output_length > 0, "check_and_throw_if_fail")
        check_and_throw_if_fail(beam_size > 0, "beam_size")
        check_and_throw_if_fail(number_of_output_sequence > 0, "number_of_output_sequence")
        check_and_throw_if_fail(beam_size >= number_of_output_sequence, "number_of_output_sequence")

        self.max_output_length = max_output_length
        self.beam_size = beam_size
        self.number_of_output_sequence = number_of_output_sequence
        super(RNNDecoderWithBeamSearch, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        # two outputs: sequence and score
        return (input_shape[0], self.number_of_output_sequence, self.max_output_length), (input_shape[0], self.number_of_output_sequence)

    def build(self, input_shape):
        check_and_throw_if_fail(len(input_shape) == 2, "input_shape=(nb_samples, source_context_input_dim")
        if self.stateful:
            check_and_throw_if_fail(not input_shape[0] , 'If a RNN is stateful, a complete  input_shape must be provided (including batch size).')
            self.states = [K.zeros((input_shape[0] * self.beam_size , self.rnn_cell.output_dim))]
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(x)    # (samples)
        initial_state = K.repeat_elements(initial_state, self.beam_size)
        initial_state = K.expand_dims(initial_state)    # (samples, 1)
        initial_state = K.tile(initial_state, [1, self.output_dim])    # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def call(self, inputs, mask=None):
        # input shape: (nb_samples, time (padded with zeros))
        x, source_context = inputs
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
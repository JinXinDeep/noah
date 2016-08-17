'''
Created on Jul 16, 2016

@author: lxh5147
'''

from keras import backend as K
from keras.engine import Layer
from .backend import reshape, reverse, inner_product, unpack, top_k
from .utils import check_and_throw_if_fail
from keras.layers import Dense, BatchNormalization
from keras.layers.wrappers import TimeDistributed

from keras import activations, initializations, regularizers, constraints
import numpy as np


class ComposedLayer(Layer):
    '''A layer that employs a set of children layers to complete its call. All the children layers must be created in its constructor.
    '''
    def __init__(self, **kwargs):
        ''' Constructor of a composed layer, which should be called as the last function call of the constructor of its sub class by that sub class to construct its children layers.
        '''
        # Logic copied from layer with small adaption
        if not hasattr(self, 'input_spec'):
            self.input_spec = None
        if not hasattr(self, 'supports_masking'):
            self.supports_masking = False

        self._uses_learning_phase = False

        # these lists will be filled via successive calls
        # to self.add_inbound_node()
        self.inbound_nodes = []
        self.outbound_nodes = []


        self._trainable_weights = []
        self._non_trainable_weights = []
        self._regularizers = []
        self._constraints = {}  # dict {tensor: constraint instance}
        self.built = False

        # these properties should be set by the user via keyword arguments.
        # note that 'input_dtype', 'input_shape' and 'batch_input_shape'
        # are only applicable to input layers: do not pass these keywords
        # to non-input layers.
        allowed_kwargs = {'input_shape',
                          'batch_input_shape',
                          'input_dtype',
                          'name',
                          'trainable',
                          'create_input_layer'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Keyword argument not understood: ' + kwarg

        name = kwargs.get('name')
        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(K.get_uid(prefix))
        self.name = name

        self.trainable = kwargs.get('trainable', True)
        if 'batch_input_shape' in kwargs or 'input_shape' in kwargs:
            # in this case we will create an input layer
            # to insert before the current layer
            if 'batch_input_shape' in kwargs:
                batch_input_shape = tuple(kwargs['batch_input_shape'])
            elif 'input_shape' in kwargs:
                batch_input_shape = (None,) + tuple(kwargs['input_shape'])
            self.batch_input_shape = batch_input_shape
            input_dtype = kwargs.get('input_dtype', K.floatx())
            self.input_dtype = input_dtype
            if 'create_input_layer' in kwargs:
                self.create_input_layer(batch_input_shape, input_dtype)

        self._updates = []
        self._stateful = False
        self._layers = []
        # layers will be constructed in build


    @property
    def updates(self):
        updates = []
        updates += self._updates
        for layer in self._layers:
            if hasattr(layer, 'updates'):
                updates += layer.updates
        return updates

    @property
    def constraints(self):
        cons = {}
        for key, value in self._constraints.items():
            cons[key] = value
        for layer in self._layers:
            for key, value in layer.constraints.items():
                if key in cons:
                    raise Exception('Received multiple constraints '
                                    'for one weight tensor: ' + str(key))
                cons[key] = value
        return cons

    @property
    def regularizers(self):
        regs = []
        regs += self._regularizers
        for layer in self._layers:
            regs += layer.regularizers
        return regs

    @property
    def stateful(self):
        if self._stateful:
            return self._stateful
        return any([(hasattr(layer, 'stateful') and layer.stateful) for layer in self._layers])

    def reset_states(self):
        for layer in self._layers:
            if hasattr(layer, 'reset_states') and getattr(layer, 'stateful', False):
                layer.reset_states()

    @property
    def uses_learning_phase(self):
        '''True if any layer in the graph uses it.
        '''
        if self._uses_learning_phase:
            return self._uses_learning_phase
        layers_learning_phase = any([layer.uses_learning_phase for layer in self._layers])
        regs_learning_phase = any([reg.uses_learning_phase for reg in self.regularizers])
        return layers_learning_phase or regs_learning_phase

    @property
    def trainable_weights(self):
        weights = []
        weights += self._trainable_weights
        for layer in self._layers:
            weights += layer.trainable_weights
        return weights

    @property
    def non_trainable_weights(self):
        weights = []
        weights += self._non_trainable_weights
        for layer in self._layers:
            weights += layer.non_trainable_weights
        return weights

class ReshapeLayer(Layer):
    '''Reshape a tensor to the target shape
    '''

    def __init__(self, target_shape, **kwargs):
        """Constructs a reshape layer
        # Parameters
        ----------
        target_shape : A tuple of int type, representing the target shape of the output tensor.
        """
        self.target_shape = tuple(target_shape)
        super(ReshapeLayer, self).__init__(**kwargs)

    def call(self, x, mask = None):
        return reshape(x, self.target_shape)

    def get_output_shape_for(self, input_shape):
        return self.target_shape

    def get_config(self):
        config = {'target_shape': self.target_shape}
        base_config = super(ReshapeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class BiDirectionalLayer(Layer):
    '''Defines a layer that combines one input sequence from left to right and the other sequence from right to left.
    '''
    def call(self, inputs, mask = None):
        """
        # Parameters
        ----------
        inputs : two input tensor, representing the input sequence from left to right and the one from right to left, respectively. Both have a shape of ..., time_steps, input_dim.
        """
        left_to_right = inputs[0]
        right_to_left = inputs[1]
        ndim = K.ndim(right_to_left)
        # reshape nb_samples, time_steps,  ... -> time_steps,nb_samples,...
        axes = [1, 0] + list(range(2, ndim))
        right_to_left = K.permute_dimensions(right_to_left, axes)
        right_to_left = reverse(right_to_left)
        right_to_left = K.permute_dimensions(right_to_left, axes)
        return K.concatenate([left_to_right, right_to_left], axis = -1)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][:-1] + (input_shapes[0][-1] + input_shapes[1][-1],)

class MLPClassifierLayer(ComposedLayer):
    '''
    Represents a mlp classifier, which consists of several hidden layers followed by a softmax/or sigmoid output layer.
    '''

    def __init__(self, output_dim, hidden_unit_numbers, hidden_unit_activation_functions,
                 output_activation_function = 'softmax', use_sequence_input = True, **kwargs):
        '''
        # Parameters
        ----------
        output_dim: output dimension

        hidden_unit_numbers: the number of hidden units of each hidden layer.
        hidden_unit_activation_functions: the activation function of each hidden layers.
        output_activation_function: activation function for classification, use 'sigmoid' for binary classification.
        use_sequence_input: the last two dimensions of the input has a shape of time_steps, input_dim
        '''
        check_and_throw_if_fail(output_dim > 0, "output_dim")
        check_and_throw_if_fail(len(hidden_unit_numbers) == len(hidden_unit_activation_functions), "hidden_unit_numbers")
        super(MLPClassifierLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.hidden_unit_numbers = hidden_unit_numbers
        self.hidden_unit_activation_functions = hidden_unit_activation_functions
        self.output_activation_function = output_activation_function
        self.use_sequence_input = use_sequence_input
        super(MLPClassifierLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        for hidden_unit_number, hidden_unit_activation_function in zip(self.hidden_unit_numbers, self.hidden_unit_activation_functions):
            dense = Dense(hidden_unit_number, activation = hidden_unit_activation_function)
            if self.use_sequence_input:
                dense = TimeDistributed(dense)
            norm = BatchNormalization(mode = 2)
            self._layers.append(dense)
            self._layers.append(norm)

        dense = Dense(self.output_dim, activation = self.output_activation_function)
        if self.use_sequence_input:
            dense = TimeDistributed(dense)
        self._layers.append(dense)

    def call(self, x, mask = None):
        output = x
        for layer in self._layers:
            output = layer(output)
        return output

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1] + (self.output_dim,)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'use_sequence_input':self.use_sequence_input,
                  'hidden_unit_numbers': self.hidden_unit_numbers,
                  'hidden_unit_activation_functions': self.hidden_unit_activation_functions,
                  'output_activation_function': self.output_activation_function}
        base_config = super(MLPClassifierLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AttentionLayer(Layer):
    '''
    Calculates a weighted sum tensor from the given input tensors, according to http://nlp.ict.ac.cn/Admin/kindeditor/attached/file/20141011/20141011133445_31922.pdf
    '''
    def __init__(self, output_dim, init_W_a = 'glorot_uniform', init_U_a = 'glorot_uniform', init_v_a = 'uniform',
                 W_a_regularizer = None, U_a_regularizer = None, v_a_regularizer = None,
                 W_a_constraint = None, U_a_constraint = None, v_a_constraint = None, **kwargs):
        '''
        # Parameters
        ----------
        output_dim: output dimension of the attention tensor
        '''
        self.output_dim = output_dim
        self.init_W_a = initializations.get(init_W_a)
        self.init_U_a = initializations.get(init_U_a)
        self.init_v_a = initializations.get(init_v_a)

        self.W_a_regularizer = regularizers.get(W_a_regularizer)
        self.U_a_regularizer = regularizers.get(U_a_regularizer)
        self.v_a_regularizer = regularizers.get(v_a_regularizer)

        self.W_a_constraint = constraints.get(W_a_constraint)
        self.U_a_constraint = constraints.get(U_a_constraint)
        self.v_a_constraint = constraints.get(v_a_constraint)

        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shapes):
        '''
        # Parameters
        ----------
        input_shapes: the input shape of s and h, respectively; s with a shape of nb_samples, input_dim while h nb_samples, time_steps, input_dim
        '''

        self.W_a = self.init_W_a((input_shapes[0][-1], self.output_dim))
        self.U_a = self.init_U_a((input_shapes[-1][-1], self.output_dim))
        self.v_a = self.init_v_a((self.output_dim,))
        self.trainable_weights = [self.W_a, self.U_a, self.v_a]

        self.regularizers = []
        if self.W_a_regularizer:
            self.W_a_regularizer.set_param(self.W_a)
            self.regularizers.append(self.W_a_regularizer)
        if self.U_a_regularizer:
            self.U_a_regularizer.set_param(self.U_a)
            self.regularizers.append(self.U_a_regularizer)
        if self.v_a_regularizer:
            self.v_a_regularizer.set_param(self.v_a)
            self.regularizers.append(self.v_a_regularizer)

        self.constraints = {}
        if self.W_a_constraint:
            self.constraints[self.W_a] = self.W_a_constraint
        if self.U_a_constraint:
            self.constraints[self.U_a] = self.U_a_constraint
        if self.v_a_constraint:
            self.constraints[self.v_a] = self.v_a_constraint

    def call(self, inputs, mask = None):
        # s: nb_sample,input_dim
        # h: nb_samples,time_steps, h_input_dim
        s, h = inputs
        U_a_h = K.dot(h, self.U_a)  # nb_samples, time_steps, output_dim
        W_a_s = K.expand_dims(K.dot(s, self.W_a), 1)  # nb_samples, 1, output_dim
        e = K.tanh (W_a_s + U_a_h)  # nb_samples, time_steps, output_dim
        e = inner_product(e, self.v_a)  # nb_samples, time_steps
        e = K.exp (e)
        e_sum = K.sum(e, -1, keepdims = True)  # nb_samples, 1
        a = e / e_sum  # nb_samples, time_steps
        a = K.expand_dims(a)  # nb_samples, time_steps, 1
        c = a * h  # nb_samples, time_steps, h_input_dim
        c = K.sum(c, axis = 1)  # nb_samples, h_input_dim
        return c

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[1][0], input_shapes[1][2])

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init_W_a': self.init_W_a.__name__,
                  'init_U_a': self.init_U_a.__name__,
                  'init_v_a': self.init_v_a.__name__,
                  'W_a_regularizer': self.W_a_regularizer.get_config() if self.W_a_regularizer else None,
                  'U_a_regularizer': self.U_a_regularizer.get_config() if self.U_a_regularizer else None,
                  'v_a_regularizer': self.v_a_regularizer.get_config() if self.v_a_regularizer else None,
                  'W_a_constraint': self.W_a_constraint.get_config() if self.W_a_constraint else None,
                  'U_a_constraint': self.U_a_constraint.get_config() if self.U_a_constraint else None,
                  'v_a_constraint': self.v_a_constraint.get_config() if self.v_a_constraint else None }
        base_config = super(AttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class GRUCell(Layer):
    '''Gated Recurrent Unit - Cho et al. 2014.

    # Arguments
    ----------
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizers: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_Ws: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    # References
    ----------
        - [On the Properties of Neural Machine Translation: Encoder–Decoder Approaches](http://www.aclweb.org/anthology/W14-4012)
        - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/pdf/1412.3555v1.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    '''

    def __init__(self, output_dim,
                 init = 'glorot_uniform', inner_init = 'orthogonal',
                 activation = 'tanh', inner_activation = 'hard_sigmoid', consume_less = 'gpu',
                 W_regularizers = None, U_regularizer = None, b_regularizer = None,
                 **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        if W_regularizers:
            self.W_regularizers = [regularizers.get(W_regularizer) for W_regularizer in  W_regularizers]
        else:
            self.W_regularizers = None
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.consume_less = consume_less

        if self.dropout_Ws and any([not dropout_W is None for dropout_W in self.dropout_Ws]):
            self.uses_learning_phase = True
        super(GRUCell, self).__init__(**kwargs)

    def build(self, input_shapes):
        self.Ws = []
        # multiple inputs supports
        input_dims = [input_shape[1] for input_shape in input_shapes[:-1]]
        if self.consume_less == 'gpu':
            for input_dim in input_dims:
                self.Ws.append (self.inner_init((input_dim, 3 * self.output_dim)))
            self.U = self.inner_init((self.output_dim, 3 * self.output_dim))
            self.b = K.variable(np.hstack((np.zeros(self.output_dim), np.zeros(self.output_dim), np.zeros(self.output_dim))))
            self.trainable_weights = self.Ws + [self.U, self.b]
        else:
            self.W_z = []
            self.W_r = []
            self.W_h = []
            self.Ws = []
            for input_dim in input_dims:
                self.W_z.append(self.inner_init((input_dim, self.output_dim)))
                self.W_r.append(self.inner_init((input_dim, self.output_dim)))
                self.W_h.append(self.inner_init((input_dim, self.output_dim)))
                self.Ws.append (K.concatenate([self.W_z[-1], self.W_r[-1], self.W_h[-1]]))

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
        if self.W_regularizers:
            for W, W_regularizer in zip(self.Ws, self.W_regularizers):
                W_regularizer.set_param(W)
                self.regularizers.append(W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

    def call(self, inputs, mask = None):
        # the last one is previous state
        h_prev = inputs[-1]
        if self.consume_less == 'gpu':
            x = self.b
            for y, W in zip(inputs[:-1], self.Ws):
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
        return  (1 - z) * h_prev + z * hh  # consistent with the formula in the paper

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'activation': self.activation.__name__,
                  'inner_activation': self.inner_activation.__name__,
                  'consume_less':self.consume_less,
                  'W_regularizers': [W_regularizer.get_config() if W_regularizer else None for W_regularizer in self.W_regularizers ] if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                }
        base_config = super(GRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def RNNLayerBase(Layer):

    def __init__(self, rnn_cell, attention, output_embedding, mlp_classifier, stateful = False, **kwargs):
        # TODO: apply drop out to inputs and inner outputs
        self.rnn_cell = rnn_cell
        self.attention = attention
        self.mlp_classifier = mlp_classifier
        self.output_embedding = output_embedding
        self.stateful = stateful
        self.uses_learning_phase = mlp_classifier.uses_learning_phase or rnn_cell.uses_learning_phase or output_embedding.output_embedding or attention.uses_learning_phase
        super(RNNLayerBase, self).__init__(**kwargs)

    def get_initial_states(self, x):
        # build an all-zero tensor of get_shape (samples, output_dim)
        initial_state = K.zeros_like(x)  # (samples, timesteps)
        initial_state = K.sum(initial_state, axis = (1,))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        initial_state = K.tile(initial_state, [1, self.output_dim])  # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def get_output_shape_for(self, input_shape):
        raise NotImplementedError

    def step(self, x, states, source_context):
        h_prev = states[0]  # previous memory
        c = self.attention(h_prev, source_context)
        h = self.rnn_cell([x, c, h_prev])
        output = self.mlp_classifier(h)
        return output, [h]

    def build(self, input_shapes):
        raise NotImplementedError

    def call(self, inputs, mask = None):
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
            # initial states: all-zero tensor of get_shape (output_dim)
            self.states = [None]

    def call(self, inputs, mask = None):
        # input get_shape: (nb_samples, time (padded with zeros))
        x, source_context = inputs
        if self.stateful:
            current_states = self.states
        else:
            current_states = self.get_initial_states(x)
        x = self.embedding(x)
        x = K.permute_dimensions(x, axes = [1, 0, 2])
        input_list = unpack(x)
        successive_states = []
        successive_outputs = []
        for current_input in input_list:
            # TODO: randomly use the real greedy output as the next input
            output, current_states = self.step(current_input, current_states, source_context)
            successive_outputs.append(output)
            successive_states.append(current_states)
        outputs = K.pack(successive_outputs)
        outputs = K.permute_dimensions(outputs, axes = [1, 0, 2])
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
    def __init__(self, max_output_length, beam_size, number_of_output_sequence = 1, eos = None, **kwargs):
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

    def call(self, inputs, mask = None):
        check_and_throw_if_fail(self.beam_size <= self.mlp_classifier.output_dim , "beam_size")
        # input get_shape: (nb_samples, time (padded with zeros))
        x, source_context = inputs
        if K.ndim(x) == 2:
            x = K.squeeze(x, 1)
        # x is the initial input, i.e., bos
        current_state = self.get_initial_states(x)[0]
        current_input = self.embedding(x)
        current_input = K.repeat_elements(current_input, self.beam_size, 0)  #  nb_samples* beam_size, input_dim
        current_state = K.repeat_elements(current_state, self.beam_size, 0)
        output_score = K.cast (K.zeros_like(x), K.common._FLOATX)  # nb_samples
        output_score = K.repeat_elements(output_score, self.beam_size, 0)  # nb_samples*beam_size
        output_score_list = []  # nb_samples, beam_size
        output_label_id_list = []
        prev_output_index_list = []
        for _ in xrange(self.max_output_length):
            output, current_states = self.step(current_input, [current_state], source_context)  # nb_samples*beam_size , output_dim
            current_state = current_states[0]
            score = K.expand_dims(output_score) + K.log(output)  # nb_samples*beam_size, output_dim
            score = K.reshape(score, shape = (-1, self.beam_size, self.mlp_classifier.output_dim))  # nb_samples, beam_size,  output_dim
            top_k_k_score, top_k_k_score_indice = top_k (score, self.beam_size)  # nb_samples, beam_size,  beam_size
            top_k_k_score = K.reshape(top_k_k_score , shape = (-1, self.beam_size * self.beam_size))  # nb_samples, beam_size* beam_size
            top_k_k_score_indice = K.reshape(top_k_k_score_indice , shape = (-1, self.beam_size * self.beam_size))
            # nb_samples, k, k
            top_k_score, top_k_scores_indice = top_k (top_k_k_score, self.beam_size)  #  nb_samples, beam_size
            x = RNNBeamSearchDecoder.gather_per_sample(top_k_k_score_indice, top_k_scores_indice)  # nb_samples, beam_size
            output_score = K.reshape(top_k_score, shape = (-1,))  # nb_samples*beam_size
            output_score_list.append (top_k_score)
            output_label_id_list.append(x)
            prev_output_index = top_k_scores_indice // self.beam_size  # nb_samples, beam_size
            prev_output_index_list.append (prev_output_index)
            current_input = self.embedding(x)  # nb_samples, beam_size, embidding.output_dim
            current_input = K.reshape(current_input, shape = (-1, self.embedding.output_dim))
            # current state:  nb_samples*beam_size, output_dim
            current_state = K.gather (current_state, K.reshape(prev_output_index, (-1,)))
        if self.eos:
            eos = self.eos + K.zeros_like(x)  # b_samples
            eos = K.reshape (K.repeat_elements(eos, self.number_of_output_sequence), shape = (-1, self.number_of_output_sequence))
        return RNNBeamSearchDecoder.get_k_best_from_lattice([output_score_list, output_label_id_list, prev_output_index], self.number_of_output_sequence, eos)

    @staticmethod
    def cond_set(cond, t1, t2):
        r = []
        t1 = K.reshape(t1, shape = (-1,))
        t2 = K.reshape(t1, shape = (-1,))
        cond = K.reshape(cond, shape = (-1,))
        for _c, _1, _2 in zip (unpack(cond), unpack(t1), unpack(t2)):
            r.append(K.switch(_c, _1, _2))
        return K.pack(r)

    @staticmethod
    def get_k_best_from_lattice(lattice, k, eos):
        # from back to front
        for l in lattice:
            l.reverse()
        output_score_list, output_label_id_list, prev_output_index = lattice
        path_list = []
        path_score, output_indice = top_k (output_score_list[0], k)
        for output_score, output_label_id, prev_output_index in zip(output_score_list, output_label_id_list, prev_output_index):
            path_list.append (RNNBeamSearchDecoder.gather_per_sample(output_label_id, output_indice))  # nb_sample, k
            score = RNNBeamSearchDecoder.gather_per_sample(output_score, output_indice)
            if eos:
                cond = K.equal(path_list[-1], eos)
                path_score = K.reshape(RNNBeamSearchDecoder.cond_set(cond, score, path_score), shape = (-1, k))
            output_indice = RNNBeamSearchDecoder.gather_per_sample(prev_output_index, output_indice)
        if eos:
            path_score, output_indice = top_k(path_score, k)  # sort the top k path by default, nb_samples, k
            path_list = [RNNBeamSearchDecoder.gather_per_sample(path, output_indice) for path in path_list]
        path_list = K.permute_dimensions(K.pack(path_list), (1, 2, 0))  # time_steps, nb_samples, k -> nb_samples, k, time_steps
        return path_list, path_score

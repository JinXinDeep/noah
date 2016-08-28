'''
Created on Jul 16, 2016

@author: lxh5147
'''

from keras import backend as K
from keras.engine import Layer
from .backend import reverse, inner_product, unpack, beam_search
from .utils import check_and_throw_if_fail
from keras.layers import  BatchNormalization
from keras.layers.wrappers import Wrapper
from keras.engine import InputSpec

from keras import activations, initializations, regularizers, constraints
import numpy as np
#================================start of overridden layers========================================#
# Copied from keras 1.0.8 and made small adpation so that we can set time steps can be None
class TimeDistributed(Wrapper):
    """This wrapper allows to apply a layer to every
    temporal slice of an input.

    The input should be at least 3D,
    and the dimension of index one will be considered to be
    the temporal dimension.

    Consider a batch of 32 samples, where each sample is a sequence of 10
    vectors of 16 dimensions. The batch input shape of the layer is then `(32, 10, 16)`
    (and the `input_shape`, not including the samples dimension, is `(10, 16)`).

    You can then use `TimeDistributed` to apply a `Dense` layer to each of the 10 timesteps, independently:
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
        # now model.output_shape == (None, 10, 8)

        # subsequent layers: no need for input_shape
        model.add(TimeDistributed(Dense(32)))
        # now model.output_shape == (None, 10, 32)
    ```

    The output will then have shape `(32, 10, 8)`.

    Note this is strictly equivalent to using `layers.core.TimeDistributedDense`.
    However what is different about `TimeDistributed`
    is that it can be used with arbitrary layers, not just `Dense`,
    for instance with a `Convolution2D` layer:

    ```python
        model = Sequential()
        model.add(TimeDistributed(Convolution2D(64, 3, 3), input_shape=(10, 3, 299, 299)))
    ```

    # Arguments
        layer: a layer instance.
    """
    def __init__(self, layer, **kwargs):
        self.supports_masking = True
        super(TimeDistributed, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape = input_shape)]
        child_input_shape = (input_shape[0],) + input_shape[2:]
        if not self.layer.built:
            self.layer.build(child_input_shape)
            self.layer.built = True
        super(TimeDistributed, self).build()

    def get_output_shape_for(self, input_shape):
        child_input_shape = (input_shape[0],) + input_shape[2:]
        child_output_shape = self.layer.get_output_shape_for(child_input_shape)
        timesteps = input_shape[1]
        return (child_output_shape[0], timesteps) + child_output_shape[1:]

    def call(self, X, mask = None):
        input_shape = self.input_spec[0].shape
        if input_shape[0]:
            # batch size matters, use rnn-based implementation
            def step(x, states):
                output = self.layer.call(x)
                return output, []

            _, outputs, _ = K.rnn(step, X,
                                                 initial_states = [])
            y = outputs
        else:
            # no batch size specified, therefore the layer will be able
            # to process batches of any size
            # we can go with reshape-based implementation for performance
            input_length = input_shape[1]
            if not input_length:
                input_length = K.shape(X)[1]
            X = K.reshape(X, (-1,) + input_shape[2:])  # (nb_samples * timesteps, ...)
            y = self.layer.call(X)  # (nb_samples * timesteps, ...)
            # (nb_samples, timesteps, ...)
            output_shape = self.get_output_shape_for(input_shape)
            y = K.reshape(y, (-1, input_length) + output_shape[2:])
        return y

#================================end of overridden layers========================================#

class ComposedLayer(Layer):
    '''A layer that employs a set of children layers to complete its call. All the children layers must be created in its constructor.
    '''
    def __init__(self, **kwargs):
        ''' Constructor of a composed layer, which should be called as the last function call of the constructor of its sub class by that sub class to construct its children layers.
        Note that most of following lines are shamelessly adapted from keras
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

class BiDirectionalLayer(Layer):
    '''Defines a layer that combines one input sequence from left to right and the other sequence from right to left.
    '''
    def __init__(self, time_step_axis = 1, **kwargs):
        self.time_step_axis = time_step_axis  # along which axis to reverse
        super(BiDirectionalLayer, self).__init__(**kwargs)

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
        if self.time_step_axis != 0:
            axes = [self.time_step_axis, 0] + [i for i in range(1, ndim) if i != self.time_step_axis]
            right_to_left = K.permute_dimensions(right_to_left, axes)
        right_to_left = reverse(right_to_left)
        if self.time_step_axis != 0:
            right_to_left = K.permute_dimensions(right_to_left, axes)
        return K.concatenate([left_to_right, right_to_left], axis = -1)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][:-1] + (input_shapes[0][-1] + input_shapes[1][-1],)

    def get_config(self):
        config = {'time_step_axis': self.time_step_axis}
        base_config = super(BiDirectionalLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MLPClassifierLayer(ComposedLayer):
    '''
    Represents a mlp classifier, which consists of several hidden layers followed by a softmax/or sigmoid output layer.
    '''
    def __init__(self, output_layer, hidden_layers = None, use_sequence_input = True, **kwargs):
        '''
        # Parameters
        ----------
        output_layer: output layer for classification, with sigmoid or softmax as activation function
        hidden_layers: hidden layers
        use_sequence_input: the last two dimensions of the input has a shape of time_steps, input_dim, and the time_steps must be specified (required by unpack)
        '''
        self.output_layer = output_layer
        self.hidden_layers = hidden_layers
        self.use_sequence_input = use_sequence_input
        super(MLPClassifierLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.hidden_layers:
            for layer in self.hidden_layers:
                if self.use_sequence_input:
                    layer = TimeDistributed(layer)
                norm = BatchNormalization(mode = 2)
                self._layers.append(layer)
                self._layers.append(norm)


        if self.use_sequence_input:
            layer = TimeDistributed(self.output_layer)
        self._layers.append(layer)

    def call(self, x, mask = None):
        output = x
        for layer in self._layers:
            output = layer(output)
        return output

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1] + (self.output_layer.output_dim,)

    def get_config(self):
        if self.hidden_layers:
            hidden_layers_config = [{'class_name': layer.__class__.__name__,
                            'config': layer.get_config()} for layer in self.hidden_layers]
        else:
            hidden_layers_config = None
        config = {'output_layer': {'class_name': self.output_layer.__class__.__name__, 'config': self.output_layer.get_config()},
                  'hidden_layers': hidden_layers_config,
                  'use_sequence_input':self.use_sequence_input
                  }
        base_config = super(MLPClassifierLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        from keras.utils.layer_utils import layer_from_config
        output_layer = layer_from_config(config.pop('output_layer'))
        hidden_layers_config = config.pop('hidden_layers')
        if hidden_layers_config:
            hidden_layers = [layer_from_config(hidden_layer_config) for hidden_layer_config in hidden_layers_config]
        else:
            hidden_layers = None
        use_sequence_input = config.pop('use_sequence_input')
        return cls(output_layer, hidden_layers, use_sequence_input, **config)

class AttentionLayer(Layer):
    '''
    Calculates a weighted sum tensor from the given input tensors, according to http://nlp.ict.ac.cn/Admin/kindeditor/attached/file/20141011/20141011133445_31922.pdf
    '''
    def __init__(self, attention_context_dim, weights = None, init_W_a = 'glorot_uniform', init_U_a = 'glorot_uniform', init_v_a = 'uniform',
                 W_a_regularizer = None, U_a_regularizer = None, v_a_regularizer = None,
                 W_a_constraint = None, U_a_constraint = None, v_a_constraint = None, **kwargs):
        '''
        # Parameters
        ----------
        attention_context_dim: dimension of the inner attention context vector
        '''
        self.attention_context_dim = attention_context_dim

        self.init_W_a = initializations.get(init_W_a)
        self.init_U_a = initializations.get(init_U_a)
        self.init_v_a = initializations.get(init_v_a)

        self.W_a_regularizer = regularizers.get(W_a_regularizer)
        self.U_a_regularizer = regularizers.get(U_a_regularizer)
        self.v_a_regularizer = regularizers.get(v_a_regularizer)

        self.W_a_constraint = constraints.get(W_a_constraint)
        self.U_a_constraint = constraints.get(U_a_constraint)
        self.v_a_constraint = constraints.get(v_a_constraint)

        self.initial_weights = weights

        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shapes):
        '''
        # Parameters
        ----------
        input_shapes: the input shape of s and h, respectively; s with a shape of nb_samples, input_dim while h nb_samples, time_steps, input_dim
        '''

        self.W_a = self.init_W_a((input_shapes[0][-1], self.attention_context_dim))
        self.U_a = self.init_U_a((input_shapes[-1][-1], self.attention_context_dim))
        self.v_a = self.init_v_a((self.attention_context_dim,))

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

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @staticmethod
    def _calc(s, h, W_a, U_a, v_a, tensors_to_debug = None):
        U_a_h = K.dot(h, U_a)  # nb_samples, time_steps, attention_context_dim
        W_a_s = K.expand_dims(K.dot(s, W_a), 1)  # nb_samples, 1, attention_context_dim
        if tensors_to_debug is not None:
            tensors_to_debug.append(W_a_s)
            tensors_to_debug.append(U_a_h)
        W_U_sum = W_a_s + U_a_h
        if tensors_to_debug is not None:
            tensors_to_debug.append(W_U_sum)
        e = K.tanh (W_U_sum)  # nb_samples, time_steps, attention_context_dim
        if tensors_to_debug is not None:
            tensors_to_debug.append(e)
        e = inner_product(e, v_a)  # nb_samples, time_steps
        if tensors_to_debug is not None:
            tensors_to_debug.append(e)
        e = K.exp (e)
        if tensors_to_debug is not None:
            tensors_to_debug.append(e)
        e_sum = K.sum(e, -1, keepdims = True)  # nb_samples, 1
        if tensors_to_debug is not None:
            tensors_to_debug.append(e_sum)
        a = e / e_sum  # nb_samples, time_steps
        if tensors_to_debug is not None:
            tensors_to_debug.append(a)
        a = K.expand_dims(a)  # nb_samples, time_steps, 1
        c = a * h  # nb_samples, time_steps, h_input_dim
        if tensors_to_debug is not None:
            tensors_to_debug.append(c)
        c = K.sum(c, axis = 1)  # nb_samples, h_input_dim
        return c

    def call(self, inputs, mask = None):
        # s: nb_sample,input_dim
        # h: nb_samples,time_steps, h_input_dim
        s, h = inputs
        return AttentionLayer._calc(s, h, self.W_a, self.U_a, self.v_a)


    def get_output_shape_for(self, input_shapes):
        return (input_shapes[1][0], input_shapes[1][2])

    def get_config(self):
        config = {'attention_context_dim': self.attention_context_dim,
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
        attention_context_dim: dimension of the internal projections and the final output.
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
        - [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](http://www.aclweb.org/anthology/W14-4012)
        - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/pdf/1412.3555v1.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    '''

    def __init__(self, output_dim,
                 init = 'glorot_uniform', inner_init = 'orthogonal',
                 activation = 'tanh', inner_activation = 'hard_sigmoid', consume_less = 'gpu',
                 W_regularizers = None, U_regularizer = None, b_regularizer = None,
                 **kwargs):
        self.attention_context_dim = output_dim
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
                self.Ws.append (self.inner_init((input_dim, 3 * self.attention_context_dim)))
            self.U = self.inner_init((self.attention_context_dim, 3 * self.attention_context_dim))
            self.b = K.variable(np.hstack((np.zeros(self.attention_context_dim), np.zeros(self.attention_context_dim), np.zeros(self.attention_context_dim))))
            self.trainable_weights = self.Ws + [self.U, self.b]
        else:
            self.W_z = []
            self.W_r = []
            self.W_h = []
            self.Ws = []
            for input_dim in input_dims:
                self.W_z.append(self.inner_init((input_dim, self.attention_context_dim)))
                self.W_r.append(self.inner_init((input_dim, self.attention_context_dim)))
                self.W_h.append(self.inner_init((input_dim, self.attention_context_dim)))
                self.Ws.append (K.concatenate([self.W_z[-1], self.W_r[-1], self.W_h[-1]]))

            self.U_z = self.inner_init((self.attention_context_dim, self.attention_context_dim))
            self.U_r = self.inner_init((self.attention_context_dim, self.attention_context_dim))
            self.U_h = self.inner_init((self.attention_context_dim, self.attention_context_dim))
            self.U = K.concatenate([self.U_z, self.U_r, self.U_h])

            self.b_z = K.zeros((self.attention_context_dim,))
            self.b_r = K.zeros((self.attention_context_dim,))
            self.b_h = K.zeros((self.attention_context_dim,))
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

            x_z = x[:, :self.attention_context_dim]
            x_r = x[:, self.attention_context_dim: 2 * self.attention_context_dim]
            x_h = x[:, 2 * self.attention_context_dim:]

            matrix_inner = K.dot(h_prev , self.U[:, :2 * self.attention_context_dim])
            inner_z = matrix_inner[:, :self.attention_context_dim]
            inner_r = matrix_inner[:, self.attention_context_dim: 2 * self.attention_context_dim]
            z = self.inner_activation(x_z + inner_z)
            r = self.inner_activation(x_r + inner_r)
            inner_h = K.dot(r * h_prev , self.U[:, 2 * self.attention_context_dim:])
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
        return (input_shapes[0][0], self.attention_context_dim)

    def get_config(self):
        config = {'attention_context_dim': self.attention_context_dim,
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

def RNNDecoderLayerBase(ComposedLayer):
    '''RNN layer base class, which employs a RNN, an attention mechanism, an embedding, and a MLP classifier to decode a sequence.
    '''
    def __init__(self, rnn_cell, attention, embedding, mlp_classifier, stateful = False, **kwargs):
        # TODO: apply drop out to inputs and inner outputs
        self.rnn_cell = rnn_cell
        self.attention = attention
        self.mlp_classifier = mlp_classifier
        self.embedding = embedding
        self.stateful = stateful
        super(RNNDecoderLayerBase, self).__init__(**kwargs)
        # Since self._layers is set to empty in parent's constructor, we must put the following line after the calling of the parent's constructor
        self._layers = [self.rnn_cell, self.attention, self.mlp_classifier, self.embedding]

    def step(self, x, state, source_context):
        h_prev = state  # previous output
        c = self.attention(h_prev, source_context)
        h = self.rnn_cell([x, c, h_prev])
        output = self.mlp_classifier(h)
        return output, h

    def get_config(self):
        config = {'rnn_cell': {'class_name': self.rnn_cell.__class__.__name__,
                            'config': self.rnn_cell.get_config()},
                  'attention': {'class_name': self.attention.__class__.__name__,
                            'config': self.attention.get_config()},
                  'embedding': {'class_name': self.embedding.__class__.__name__,
                            'config': self.embedding.get_config()},
                  'mlp_classifier': {'class_name': self.mlp_classifier.__class__.__name__,
                            'config': self.mlp_classifier.get_config()},
                  'stateful': self.stateful
                  }
        base_config = super(RNNDecoderLayerBase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        from keras.utils.layer_utils import layer_from_config
        rnn_cell = layer_from_config(config.pop('rnn_cell'))
        attention = layer_from_config(config.pop('attention'))
        embedding = layer_from_config(config.pop('embedding'))
        mlp_classifier = layer_from_config(config.pop('mlp_classifier'))
        stateful = config.pop('stateful')
        return cls(rnn_cell, attention, embedding, mlp_classifier, stateful, **config)

    def build(self, input_shapes):
        raise NotImplementedError


def RNNDecoderLayer(RNNDecoderLayerBase):
    '''Defines a RNN based decoder for training, using the ground truth output
    '''
    def get_output_shape_for(self, input_shapes):
        input_shape, _ = input_shapes
        return (input_shape[0], input_shape[1], self.mlp_classifier.attention_context_dim)

    def build(self, input_shapes):
        input_shape, _ = input_shapes
        if self.stateful:
            check_and_throw_if_fail(not input_shape[0] , 'If a RNN is stateful, a complete  input_shape must be provided (including batch size).')
            self.state = K.zeros((input_shape[0], self.rnn_cell.attention_context_dim))
        else:
            # initial states: all-zero tensor of get_shape (attention_context_dim)
            self.state = None

    def call(self, inputs, mask = None):
        input_x, context = inputs
        if self.stateful:
            current_state = self.states
        else:
            current_state = K.zeros(shape = K.pack([K.shape(input_x)[0], self.rnn_cell.attention_context_dim]))

        input_x = self.embedding(input_x)
        input_x = K.permute_dimensions(input_x, axes = [1, 0, 2])  # shape: time_steps, batch-size, input_dim
        input_x_list = unpack(input_x)
        successive_state_list = []
        successive_output_list = []
        for current_input in input_x_list:
            # TODO: randomly use the real greedy output as the next input_x
            output, current_state = self.step(current_input, current_state, context)
            successive_output_list.append(output)
            successive_state_list.append(current_state)
        output_sequence = K.pack(successive_output_list)
        output_sequence = K.permute_dimensions(output_sequence, axes = [1, 0, 2])
        new_state = successive_state_list[-1]

        if self.stateful:
            # Warning: self.state is shared by all calls on this layer
            self.updates = []
            self.updates.append((self.state, new_state))
        return output_sequence


def RNNDecoderLayerWithBeamSearch(RNNDecoderLayerBase):
    '''Defines a RNN based decoder for prediction, using beam search.
    '''
    def __init__(self, max_output_length, beam_size, **kwargs):
        check_and_throw_if_fail(max_output_length > 0, "max_output_length")
        check_and_throw_if_fail(beam_size > 0, "beam_size")
        self.max_output_length = max_output_length
        self.beam_size = beam_size
        super(RNNDecoderLayerWithBeamSearch, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        # output three tensors: output_label_id_list, prev_output_index_list and output_score_list
        nb_samples = input_shape[0]
        return (self.max_output_length, nb_samples, self.beam_size), \
               (self.max_output_length, nb_samples, self.beam_size), \
               (self.max_output_length, nb_samples, self.beam_size)

    def call(self, inputs, mask = None):
        initial_input, source_context = inputs
        if K.ndim(initial_input) == 2:
            initial_input = K.squeeze(initial_input, 1)
        initial_state = K.zeros(shape = K.pack([K.shape(initial_input)[0], self.rnn_cell.attention_context_dim]))  # (nb_samples, rnn_cell_output_dim)
        return  beam_search(initial_input, initial_state, source_context, self.embedding,
                                         step_func = lambda current_input, current_state, constant_context: self.step(current_input, current_state, constant_context),
                                         beam_size = self.beam_size, max_length = self.max_output_length)

    def get_config(self):
        config = {'max_output_length': self.max_output_length,
                  'beam_size': self.beam_size,
                  }
        base_config = super(RNNDecoderLayerWithBeamSearch, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

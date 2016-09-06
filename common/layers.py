'''
Created on Jul 16, 2016

@author: lxh5147
'''

from keras import backend as K
from keras.engine import Layer
from .backend import reverse, inner_product, beam_search, dot
from .utils import check_and_throw_if_fail
from keras.layers import  BatchNormalization
from keras.layers.wrappers import Wrapper
from keras.engine import InputSpec
from keras import  initializations, regularizers, constraints

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
        self.input_spec = [InputSpec(shape=input_shape)]
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

    def call(self, X, mask=None):
        input_shape = self.input_spec[0].shape
        if input_shape[0]:
            # batch size matters, use rnn-based implementation
            def step(x, states):
                output = self.layer.call(x)
                return output, []

            _, outputs, _ = K.rnn(step, X, initial_states=[])
            y = outputs
        else:
            # no batch size specified, therefore the layer will be able
            # to process batches of any size
            # we can go with reshape-based implementation for performance
            input_length = input_shape[1]
            if not input_length:
                input_length = K.shape(X)[1]
            X = K.reshape(X, (-1,) + input_shape[2:])    # (nb_samples * timesteps, ...)
            y = self.layer.call(X)    # (nb_samples * timesteps, ...)
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
        self._constraints = {}    # dict {tensor: constraint instance}
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
    def __init__(self, time_step_axis=1, **kwargs):
        self.time_step_axis = time_step_axis    # along which axis to reverse
        self.supports_masking = True
        super(BiDirectionalLayer, self).__init__(**kwargs)

    def call(self, inputs, masks=None):
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
        output = K.concatenate([left_to_right, right_to_left], axis=-1)
        if masks is not None:
            mask_left_to_right = masks[0]
            if K.ndim(mask_left_to_right) == K.ndim(left_to_right) - 1:
                K.expand_dims(mask_left_to_right)
                output = mask_left_to_right * output
            else:
                masks = K.concatenate(masks, axis=-1)
                output = masks * output
        return output

    def compute_mask(self, inputs, masks):
        if masks is None:
            return None

        left_to_right = inputs[0]

        mask_left_to_right = masks[0]
        if K.ndim(mask_left_to_right) == K.ndim(left_to_right) - 1:
            return mask_left_to_right
        else:
            return K.concatenate(masks, axis=-1)

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
    def __init__(self, output_layer, hidden_layers=None, **kwargs):
        '''
        # Parameters
        ----------
        output_layer: output layer for classification, with sigmoid or softmax as activation function
        hidden_layers: hidden layers
        '''
        self.output_layer = output_layer
        self.hidden_layers = hidden_layers
        super(MLPClassifierLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.hidden_layers:
            for layer in self.hidden_layers:
                layer.build(input_shape)
                layer.built = True
                input_shape = layer.get_output_shape_for(input_shape)

                norm = BatchNormalization(mode=2)
                norm.build(input_shape)
                norm.built = True

                input_shape = norm.get_output_shape_for(input_shape)
                self._layers.append(layer)
                self._layers.append(norm)

        layer = self.output_layer
        layer.build(input_shape)
        layer.built = True

        self._layers.append(layer)

        super(MLPClassifierLayer, self).build(input_shape)

    def call(self, x, mask=None):
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
                  'hidden_layers': hidden_layers_config
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
        return cls(output_layer, hidden_layers, **config)

class AttentionLayer(Layer):
    '''
    Calculates a weighted sum tensor from the given input tensors, according to http://nlp.ict.ac.cn/Admin/kindeditor/attached/file/20141011/20141011133445_31922.pdf
    '''
    def __init__(self, attention_context_dim, weights=None, init_W_a='glorot_uniform', init_U_a='glorot_uniform', init_v_a='uniform',
                 W_a_regularizer=None, U_a_regularizer=None, v_a_regularizer=None,
                 W_a_constraint=None, U_a_constraint=None, v_a_constraint=None, **kwargs):
        '''
        # Parameters
        ----------
        attention_context_dim: dimension of the inner attention context vector
        '''
        self.supports_masking = True

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

        super(AttentionLayer, self).build(input_shapes)

    @staticmethod
    def _calc(s, h, W_a, U_a, v_a, mask_h=None, tensors_to_debug=None):
        U_a_h = dot(h, U_a)    # nb_samples, time_steps, attention_context_dim
        W_a_s = K.expand_dims(dot(s, W_a), 1)    # nb_samples, 1, attention_context_dim
        if tensors_to_debug is not None:
            tensors_to_debug.append(W_a_s)
            tensors_to_debug.append(U_a_h)

        W_U_sum = W_a_s + U_a_h
        if tensors_to_debug is not None:
            tensors_to_debug.append(W_U_sum)

        e = K.tanh (W_U_sum)    # nb_samples, time_steps, attention_context_dim
        if tensors_to_debug is not None:
            tensors_to_debug.append(e)

        e = inner_product(e, v_a)    # nb_samples, time_steps
        if tensors_to_debug is not None:
            tensors_to_debug.append(e)

        e = K.exp (e)    # nb_samples, time_steps
        if mask_h:    # nb_samples, time_steps
            e = e * mask_h

        if tensors_to_debug is not None:
            tensors_to_debug.append(e)

        e_sum = K.sum(e, -1, keepdims=True)    # nb_samples, 1
        if tensors_to_debug is not None:
            tensors_to_debug.append(e_sum)

        a = e / e_sum    # nb_samples, time_steps
        if tensors_to_debug is not None:
            tensors_to_debug.append(a)

        a = K.expand_dims(a)    # nb_samples, time_steps, 1
        c = a * h    # nb_samples, time_steps, h_input_dim
        if tensors_to_debug is not None:
            tensors_to_debug.append(c)

        c = K.sum(c, axis=1)    # nb_samples, h_input_dim
        return c

    def call(self, inputs, masks=None):
        # s: nb_sample,input_dim
        # h: nb_samples,time_steps, h_input_dim
        s, h = inputs
        mask_h = None if masks is None else masks[0]
        return AttentionLayer._calc(s, h, self.W_a, self.U_a, self.v_a, mask_h)

    def compute_mask(self, inputs, masks):
        return None

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

class RNNDecoderLayerBase(ComposedLayer):
    '''RNN layer decoder base class, which employs a rnn cell (re-use those defined by keras, such as GRU and LSTM), an attention mechanism, an embedding, and a MLP classifier to decode a sequence.
    '''
    def __init__(self, rnn_cell, attention, embedding, **kwargs):
        self.rnn_cell = rnn_cell
        self.attention = attention
        self.embedding = embedding
        super(RNNDecoderLayerBase, self).__init__(**kwargs)

    def step(self, x, states, source_context):
        current_state = states[0]    # previous output
        # including current input as part of the input of attention
        attention_input = K.concatenate([x, current_state])
        c = self.attention.call([attention_input, source_context])
        # input of rnn_cell includes current attention
        rnn_cell_step_input = K.concatenate([x, c])
        processed_rnn_cell_step_input = K.squeeze(self.rnn_cell.preprocess_input(K.expand_dims(rnn_cell_step_input, 1)), 1)
        h, _ = self.rnn_cell.step(processed_rnn_cell_step_input, states=states)
        return h, [h]

    def build(self, input_shapes):
        # build the layers manually, since we are going to use these layers on non-keras tensors, which will otherwise throw exception
        x_shape, source_context_shape = input_shapes
        attention_input_shapes = [(x_shape[0], self.embedding.output_dim + self.rnn_cell.output_dim), source_context_shape]
        self.attention.build(input_shapes=attention_input_shapes)
        self.attention.built = True

        attention_output_dim = self.attention.get_output_shape_for(attention_input_shapes)[-1]
        # rnn cell requires a 3D shape, and use the last as the input_dim
        rnn_cell_input_shape = (x_shape[0], None, self.embedding.output_dim + attention_output_dim)
        self.rnn_cell.build(rnn_cell_input_shape)
        self.rnn_cell.built = True

        self._layers = [self.attention, self.rnn_cell, self.embedding]

        super(RNNDecoderLayerBase, self).build(input_shapes)

    def call(self, inputs, mask=None):
        raise NotImplementedError

    def get_config(self):
        config = {'rnn_cell': {'class_name': self.rnn_cell.__class__.__name__,
                            'config': self.rnn_cell.get_config()},
                  'attention': {'class_name': self.attention.__class__.__name__,
                            'config': self.attention.get_config()},
                  'embedding': {'class_name': self.embedding.__class__.__name__,
                            'config': self.embedding.get_config()}
                  }
        base_config = super(RNNDecoderLayerBase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects={}):
        from keras.utils.layer_utils import layer_from_config
        rnn_cell = layer_from_config(config.pop('rnn_cell'), custom_objects)
        attention = layer_from_config(config.pop('attention'), custom_objects)
        embedding = layer_from_config(config.pop('embedding'), custom_objects)
        return cls(rnn_cell, attention, embedding, **config)

class RNNDecoderLayer(RNNDecoderLayerBase):
    '''Defines a RNN decoder for training, using the ground truth output
    '''
    def get_output_shape_for(self, input_shapes):
        input_shape, _ = input_shapes
        return (input_shape[0], input_shape[1], self.rnn_cell.output_dim)

    def call(self, inputs, masks=None):
        input_x, context = inputs

        input_x_mask = None
        context_mask = None
        if masks:
            input_x_mask = masks[0]
            context_mask = masks[1]


        input_x = self.embedding(input_x)

        if self.stateful:
            initial_states = self.rnn_cell.states
        else:
            initial_states = self.rnn_cell.get_initial_states(input_x)

        constants = self.rnn_cell.get_constants(input_x)

        last_output, outputs, states = K.rnn(lambda x, states: self.step(x, states, context),
                                             input_x,
                                             initial_states,
                                             go_backwards=self.rnn_cell.go_backwards,
                                             masks=input_x_mask,
                                             constants=constants,
                                             unroll=self.rnn_cell.unroll)

        if self.rnn_cell.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.rnn_cell.states[i], states[i]))

        if self.rnn_cell.return_sequences:
            return outputs
        else:
            return last_output

class RNNDecoderLayerWithBeamSearch(RNNDecoderLayerBase):
    '''Defines a RNN based decoder for prediction, using beam search.
    '''
    def __init__(self, max_output_length, beam_size, rnn_cell, attention, embedding, mlp_classifier, **kwargs):
        check_and_throw_if_fail(max_output_length > 0, "max_output_length")
        check_and_throw_if_fail(beam_size > 0, "beam_size")
        self.mlp_classifier = mlp_classifier
        self.max_output_length = max_output_length
        self.beam_size = beam_size
        super(RNNDecoderLayerWithBeamSearch, self).__init__(rnn_cell, attention, embedding, **kwargs)

    def build(self, input_shapes):
        # if nb_samples is not None, we will need to update the nb_samples to nb_samples*beam_size
        input_shapes = [input_shape if input_shape[0] is None or input_shape[0] == -1  else [input_shape[0] * self.beam_size] + input_shape[1:]  for input_shape in input_shapes]
        # build the layers manually, since we are going to use these layers on non-keras tensors, which will otherwise throw exception
        super(RNNDecoderLayerWithBeamSearch, self).build(input_shapes)
        x_shape, _ = input_shapes
        mlp_classifier_input_shape = (x_shape[0], self.rnn_cell.output_dim)
        self.mlp_classifier.build(mlp_classifier_input_shape)
        self.mlp_classifier.built = True
        self._layers.append(self.mlp_classifier)

    def get_output_shape_for(self, input_shapes):
        # output three tensors: output_label_id_list, prev_output_index_list and output_score_list
        nb_samples = input_shapes[0][0]
        # returning a list instead of a tuple of tensor shapes, required by keras
        return [(self.max_output_length, nb_samples, self.beam_size), \
               (self.max_output_length, nb_samples, self.beam_size), \
               (self.max_output_length, nb_samples, self.beam_size)]

    def call(self, inputs, mask=None):
        initial_input, source_context = inputs
        if K.ndim(initial_input) == 2:
            initial_input = K.squeeze(initial_input, 1)

        initial_input = self.embedding(initial_input)
        # initial_input is 2D tensor: nb_samples, input_dim, convert to 3D tensor: nb_samples,1,input_dim
        initial_state = self.rnn_cell. get_initial_states(K.expand_dims(initial_input, 1))[0]

        # initial_input is 2D tensor, 3D tensor is required
        constants = self.rnn_cell.get_constants(K.expand_dims(initial_input, 1))

        def step(current_input, current_state, constant_context):
            rnn_output, _ = self.step(current_input, [current_state] + constants, constant_context)
            classifier_output = self.mlp_classifier.call(rnn_output)
            return classifier_output, rnn_output

        return  beam_search(initial_input, initial_state,
                            source_context, self.embedding,
                            step_func=step,
                            beam_size=self.beam_size, max_length=self.max_output_length)

    def compute_mask(self, input_tensors, input_masks):
        # mask is not supported, ignore
        return [None, None, None]

    @classmethod
    def from_config(cls, config, custom_objects={}):
        from keras.utils.layer_utils import layer_from_config
        mlp_classifier = layer_from_config(config.pop('mlp_classifier'), custom_objects)
        max_output_length = config.pop('max_output_length')
        beam_size = config.pop('beam_size')
        # TODO: remove repeated code
        rnn_cell = layer_from_config(config.pop('rnn_cell'), custom_objects)
        attention = layer_from_config(config.pop('attention'), custom_objects)
        embedding = layer_from_config(config.pop('embedding'), custom_objects)
        return cls(max_output_length, beam_size, rnn_cell, attention, embedding, mlp_classifier, **config)

    def get_config(self):
        config = {'mlp_classifier': {'class_name': self.mlp_classifier.__class__.__name__,
                            'config': self.mlp_classifier.get_config()},
                  'max_output_length': self.max_output_length,
                  'beam_size': self.beam_size,
                  }
        base_config = super(RNNDecoderLayerWithBeamSearch, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

'''
Created on Sep 4, 2016

@author: lxh5147
'''
import keras.backend as K
from keras.models import Model
from keras.layers import Input
from  keras.engine.training import collect_trainable_weights

if K._BACKEND == 'tensorflow':

    import tensorflow as tf

    def _clone_input(x):
        return Input(shape = K.int_shape(x)[1:], dtype = x.dtype)

    # inputs, targets, weights -> model0: inputs, targets, weights + model1: inputs, targets, weights + ...
    def _split(x, n):
        batch_size = len(x) / n
        parts = []
        for i in range(n - 1):
            part = x[i * batch_size:i * batch_size + batch_size]
            parts.append(part)
        parts.append(x [(n - 1) * batch_size:])
        return parts

    def _expand_for_multiple_models(inputs, n):
        # nb_input * nb_model
        input_model_list = [_split(x, n) for x in inputs]
        model_input_list = []
        for model_id in range(n):
            model_input = []
            for input_id in range(len(inputs)):
                model_input.append(input_model_list[input_id][model_id])
            model_input_list.append(model_input)
        return model_input_list

    def _get_averaged_updates(trainable_weights, training_updates):
        training_updates_merged = dict()
        for p, new_p in training_updates:
            if p in training_updates_merged:
                v = training_updates_merged[p]
                v.append(new_p)
            else:
                training_updates_merged[p] = [new_p]
        averaged_training_updates = []
        for p, new_p_list in training_updates_merged.items():
            averaged_training_updates.append(K.update(p, sum(new_p_list) / len(new_p_list)))
        return averaged_training_updates

    def convert_to_model_with_parallel_training(model, devices):
        assert not model.state_updates

        models = [model]

        for _ in range(1, len(devices)):
            # split inputs

            cloned_inputs = [_clone_input(i) for i in model.inputs]
            cloned_outputs = model(cloned_inputs)
            models.append(Model(input = cloned_inputs, output = cloned_outputs))

        def _compile(optimizer, loss, metrics = [], loss_weights = None,
                sample_weight_mode = None, **kwargs):
            model._function_kwargs = kwargs
            for model in models:
                model.compile(optimizer, loss, metrics, loss_weights, sample_weight_mode)

        def _make_train_function():
            inputs = []
            for model in models:
                inputs += model.inputs
            for model in models:
                inputs += model.targets
            for model in models:
                inputs += model.sample_weights

            if model.uses_learning_phase and type(K.learning_phase()) is not int:
                inputs .append (K.learning_phase())

            # training_updates
            trainable_weights = collect_trainable_weights(model)
            training_updates = None

            # temporally override K.update
            K_update = K.update
            def _update(x, new_x):
                return (x, new_x)
            K.update = _update
            for device in devices:
                model = next(models)
                with tf.device(device):
                    cur_training_updates = model.optimizer.get_updates(trainable_weights, model.constraints, model.total_loss)
                    if training_updates is None:
                        training_updates = cur_training_updates
                    else:
                        training_updates += cur_training_updates
            # restore K.update
            K.update = K_update

            training_updates = _get_averaged_updates(trainable_weights, training_updates)
            # weights will be updated as the averaged weights
            updates = training_updates

            outputs = []

            for model in models:
                outputs.append(model.total_loss)
                outputs.append(model.metrics_tensors)

            # returns loss and metrics. Updates weights at each call.
            f = K.function(inputs,
                              outputs = outputs,
                              updates = updates,
                              **model._function_kwargs)

            def _f (inputs):
                n = len(devices)
                expanded_inputs = []

                start = 0
                end = len(model.inputs)
                expanded_inputs += _expand_for_multiple_models (inputs[start:end], n)

                start = end
                end = start + len(model.targets)
                expanded_inputs += _expand_for_multiple_models(inputs[start:end], n)

                if model.sample_weights:
                    start = end
                    end = start + len(model.sample_weights)
                    expanded_inputs = _expand_for_multiple_models (inputs[start: end ], n)

                return f(expanded_inputs + inputs[:end])

            return _f

        model.compile = _compile
        model._make_train_function = _make_train_function

        return model

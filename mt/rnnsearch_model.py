'''
Created on Jul 15, 2016
NMT translation model
@author: lxh5147
'''

from keras.models import Model
from keras.layers import Input, GRU, Embedding, Dense, Dropout
import keras.backend as K

from common import get_k_best_from_lattice, BiDirectionalLayer, AttentionLayer, RNNDecoderLayer, RNNDecoderLayerWithBeamSearch, MLPClassifierLayer, TimeDistributed, convert_to_model_with_parallel_training
import numpy as np
from nltk.translate.bleu_score import corpus_bleu


def build_rnn_search_model(source_vacabuary_size,
                           source_embedding_dim,
                           source_initia_embedding,
                           encoder_rnn_output_dim_list,
                           attention_context_dim,
                           decoder_rnn_output_dim,
                           decoder_rnn_output_dropout_rate,
                           target_vacabuary_size,
                           target_embedding_dim,
                           target_initia_embedding,
                           decoder_hidden_unit_numbers,
                           decoder_hidden_unit_activation_functions,
                           optimizer = 'rmsprop',
                           beam_search_max_output_length,
                           beam_size,
                           devices = None):

    # TODO: support weight regularizer

    source_word = Input((None,), dtype = 'int32')
    source_word_mask = Input((None,), dtype = 'int32')

    # source_word = trim_right_padding(source_word)
    source_embedding = Embedding(source_vacabuary_size, source_embedding_dim, weights = [source_initia_embedding])
    # apply embedding
    encoder_output = source_embedding(source_word)

    # multiple bi-directional rnn layers
    for  encoder_rnn_output_dim in  encoder_rnn_output_dim_list:
        recurrent_left_to_right = GRU(encoder_rnn_output_dim, return_sequences = True)
        recurrent_right_to_left = GRU(encoder_rnn_output_dim, return_sequences = True, go_backwards = True)
        h1 = recurrent_left_to_right(encoder_output, source_word_mask)
        h2 = recurrent_right_to_left(encoder_output, source_word_mask)
        encoder_output = BiDirectionalLayer()([h1, h2])

    # the output of the last bi-directional RNN layer is the source context
    source_context = encoder_output
    # attention
    attention = AttentionLayer(attention_context_dim = attention_context_dim)

    # decoder
    decoder_input_sequence = Input((None,), dtype = 'int32')  # starting with bos
    decoder_input_sequence_mask = Input((None,), dtype = 'int32')

    decoder_rnn_cell = GRU(decoder_rnn_output_dim, return_sequences = True)
    target_embedding = Embedding(target_vacabuary_size, target_embedding_dim, weights = [target_initia_embedding])

    rnn_decoder = RNNDecoderLayer(decoder_rnn_cell, attention, target_embedding)

    rnn_decoder_output = rnn_decoder([decoder_input_sequence, source_context], [decoder_input_sequence_mask, source_word_mask])
    rnn_decoder_output_dropout = Dropout(decoder_rnn_output_dropout_rate)
    rnn_decoder_output = rnn_decoder_output_dropout(rnn_decoder_output)

    mlp_classifier_hidden_layers = []
    for decoder_hidden_unit_number, decoder_hidden_unit_activation_function in zip(decoder_hidden_unit_numbers, decoder_hidden_unit_activation_functions):
        layer = Dense(decoder_hidden_unit_number, activation = decoder_hidden_unit_activation_function)
        mlp_classifier_hidden_layers.append(layer)

    mlp_classifier_output_layer = Dense(output_dim = target_vacabuary_size, activation = 'softmax')

    mlp_classifier = MLPClassifierLayer(mlp_classifier_hidden_layers, mlp_classifier_output_layer)

    time_distributed_mlp_classifier = TimeDistributed(mlp_classifier)
    time_distributed_mlp_classifier_output = time_distributed_mlp_classifier(rnn_decoder_output, mask = decoder_input_sequence_mask)
    # output and its mask will will be used to generate proper loss function by the optimizer
    rnn_search_model = Model(input = [source_word, source_word_mask, decoder_input_sequence, decoder_input_sequence_mask], output = time_distributed_mlp_classifier_output)
    # training with multiple devices
    if devices:
        rnn_search_model = convert_to_model_with_parallel_training(rnn_search_model, devices)

    # TODO: try other loss, such as importance sampling based loss, e.g., sampled_softmax_loss (this will need to extend Keras model, which assumes that the loss function does not hold any trainable parameters
    # TODO: see if we need to apply constrains on gradient
    rnn_search_model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    beam_search_initial_input = Input(get_shape = (1,))
    rnn_decoder_with_beam_search = RNNDecoderLayerWithBeamSearch(beam_search_max_output_length, beam_size, decoder_rnn_cell, attention, target_embedding, mlp_classifier)

    # use average state to initialize rnn decoder init state
    beam_search_output_lattice = rnn_decoder_with_beam_search([beam_search_initial_input, source_context])
    rnn_search_runtime_model = Model(input = [source_word, source_word_mask, beam_search_initial_input], output = beam_search_output_lattice)

    return (rnn_search_model, rnn_search_runtime_model)

def train(rnn_search_model,
          generator,
          samples_per_epoch,
          nb_epoch,
          verbose = 1,
          callbacks = [],
          validation_data = None,
          nb_val_samples = None,
          class_weight = {},
          max_q_size = 10,
          nb_worker = 1,
          pickle_safe = False):
    return rnn_search_model.fit_generator(generator,
                                          samples_per_epoch, nb_epoch,
                                          verbose,
                                          callbacks,
                                          validation_data, nb_val_samples,
                                          class_weight,
                                          max_q_size, nb_worker, pickle_safe)

def _build_predict_func(rnn_search_runtime_model, eos = None):
    if rnn_search_runtime_model.uses_learning_phase:
        inputs = rnn_search_runtime_model.inputs + [K.learning_phase()]
    else:
        inputs = rnn_search_runtime_model.inputs
    lattice = rnn_search_runtime_model.outputs
    outputs = get_k_best_from_lattice(lattice, k = 1, eos = eos)
    return K.function(inputs, outputs)

def _predict_loop(model, f, ins, batch_size = 32, verbose = 0):
    nb_sample = len(ins[0])
    outs = []
    if verbose == 1:
        progbar = model.Progbar(target = nb_sample)
    batches = model.make_batches(nb_sample, batch_size)
    index_array = np.arange(nb_sample)
    for _, (batch_start, batch_end) in enumerate(batches):
        batch_ids = index_array[batch_start:batch_end]
        if type(ins[-1]) is float:
            # do not slice the training phase flag
            ins_batch = model.slice_X(ins[:-1], batch_ids) + [ins[-1]]
        else:
            ins_batch = model.slice_X(ins, batch_ids)

        batch_outs = f(ins_batch)
        outs += batch_outs
        if verbose == 1:
            progbar.update(batch_end)

    return outs

def predict(rnn_search_runtime_model, x, bos, batch_size = 32, verbose = 1, eos = None):
    f = _build_predict_func(rnn_search_runtime_model, eos)
    # get all outputs
    initial_input_val = [bos for _ in range(len(x))]
    if rnn_search_runtime_model.uses_learning_phase:
        ins = x + initial_input_val + [0.]
    else:
        ins = x + initial_input_val

    outputs_val = _predict_loop(rnn_search_runtime_model,
                          f,
                          ins, batch_size,
                          verbose)

    # nb_samples, k, time_steps -> nb_samples, time_steps
    outputs_val = np.squeeze(outputs_val, axis = (1,))
    return outputs_val

def evaluate(rnn_search_runtime_model, x, y, bos, batch_size = 32, verbose = 1, sample_weight = None, eos = None):
    outputs_val = predict(rnn_search_runtime_model, x, bos, batch_size, verbose, eos)
    # calculate BLEU on multiple reference
    return _calc_bleu(outputs_val, y, sample_weight, bos, eos)

def _calc_bleu(predict, reference, sample_weight = None, bos = None, eos = None):
    # predict: nb_samples, time_steps
    # reference nb_samples, nb_references, time_steps
    # sample_weight: None or nb_samples

    def norm_sentence(sentence, bos, eos):
        # remove bos and eos from a sentence
        return [ token_id for token_id in sentence if token_id != bos and token_id != eos ]

    predict_sentence_list = [norm_sentence(sentence, bos, eos) for sentence in predict]
    reference_sentence_list = [ [norm_sentence(sentence, bos, eos) for sentence in reference_sentences] for reference_sentences in reference ]
    # TODO: if it is a real concern to introduce dependency on nltk, we will re-implement BLEU
    return corpus_bleu(list_of_references = reference_sentence_list,
                        hypotheses = predict_sentence_list,
                        weights = (0.25, 0.25, 0.25, 0.25),  # from 1-gram to 4-gram
                        smoothing_function = None)


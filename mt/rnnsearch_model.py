'''
Created on Jul 15, 2016
Implements the end-to-end ASR system introduced by: Listen, Attend and Spell : https://arxiv.org/pdf/1508.01211v2
@author: lxh5147
'''
from keras.layers.recurrent import GRU
from keras.models import Model
from keras.layers import Input
from keras.layers.embeddings import Embedding
from ..common import  BiDirectionalLayer, AttentionLayer, GRUCell, RNNLayer, RNNBeamSearchDecoder, MLPClassifierLayer, categorical_crossentropy_ex


def build_model(max_timesteps, source_vacabuary_size, source_embedding_dim, source_initia_embedding, recurrent_input_lengths,
                 target_vacabuary_size, target_embedding_dim,
                 target_initia_embedding, recurrent_output_dim, max_output_length, output_dim, hidden_unit_numbers, hidden_unit_activation_functions,
                 max_output_length, beam_size, number_of_output_sequence, eos):
    source_word = Input(shape=(max_timesteps,))
    input_embedding = Embedding(source_vacabuary_size, source_embedding_dim, weights=[source_initia_embedding])
    output = input_embedding(source_word)
    # listen: recurrent Layers
    for recurrent_input_length, recurrent_output_dim in zip(recurrent_input_length, recurrent_output_dim):
        recurrent_left_to_right = GRU(recurrent_output_dim, return_sequences=True)
        recurrent_right_to_left = GRU(recurrent_output_dim, return_sequences=True, go_backwards=True)
        h1 = recurrent_left_to_right(output)
        h2 = recurrent_right_to_left(output)
        output = BiDirectionalLayer()(h1, h2)
    source_context = output
    # attention
    attention = AttentionLayer()
    # speller
    rnn_cell = GRUCell(recurrent_output_dim)
    output_embedding = Embedding(target_vacabuary_size, target_embedding_dim, weights=[target_initia_embedding])
    mlp_classifier = MLPClassifierLayer(output_dim, hidden_unit_numbers, hidden_unit_activation_functions)
    output_true = Input(shape=(max_output_length,), dtype='int32')
    output = RNNLayer(rnn_cell, attention, output_embedding, mlp_classifier)(output_true, source_context)
    rnn_search = Model(input=[source_word, output_true], output=output)
    # TODO: try advanced loss function based on negative sampling
    rnn_search.compile(optimizer='rmsprop', loss=categorical_crossentropy_ex, metrics=['accuracy'])
    bos = Input(shape=(1,))
    decoder = RNNBeamSearchDecoder(rnn_cell, attention, output_embedding, mlp_classifier)(output_true,
                            source_context, max_output_length, beam_size, number_of_output_sequence, eos)
    pathes, path_scores = decoder (bos, source_context)
    rnn_search_runtime = Model(input=[source_word, bos], output=[pathes, path_scores])

    return (rnn_search, rnn_search_runtime)

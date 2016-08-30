'''
Created on Jul 15, 2016
Implements the end-to-end ASR system introduced by: Listen, Attend and Spell : https://arxiv.org/pdf/1508.01211v2
@author: lxh5147
'''
from keras import backend as K
from keras.layers.recurrent import GRU
from keras.models import Model
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.layers.core import Flatten, Reshape
from ..common import check_and_throw_if_fail, trim_right_padding, BiDirectionalLayer, AttentionLayer, RNNDecoderLayerWithBeamSearch, GRUCell, RNNDecoderLayer, MLPClassifierLayer, categorical_crossentropy_ex


def build_rnn_search_model(max_timesteps, input_dim, recurrent_input_lengths,
                 target_vacabuary_size, target_embedding_dim,
                 target_initia_embedding, recurrent_output_dim, max_output_length, output_dim, hidden_unit_numbers, hidden_unit_activation_functions,
                 max_output_length, beam_size, number_of_output_sequence):
    spectrogram = Input(get_shape=(max_timesteps, input_dim))
    output = trim_right_padding (spectrogram)
    # listen: recurrent Layers
    for recurrent_input_length, recurrent_output_dim in zip(recurrent_input_length, recurrent_output_dim):
        if recurrent_input_length > 1 :
            timesteps, input_dim = K.int_shape(output)[1:]
            check_and_throw_if_fail(timesteps % recurrent_input_length == 0, "timesteps")
            output = Flatten()(output)
            output = Reshape(target_shape={timesteps / recurrent_input_length, input_dim * recurrent_input_length})(output)
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
    output_true = Input(get_shape=(max_output_length,), dtype='int32')
    output = RNNDecoderLayer(rnn_cell, attention, output_embedding, mlp_classifier)(output_true, source_context)
    las = Model(input=[spectrogram, output_true], output=output)
    # TODO: try advanced loss function based on negative sampling
    las.compile(optimizer='rmsprop', loss=categorical_crossentropy_ex, metrics=['accuracy'])
    bos = Input(get_shape=(1,))
    decoder = RNNDecoderLayerWithBeamSearch(rnn_cell, attention, output_embedding, mlp_classifier)(output_true, source_context, max_output_length, beam_size, number_of_output_sequence)
    pathes, path_scores = decoder (bos, source_context)
    las_runtime = Model(input=[spectrogram, bos], output=[pathes, path_scores])
    return (las, las_runtime)

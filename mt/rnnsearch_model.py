'''
Created on Jul 15, 2016
NMT translation model
@author: lxh5147
'''

from keras.models import Model
from keras.layers import Input, GRU, Embedding, Dense

from ..common import  BiDirectionalLayer, AttentionLayer, RNNDecoderLayer, RNNDecoderLayerWithBeamSearch, MLPClassifierLayer, TimeDistributed, categorical_crossentropy_ex


def build_rnn_search_model(source_vacabuary_size, source_embedding_dim, source_initia_embedding,
                 encoder_rnn_output_dim_list,
                 attention_context_dim,
                 decoder_rnn_output_dim,
                 target_vacabuary_size, target_embedding_dim, target_initia_embedding,
                 decoder_hidden_unit_numbers, decoder_hidden_unit_activation_functions,
                 optimizer='rmsprop',
                 beam_search_max_output_length, beam_size):
    source_word = Input((None,), dtype='int32')
    # source_word = trim_right_padding(source_word)
    source_embedding = Embedding(source_vacabuary_size, source_embedding_dim, weights=[source_initia_embedding])
    # apply embedding
    encoder_output = source_embedding(source_word)
    # multiple bi-directional rnn layers
    for  encoder_rnn_output_dim in  encoder_rnn_output_dim_list:
        recurrent_left_to_right = GRU(encoder_rnn_output_dim, return_sequences=True)
        recurrent_right_to_left = GRU(encoder_rnn_output_dim, return_sequences=True, go_backwards=True)
        h1 = recurrent_left_to_right(encoder_output)
        h2 = recurrent_right_to_left(encoder_output)
        encoder_output = BiDirectionalLayer()(h1, h2)
    # the output of the last bi-directional RNN layer is the source context
    source_context = encoder_output
    # attention
    attention = AttentionLayer(attention_context_dim=attention_context_dim)

    # decoder
    decoder_rnn_cell = GRU(decoder_rnn_output_dim, return_sequences=True)
    target_embedding = Embedding(target_vacabuary_size, target_embedding_dim, weights=[target_initia_embedding])
    decoder_input_sequence = Input((None,), dtype='int32')    # starting with bos
    rnn_decoder = RNNDecoderLayer(decoder_rnn_cell, attention, target_embedding)
    rnn_decoder_output = rnn_decoder([decoder_input_sequence, source_context])

    mlp_classifier_hidden_layers = []
    for decoder_hidden_unit_number, decoder_hidden_unit_activation_function in zip(decoder_hidden_unit_numbers, decoder_hidden_unit_activation_functions):
        layer = Dense(decoder_hidden_unit_number, activation=decoder_hidden_unit_activation_function)
        mlp_classifier_hidden_layers.append(layer)

    mlp_classifier_output_layer = Dense(output_dim=target_vacabuary_size, activation='softmax')

    mlp_classifier = MLPClassifierLayer(mlp_classifier_hidden_layers, mlp_classifier_output_layer)

    time_distributed_mlp_classifier = TimeDistributed(mlp_classifier)
    time_distributed_mlp_classifier_output = time_distributed_mlp_classifier(rnn_decoder_output)


    rnn_search_model = Model(input=[source_word, decoder_input_sequence], output=time_distributed_mlp_classifier_output)
    # TODO: try advanced loss function based on negative sampling
    rnn_search_model.compile(optimizer=optimizer, loss=categorical_crossentropy_ex, metrics=['accuracy'])

    beam_search_initial_input = Input(get_shape=(1,))
    rnn_decoder_with_beam_search = RNNDecoderLayerWithBeamSearch(beam_search_max_output_length, beam_size, decoder_rnn_cell, attention, target_embedding, mlp_classifier)

    beam_search_output_lattice = rnn_decoder_with_beam_search([beam_search_initial_input, source_context])
    rnn_search_runtime_model = Model(input=[source_word, beam_search_initial_input], output=beam_search_output_lattice)

    return (rnn_search_model, rnn_search_runtime_model)

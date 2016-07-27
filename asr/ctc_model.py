'''
Created on Jul 15, 2016
refer to the paper: Deep Speech 2: End-to-End Speech Recognition in English and Mandarin

@author: lxh5147
'''

from keras import backend as K
from keras.layers import Dense, Activation
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Convolution1D
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input
from keras.callbacks import EarlyStopping
from ..common import BiDirectionalLayer

import logging

# TODO; import the following two functions from Tensorflow master branch
def ctc_beam_search_decoder(inputs, sequence_length, beam_width=100, top_paths=1, merge_repeated=True):
    pass

def ctc_loss(inputs, labels, sequence_length, preprocess_collapse_repeated=False, ctc_merge_repeated=True):
    pass

def label_error_rate(labels_true, labels_pred):
    pass

logger = logging.getLogger(__name__)

def length(x):
    '''
    x: 3D tensor of shape batch_size, timestpes, input_dim
    returns 1D integer tensor of (batch_size,) that stores the real timesteps for each sample in the batch
    '''
    # Reference: https://danijar.com/variable-sequence-lengths-in-tensorflow/
    used = K.sign(K.max(K.abs(x), axis=-1))
    length = K.sum(used, axis=-1)
    length = K.cast(length, 'int32')
    return length

# lose function
def ctc(y_true, y_pred):
    '''
    y_true: sparse tensor of batch_size, time_steps,label_id
    y_pred: tensor of batch_size, time_steps, output_dim
    
    delegate to: https://www.tensorflow.org/versions/master/api_docs/python/nn.html#ctc_loss
    tf.nn.ctc_loss(inputs, labels, sequence_length, preprocess_collapse_repeated=False, ctc_merge_repeated=True)
    '''
    # swap time_step and batch
    inputs = K.permute_dimensions(y_pred, (1, 0, 2))
    labels = y_true
    sequence_length = length(y_pred)
    return ctc_loss(inputs, labels, sequence_length, preprocess_collapse_repeated=False, ctc_merge_repeated=True)

def error_rate(y_true, y_pred):
    '''
    label level error rate
    '''
    inputs = K.permute_dimensions(y_pred, (1, 0, 2))
    labels_true = y_true
    sequence_length = length(y_pred)
    labels_pred = ctc_beam_search_decoder(inputs, sequence_length, beam_width=100, top_paths=1, merge_repeated=True)
    return label_error_rate(labels_true, labels_pred)


def attach_ctc_beam_decodeder_to_model(model):
    output = K.permute_dimensions(model.output, (1, 0, 2))
    sequence_length = length(output)
    output = ctc_beam_search_decoder(output, sequence_length, beam_width=100, top_paths=1, merge_repeated=True)
    return Model(input=model.input, output=output)

def build_model(max_timesteps, input_dim, conv_output_dims, conv_filter_lengths, recurrent_output_dims, lookahead_conv_output_dims, lookahead_conv_filter_lengths, dense_output_dims, softmax_output_dim):
    # input shape: batch_size, timesteps, input_dim
    spectrogram = Input(shape=(max_timesteps, input_dim))
    output = spectrogram
    # id conv layers
    for conv_output_dim, conv_filter_length  in zip(conv_output_dims, conv_filter_lengths):
        conv = Convolution1D(conv_output_dim, conv_filter_length, border_mode='same')
        output = conv (output)
        batch_norm = BatchNormalization()
        output = batch_norm(output)
    # bi-directional recurrent layers
    for recurrent_output_dim in recurrent_output_dims:
        recurrent_left_to_right = GRU(recurrent_output_dim, return_sequences=True)
        recurrent_right_to_left = GRU(recurrent_output_dim, return_sequences=True, go_backwards=True)
        h1 = recurrent_left_to_right(output)
        h2 = recurrent_right_to_left(output)
        output = BiDirectionalLayer()(h1, h2)
        batch_norm = BatchNormalization()
        output = batch_norm(output)

    # lookahead conv layers
    for conv_output_dim, conv_filter_length  in zip(lookahead_conv_output_dims, lookahead_conv_filter_lengths):
        conv = Convolution1D(conv_output_dim, conv_filter_length, border_mode='valid')
        output = conv (output)
        batch_norm = BatchNormalization()
        output = batch_norm(output)

    # full connection layers
    for dense_output_dim in dense_output_dims:
        dense = Dense(dense_output_dim)
        output = TimeDistributed(dense)(output)
        output = TimeDistributed(Activation('relu'))(output)
        batch_norm = BatchNormalization()
        output = batch_norm(output)
    # softmax output layer
    output = TimeDistributed(Dense(softmax_output_dim, init='uniform'))(output)
    output = TimeDistributed(Activation('softmax'))(output)
    # connect output with CTC beam search
    model = Model (input=spectrogram, output=output)
    # compile the model, so that we can train
    model.compile(optimizer='rmsprop', loss=ctc, metrics=[error_rate])

    model_with_ctc_beam_decoder = attach_ctc_beam_decodeder_to_model(model)
    return model, model_with_ctc_beam_decoder

def train_model(model, x, y, batch_size=32, nb_epoch=10, validation_split=.1):
    return model.fit(x, y, batch_size, nb_epoch, verbose=1, callbacks=[EarlyStopping(mode='min', patience=5)], validation_split, validation_data=None, shuffle=True, class_weight=None, sample_weight=None)

def evaluate_model(model, x, y, batch_size=32):
    return model.evaluate(x, y, batch_size, verbose=1, sample_weight=None)

def predict_with_model_and_ctc_decoder(model_with_ctc_decoder, x, batch_size=32):
    '''
    model_with_ctc_decoder: model with ctc decoder
    '''
    return model_with_ctc_decoder.predict(x, batch_size=32, verbose=0)

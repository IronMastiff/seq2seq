import tensorflow as tf
import tensorflow.contrib as tf_contrib

import time
import preprocessor
import utils
import test

'''--------Mode--------'''
# Hyperparameters
# Number of Epochs
epochs = 60
# Batch_Size
Batch_size = 128
# RNN Size
rnn_size = 50
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 15
decoding_embedding_size = 15
# Learning Rate
learning_rate = 0.001


# Input
def get_model_inputs():
    input_data = tf.placeholder( tf.int32, [None, None], name = 'input' )
    targets = tf.placeholder( tf.int32, [None, None], name = 'targets' )
    Lr = tf.placeholder( tf.float32, name = 'learning_rate' )

    target_sequence_length = tf.placeholder( tf.int32, ( None, ), name = 'target_sequence_length' )
    max_target_sequence_lenght = tf.reduce_max( target_sequence_length, name = "max_target_len" )
    source_sequence_length = tf.placeholder( tf.int32, ( None, ), name = 'source_sequence_length' )

    return input_data, targets, Lr, target_sequence_length, max_target_sequence_length, source_sequence_length


# Encoder
def encoding_layer( input_data, rnn_size, num_layers, source_sequence_length, source_vocab_size,
                    encoding_embedding_size ):

    # Encoding embedding
    enc_embed_input = tf_contrib.layers.embed_sequence( input_data, source_vocab_size, encoding_embedding_size )

    # RNN cell
    def make_cell( rnn_cell ):
        enc_cell = tf_contrib.rnn.LSTMCell( rnn.size, initializer = tf.random_uniform_initializer( -0.1, 0.1, seed = 2 ) )

        return enc_cell

    enc_output, enc_state = tf.nn.dynamic_rnn( enc_cell, enc_embed_input, sequence_length = source_sequence_length, dtype = tf.float32 )

    return enc_output, enc_state



import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np
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
    def make_cell( rnn_size ):
        enc_cell = tf_contrib.rnn.LSTMCell( rnn.size, initializer = tf.random_uniform_initializer( -0.1, 0.1, seed = 2 ) )

        return enc_cell

    enc_cell = tf_contrib.rnn.MultiRNNCell( [make_cell( rnn_size ) for _ in range( num_layers )] )

    enc_output, enc_state = tf.nn.dynamic_rnn( enc_cell, enc_embed_input, sequence_length = source_sequence_length, dtype = tf.float32 )

    return enc_output, enc_state


# Process the input we'll feed to the decoder
def process_decoder_input( target_data, vocab_to_int, batch_size ):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
    ending = tf.strided_slice( target_data, [0, 0], [batch_size, -1], [1, 1] )
    dec_input = tf.concat( [tf.fill( [batch_size, 1], vocab_to_int['<GO>'] ), ending], 1 )

    return dec_input


def decoding_layer( target_latter_to_int, decoding_embedding_size, num_lyaers, rnn_size, target_seqence_length,
                    max_target_sequence_length, enc_state, dec_input ):
    # 1.Decoder Embedding
    target_vocab_size = len( target_letter_to_int )
    dec_embeddings = tf.Variable( tf.ramdom_uniform( [target_vocab_size, decoding_embedding_size] ) )
    dec_embed_input = tf.nn.embedding_lookup( dec_embeddings, dec_input )

    # 2. Construct the decoder cell
    def make_cell( rnn_size ):
        dec_cell = tf_contrib.rnn.LSTMCell( rnn_size, initializer = tf.random_uniform_initializer( -0.1, 0.1, seed - 2 ) )

        return dec_cell

    dec_cell = tf_contrib.rnn.MultiRNNCell( [make_cell( rnn_size ) for _ in range( num_layers )] )

    # 3. Dense layer to translate the docoder's output at each time
    # Step into a choise from the target vocabulary
    output_layer = tf.layers.Dense( target_vocab_size, kernel_initializer = tf.truncated_normal_initializer( mean = 0.0, stddev = 0.1 ) )

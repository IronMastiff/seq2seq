import tensorflow as tf
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
    Ir = tf.placeholder( tf.float32, name = 'learning_rate' )

    target_seqence_length = tf.placeholder( tf.int32, ( None ), name = 'target_sequence_length' )
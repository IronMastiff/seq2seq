import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np
import time

import preprocessor
import utils
import test

# Hyperparameters
# Number of Epochs
epochs = 60
# Batch_Size
batch_size = 128
# RNN Size
rnn_size = 50
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 15
decoding_embedding_size = 15
# Learning Rate
learning_rate = 0.001

'''--------Prediction--------'''
def source_to_seq( text ):
    '''Prepare the text for the model'''
    sequence_length = 7
    return [preprocessor.source_letter_to_int.get( word, preprocessor.source_letter_to_int['<UNK>']) for word in text] + \
            [preprocessor.source_letter_to_int['<PAD>']] * ( sequence_length - len( text ) )


input_sentence = 'zab'
text = source_to_seq( input_sentence )

checkpoint = './best_model.ckpt'

loaded_graph = tf.Graph()
with tf.Session( graph = loaded_graph ) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph( checkpoint + '.meta' )
    loader.restore( sess, checkpoint )

    input_data = loaded_graph.get_tensor_by_name( 'input:0' )
    logits = loaded_graph.get_tensor_by_name( 'predictions:0' )
    source_sequence_length = loaded_graph.get_tensor_by_name( 'source_sequence_length:0' )
    target_sequence_length = loaded_graph.get_tensor_by_name( 'target_sequence_length:0' )

    # Multiply by batch_size to match the model's input parameters
    answer_logits = sess.run( logits, {
        input_data : [text] * batch_size,
        target_sequence_length : [len( text )] * batch_size,
        source_sequence_length : [len( text )] * batch_size
    })[0]

pad = preprocessor.source_letter_to_int['<PAD>']

print( 'Original Text:', input_sentence )

print( '\nSource' )
print( '  Word Ids:     {}'.format( [i for i in text] ) )
print( '  Input Words:  {}'.format( " ".join( [preprocessor.source_int_to_letter[i] for i in text] ) ) )

print( '\nTarghet' )
print( '  Word Ids:       {}'.format( [i for i in answer_logits if i != pad] ) )
print( '  Response Words: {}'.format( " ".join( [preprocessor.target_int_to_letter[i] for i in answer_logits if i != pad] ) ) )
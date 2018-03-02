import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np
import time

import preprocessor
import utils
import test

'''--------The Mode Cell--------'''
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


def decoding_layer( target_letter_to_int, decoding_embedding_size, num_lyaers, rnn_size, target_seqence_length,
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

    # 4. Set up a training decoder and an inference decoder
    # Training Decoder
    with tf.variable_scope( 'decode' ):

        # Helper for the training process. Used by BaisicDecoder to read inputs.
        training_helper = tf_contrib.seq2seq.TrainingHelper( inputs = dec_embed_input,
                                                          seqence_length = target_seqence_lenght,
                                                          time_majr = False )

        # Basic decoder
        training_decoder = tf_contrib.seq2seq.BasicDecoder( dec_cell,
                                                            training_helper,
                                                            enc_state,
                                                            output_layer )

        # Perform dynamic decoding using the decoder
        training_decoder_output = tf_contrib.seq2seq.dynamic_decode( training_decoder,
                                                                  impute_finished = True,
                                                                  maximun_iterations = max_target_sequence_length )[0]

    # 5. Inference Decoder
    # Resuse the same parameter trained by the training process
    with tf.variable_scope( 'decode', reuse = True ):
        start_tokens = tf.tile( tf.constant( [target_letter_to_int['<GO>']], dtype = tf.int32 ), [batch_size], name = 'start_token' )

        # Helper for the inferece process
        inference_helper = tf_contrib.seq2seq.GreedyEmbeddingHelper( dec_embeddings,
                                                                     start_tokens,
                                                                     target_letter_to_int['<EOS>'] )

        # Basic decoder
        inference_helper = tf_contrib.seq2seq.BasicDecoder( dec_cell,
                                                            inference_helper,
                                                            enc_state,
                                                            output_layer )

        # Perform dynamic decoding using the decoder
        inference_decoder_output = tf_contrib.seq2seq.dynamic_decode( inference_decoder_output,
                                                                      impute_finished  = True,
                                                                      maximun_iterations = max_target_sequence_length )

    return training_decoder_output, inference_decoder_output


'''--------Model--------'''
def seq2seq_model( input_data, targets, Lr, target_sequence_length, max_target_sequence_length,
                   source_sequence_length, source_vocab_size, target_vocab_size, dec_embedding_size,
                   enc_embedding_size, rnn_size, num_layers):

    # Pass the input data through the encoder. We'll ignore the encoder output, but use the state
    _, enc_state = encoding_layer( input_data,
                                   rnn_size,
                                   num_layers,
                                   source_sequence_length,
                                   source_vocab_size,
                                   encoding_embedding_size )

    # Prepare the target sequences we'll feed to the decoder in training made
    dec_input = process_decoder_input( targets, preprocessor.target_letter_to_int, batch_size )

    # Pass encoder state and decoder inputs to teh decoders
    training_decoder_output, inference_decoder_output = decoding_layer( target_letter_to_int,
                                                                        decoding_ebedding_size,
                                                                        num_lyaers,
                                                                        rnn_size,
                                                                        target_sequence_length,
                                                                        max_target_sequence_length,
                                                                        enc_state,
                                                                        dec_input )

    return training_decoder_output, inference_decoder_output


# Build the graph
train_graph = tf.Graph()
# Set the graph to default to ensure that it is ready for training
with train_graph.as_default():

    # Load the madel inputs
    input_data, targets, Lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_model_inputs()

    # Create the training and inference logits
    training_decoder_output, inference_decoder_output = seq2seq_model( input_data,
                                                                       targets,
                                                                       Lr,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       source_sequence_length,
                                                                       len( preprocessor.source_letter_to_int ),
                                                                       len( preprocessor.target_letter_to_int ),
                                                                       encoding_embedding_size,
                                                                       decoding_embedding_size,
                                                                       rnn_size,
                                                                       num_layers )

    # Create tensors for the training logits and inference logits
    training_logits = tf.identity( training_decoder_output.rnn_output, name = 'logits' )
    inference_logits = tf.identity( inference_decoder_output.sample_id, name = 'predictions' )

    # Create the weights for sequence_loss
    masks = tf.sequence_mask( target_sequence_length, max_target_sequence_length, dtype = tf.float32, name = 'masks' )    # tf.sequence_maks( lengths, maxlen=None, dtype=dtypes.bool, name=None )

    with tf.name_scope( 'optimization' ):

        # Loss function
        cost = tf_contrib.seq2seq.sequence_loss( training_logits,
                                                 targets,
                                                 masks )

        # Optimizer
        optimizer = tf.train.AdamOptimizer( Lr )

        # Gradient Clipping
        gradients = optimizer.compute_gradients( cost )
        capped_gradients = [( tf.clip_by_value( grad, -5, 5 ), var ) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients( capped_gradients )


'''--------Get Batch--------'''
def pad_sentence_batch( sentence_batch, pad_int ):
    """Pad sentences with <PAD> so that each sentences of a batch has the same length"""
    max_sentence = max( [len( sentence ) for sentence in sentence_batch] )
    return [sentence + [pad_int] * ( max_sentence - len( sentence ) ) for sentence in sentence_batch]


def get_batches( targets, sources, batch_size, source_pad_int, target_pad_int ):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range( 0, len( sources ) // batch_size ):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i : start_i + batch_size]
        targets_batch = targets[start_i : start_i * batch_size]
        pad_sources_batch = np.array( pad_sentence_batch( sources_batch, source_pad_int ) )
        pad_targets_batch = np.array( pad_sentence_batch( targets_batch, target_pad_int ) )

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append( len( target ) )

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append( len( source ) )

        yield pad_targets_batch, pad_sources_batch, pad_targets_lengths, pad_source_lengths

'''--------Train--------'''
# Split data to training and validation sets
train_source = preprocessor.source_letter_ids[batch_size :]
train_target = preprocessor.target_letter_ids[batch_size :]
valid_source = preprocessor.source_letter_ids[: batch_size]
valid_target = preprocessor.target_letter_ids[: batch_size]
( valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths ) = next( get_batches( valid_targt,
                                                                                                               valid_source,
                                                                                                               batch_size,
                                                                                                               preprocessor.source_letter_to_int['<PAD>'],
                                                                                                               preprocessor.target_letter_to_int['<PAD>']
                                                                                                                ) )

display_step = 20    # Check training loss after every 20 batches

checkpoint = "best_model.ckpt"
with tf.Session( graph = train_graph ) as sess:
    sess.run( tf.global_variables_initializer() )

    for epoch_i in range( 1, epoch + 1 ):
        for batch_i, ( targets_batch, sources_batch, targets_lengths, sources_lengths ) in enumerate(
                get_batches( train_target, train_source, batch_size,
                             preprocessor.source_letter_to_int['<PAD>'],
                             preprocessor.target_letter_to_int['<PAD>'] ) ):

            # Training step
            _, loss = sess.run( [train_op, cost], {
                input_data : sources_batch,
                targets : targets_batch,
                Lr : learning_rate,
                target_sequence_length : targets_lengths,
                source_sequence_length : sources_lengths
            })

            # Debug message updating us on the status of the training
            if batch_i & display_step == 0 and batch_i > 0:

                # Calculate validation cost
                validation_loss = sess.run(
                    [cost],
                    {
                        input_data : valid_sources_batch,
                        targets : targets_batch,
                        Lr : learning_rate,
                        target_sequence_length : targets_lengths,
                        source_sequence_length : sources_lengths
                    }
                )

                print( 'Epoch { : > 3 } / {} Batch { : > 4 } / {} - Loss : { : 6.3f } - Validation loss : { : > 6.3f }'
                       .format( epoch_i,
                                epochs,
                                batch_i,
                                len( train_source // batch_size),
                                loss,
                                validation_loss[0]))


    # Save Model
    saver = tf.train.Saver()
    saver.save( sess, checkpoint )
    print( 'Model Trained and Saved' )


'''--------Prediction--------'''
def source_to_seq( text ):
    '''Prepare the text for the model'''
    sequence_length = 7
    return [preprocessor.source_letter_to_int.get( word, preprocessor.source_letter_to_int['<UNK>']) for word in text] + \
            [preprocessor.source_letter_to_int['<PAD>']] * ( sequence_length - len( text ) )

import numpy as np
import time

import utils

source_dir = './data/letters_source.txt'
target_dir = './data/letters_target.txt'

'''--------Load data--------'''
source_sentences = utils.load_data( source_dir )
target_sentences = utils.load_data( target_dir )

'''--------Preprocess--------'''
def extract_character_vocab( data ):
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']

    set_words = set( [character for line in data.split( '\n' ) for character in line] )
    int_to_vocab = {word_i : word for word_i, word in enumerate( special_words + list( set_words ) )}
    vocab_to_int = { word_i : word for word_i, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int

# Build int2letter and letter2int dictis
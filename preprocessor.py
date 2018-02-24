import numpy as np
import time
import test
import tensorflow as tf


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
    vocab_to_int = {word : word_i for word_i, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int


'''--------Build int2letter and letter2int dictis--------'''
source_int_to_letter, source_letter_to_int = extract_character_vocab( source_sentences )
target_int_to_letter, target_letter_to_int= extract_character_vocab( target_sentences )


'''--------Convert character to ids--------'''
source_letter_ids = [[source_letter_to_int.get( letter, source_letter_to_int['<UNK>'] ) for letter in line]
                     for line in source_sentences.split( '\n' )]
target_letter_ids = [[target_letter_to_int.get( letter, target_letter_to_int['<UNK>'] ) for letter in line] +
                     [target_letter_to_int['<EOS>']] for line in target_sentences.split( '\n' )]


if __name__ == "__main__":
    '''-------Test code--------'''
    TEST = 1  # The switch of test
    test.test_print( source_sentences[: 100], TEST )
    test.test_print( [character for line in source_sentences.split( '\n' ) for character in line][:100], TEST )
    test.test_print( source_letter_ids[: 100], TEST )
    test.test_print( source_letter_to_int.get( 'b', source_letter_to_int['<UNK>'] ), TEST )    # dict.get( key, default = None ) 若key不存在时则返回default
    test.test_print(target_sentences[: 100], TEST)
    test.test_print([character for line in target_sentences.split('\n') for character in line][:100], TEST)
    test.test_print(target_letter_ids[: 100], TEST)
    test.test_print(target_letter_to_int.get('b', target_letter_to_int['<UNK>']), TEST)
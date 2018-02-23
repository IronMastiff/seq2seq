import os

def load_data( path ):
    input_file = os.path.join( path )
    with open( input_file, 'r', encoding = 'utf-8', errors = 'ignore' ):
        data = f.read()

    return data

def extract_vocab( data ):
    special_words = ['<pad>', '<unk>', '<s>', '<\s>']

    set_words = set( [words for line in data.split( '\n' ) for word in line.split()] )
    int_to_vocab = { word_i : word for word_i, word in enumerate( special_words + list( set_words ) ) }
    vocab_to_int = { word : word_i for word_i, word in int_to_vocab.items }

    return int_to_vocab, vocab_to_int

def pad_id_sequences( source_ids, source_vocab_to_int, target_ids, target_vocab_to_int, sequence_lengh ):
    new_source_ids = [list( reversed( sentence + [source_vocab_to_int['<pad>']] * ( sequence_lengh - len( sentence ) ) ))]
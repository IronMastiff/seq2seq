import numpy as np
import time

import utils

source_dir = './data/letters_source.txt'
target_dir = './data/letters_target.txt'

source_sentences = utils.load_data( source_dir )
target_sentences = utils.load_data( target_dir )

print( source_sentences[: 50].split( '\n' ) )
import numpy as np

from rulm.vocabulary import Vocabulary

def process_line(line: str, vocabulary: Vocabulary, max_length: int, reverse: bool):
    words = line.strip().split()
    indices = vocabulary.numericalize_inputs(words, reverse=reverse)
    indices += [vocabulary.get_eos()]
    indices = vocabulary.pad_indices(indices, max_length)
    return np.array(indices, dtype="int32")


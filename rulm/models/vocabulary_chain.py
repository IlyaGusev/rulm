from typing import Tuple, List

import numpy as np
from allennlp.common import Params
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Vocabulary
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN

from rulm.language_model import LanguageModel
from rulm.transform import Transform


@LanguageModel.register("vocabulary_chain")
class VocabularyChainLanguageModel(LanguageModel):
    def __init__(self, vocab: Vocabulary, transforms: Tuple[Transform]=tuple()):
        LanguageModel.__init__(self, vocab, transforms)

    def train(self,
              inputs: List[List[str]],
              train_params: Params,
              serialization_dir: str=None,
              **kwargs):
        pass

    def train_file(self,
                   file_name: str,
                   train_params: Params,
                   serialization_dir: str=None,
                   **kwargs):
        pass

    def normalize(self):
        pass

    def _load(self,
              params: Params,
              vocab: Vocabulary,
              serialization_dir: str,
              weights_file: str,
              cuda_device: int=-1):
        pass

    def predict(self, inputs: List[int]):
        probabilities = np.zeros(self.vocab.get_vocab_size())
        last_index = inputs[-1]
        aux = (START_SYMBOL, END_SYMBOL, DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN)
        aux_indices = [self.vocab.get_token_index(s) for s in aux]
        first_not_aux_index = 0
        for i in range(self.vocab.get_vocab_size()):
            if i in aux_indices:
                continue
            first_not_aux_index = i
            break
        bos_index = aux_indices[0]
        if last_index == bos_index:
            probabilities[first_not_aux_index] = 1.
        elif last_index != self.vocab.get_vocab_size() - 1:
            probabilities[last_index + 1] = 1.
        return probabilities

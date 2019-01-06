from typing import Tuple, List

import numpy as np
from allennlp.common import Params
from allennlp.common.util import START_SYMBOL
from allennlp.data import Vocabulary
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN

from rulm.language_model import LanguageModel
from rulm.transform import Transform


@LanguageModel.register("equiprobable")
class EquiprobableLanguageModel(LanguageModel):
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
        vocab_size = self.vocab.get_vocab_size()
        probabilities = np.full((vocab_size,), 1./(vocab_size-2))
        probabilities[self.vocab.get_token_index(START_SYMBOL)] = 0.
        probabilities[self.vocab.get_token_index(DEFAULT_PADDING_TOKEN)] = 0.
        return probabilities

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
    def __init__(self, vocabulary: Vocabulary, transforms: Tuple[Transform]=tuple()):
        LanguageModel.__init__(self, vocabulary, transforms)

    def train(self, inputs: List[List[str]], train_params: Params):
        pass

    def train_file(self, file_name: str, train_params: Params):
        pass

    def normalize(self):
        pass

    def predict(self, inputs: List[int]):
        vocab_size = self.vocabulary.get_vocab_size()
        probabilities = np.full((vocab_size,), 1./(vocab_size-2))
        probabilities[self.vocabulary.get_token_index(START_SYMBOL)] = 0.
        probabilities[self.vocabulary.get_token_index(DEFAULT_PADDING_TOKEN)] = 0.
        return probabilities

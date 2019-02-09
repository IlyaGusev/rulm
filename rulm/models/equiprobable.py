from typing import List, Dict

import numpy as np
from torch import Tensor
from allennlp.common import Params
from allennlp.common.util import START_SYMBOL
from allennlp.data import Vocabulary
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN

from rulm.language_model import LanguageModel
from rulm.transform import Transform


@LanguageModel.register("equiprobable")
class EquiprobableLanguageModel(LanguageModel):
    def __init__(self, vocab: Vocabulary, transforms: List[Transform]=None):
        LanguageModel.__init__(self, vocab, transforms)

    def train(self,
              file_name: str,
              train_params: Params,
              serialization_dir: str=None,
              **kwargs):
        pass

    def _load(self,
              params: Params,
              vocab: Vocabulary,
              serialization_dir: str,
              weights_file: str,
              cuda_device: int=-1,
              **kwargs):
        pass

    def predict(self, batch: Dict[str, Dict[str, Tensor]], **kwargs) -> np.ndarray:
        inputs = batch["source_tokens"]["tokens"]
        batch_size = inputs.size(0)
        vocab_size = self.vocab.get_vocab_size()
        probabilities = np.full((vocab_size,), 1./(vocab_size-2))
        probabilities[self.vocab.get_token_index(START_SYMBOL)] = 0.
        probabilities[self.vocab.get_token_index(DEFAULT_PADDING_TOKEN)] = 0.

        result = np.zeros((batch_size, vocab_size), dtype="float")
        for i in range(batch_size):
            result[i] = probabilities
        return result

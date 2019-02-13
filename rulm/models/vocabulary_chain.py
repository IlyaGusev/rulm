from typing import List, Dict

import numpy as np
from torch import Tensor
from allennlp.common import Params
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Vocabulary
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN

from rulm.language_model import LanguageModel
from rulm.transform import Transform


@LanguageModel.register("vocabulary_chain")
class VocabularyChainLanguageModel(LanguageModel):
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
              cuda_device: int=-1.,
              **kwargs):
        pass

    def predict(self, batch: Dict[str, Dict[str, Tensor]], **kwargs) -> np.ndarray:
        inputs = batch["source_tokens"]["tokens"]
        batch_size = inputs.size(0)
        vocab_size = self.vocab.get_vocab_size("tokens")
        result = np.zeros((batch_size, vocab_size), dtype="float")
        for sample_number in range(batch_size):
            probabilities = np.zeros(vocab_size)
            aux = (START_SYMBOL, END_SYMBOL, DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN)
            aux_indices = [self.vocab.get_token_index(s) for s in aux]
            bos_index = aux_indices[0]
            eos_index = aux_indices[1]
            last_index = inputs[sample_number][-1]
            first_not_aux_index = 0
            for i in range(self.vocab.get_vocab_size()):
                if i in aux_indices:
                    continue
                first_not_aux_index = i
                break
            if last_index == bos_index:
                probabilities[first_not_aux_index] = 1.
            elif last_index != self.vocab.get_vocab_size() - 1:
                probabilities[last_index + 1] = 1.
            elif last_index == self.vocab.get_vocab_size() - 1:
                probabilities[eos_index] = 1.
            result[sample_number] = probabilities
        return result

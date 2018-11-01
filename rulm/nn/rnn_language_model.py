from typing import Tuple

import torch

from rulm.transform import Transform
from rulm.vocabulary import Vocabulary
from rulm.nn.language_model import NNLanguageModel
from rulm.nn.models.rnn import RNNModule, RNNModuleConfig


class RNNLanguageModel(NNLanguageModel):
    def __init__(self,
                 vocabulary: Vocabulary,
                 transforms: Tuple[Transform]=tuple(),
                 reverse: bool=False,
                 config: RNNModuleConfig=RNNModuleConfig()):
        NNLanguageModel.__init__(self, vocabulary, transforms, reverse)
        self.config = config
        self.config.vocabulary_size = min(self.config.vocabulary_size, len(vocabulary))
        self.model = RNNModule(self.config)


from typing import Tuple

import torch
from rulm.transform import Transform
from rulm.vocabulary import Vocabulary
from rulm.nnlm import NNLanguageModel
from rulm.models.rnn import RNNModule, RNNModuleConfig

class RNNLanguageModel(NNLanguageModel):
    def __init__(self, vocabulary: Vocabulary,
                 transforms: Tuple[Transform]=tuple(),
                 reverse: bool=False, config: RNNModuleConfig=RNNModuleConfig()):
        NNLanguageModel.__init__(self, vocabulary, transforms, reverse)
        self.config = config
        self.config.vocabulary_size = min(self.config.vocabulary_size, len(vocabulary))
        self.model = RNNModule(self.config)
        use_cuda = torch.cuda.is_available()
        self.model.cuda() if use_cuda else self.model
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)


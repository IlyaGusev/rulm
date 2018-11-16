import os
from typing import List, Tuple, Iterable
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from allennlp.common.params import Params
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer import Trainer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from rulm.transform import Transform
from rulm.language_model import LanguageModel
from rulm.stream_reader import LanguageModelingStreamReader
from rulm.nn.models.lm import LMModule

use_cuda = torch.cuda.is_available()
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor


class NNLanguageModel(LanguageModel):
    def __init__(self,
                 vocabulary: Vocabulary,
                 params: Params,
                 transforms: Tuple[Transform]=tuple(),
                 reverse: bool=False,
                 seed: int=42):
        LanguageModel.__init__(self, vocabulary, transforms, reverse)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.set_flags(True, False, True, True)

        self.params = params

        self.model = LMModule.from_params(self.params.pop('model'), vocab=vocabulary)
        print(self.model)
        print("Trainable params count: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.reader = DatasetReader.from_params(self.params.pop("dataset_reader"), reverse=self.reverse)
        self.iterator = DataIterator.from_params(self.params.pop("data_iterator"),
                                                 sorting_keys=[("input_tokens", "num_tokens")])
        self.iterator.index_with(self.vocabulary)

    def train_file(self, file_name: str, serialization_dir: str=None):
        assert os.path.exists(file_name)
        dataset = self.reader.read(file_name)

        trainer_params = self.params.pop('trainer')

        self.trainer = Trainer.from_params(self.model, serialization_dir, self.iterator,
                                           dataset, None, trainer_params)
        self.trainer.train()
        trainer_params.assert_empty("Bad trainer params")


    def predict(self, indices: List[int]) -> List[float]:
        self.model.eval()

        indices = LongTensor(indices)
        indices = torch.unsqueeze(indices, 0)
        input_tokens = {"tokens": indices}
        logits = self.model.forward(input_tokens=input_tokens)["logits"]
        result = logits.transpose(1, 2).transpose(0, 1)
        result = torch.exp(torch.squeeze(result, 1)[-1]).cpu().detach().numpy()
        return result


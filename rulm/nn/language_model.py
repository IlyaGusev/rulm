import os
from typing import List, Tuple, Iterable
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from allennlp.nn import util
from allennlp.common.params import Params
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer import Trainer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.models.model import Model
from allennlp.data.tokenizers import Token

from rulm.transform import Transform
from rulm.language_model import LanguageModel
from rulm.stream_reader import LanguageModelingStreamReader
from rulm.nn.models.encoder_only import EncoderOnlyLanguageModel


_DEFAULT_PARAMS = "params.json"
_DEFAULT_VOCAB_DIR = "vocabulary"


@LanguageModel.register("nn_language_model")
class NNLanguageModel(LanguageModel):
    def __init__(self,
                 vocab: Vocabulary,
                 model: Model,
                 transforms: Tuple[Transform]=tuple(),
                 reverse: bool=False,
                 seed: int=42):
        LanguageModel.__init__(self, vocab, transforms, reverse)

        self.set_seed(seed)
        self.model = model

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.set_flags(True, False, True, True)

    def train(self, inputs: Iterable[List[str]], train_params: Params, serialization_dir: str=None):
        raise NotImplementedError()

    def train_file(self, file_name: str, train_params: Params, serialization_dir: str=None):
        assert os.path.exists(file_name)
        reader = DatasetReader.from_params(train_params.pop('reader'), reverse=self.reverse)
        dataset = reader.read(file_name)
        self._train_dataset(dataset, train_params, serialization_dir)

    def _train_dataset(self,
                       dataset: Iterable[Instance],
                       train_params: Params,
                       serialization_dir: str=None):
        if serialization_dir:
            vocab_dir = os.path.join(serialization_dir, _DEFAULT_VOCAB_DIR)
            self.vocabulary.save_to_files(vocab_dir)

        iterator = DataIterator.from_params(train_params.pop('iterator'))
        iterator.index_with(self.vocabulary)
        trainer = Trainer.from_params(self.model, serialization_dir, iterator,
                                      dataset, None, train_params.pop('trainer'))
        train_params.assert_empty("Trainer")
        trainer.train()

    def predict(self, indices: List[int]) -> List[float]:
        self.model.eval()
        LongTensor = torch.cuda.LongTensor if next(self.model.parameters()).is_cuda else torch.LongTensor
        indices = LongTensor(indices)
        indices = torch.unsqueeze(indices, 0)
        input_tokens = {"tokens": indices}

        logits = self.model.forward(input_tokens=input_tokens)["logits"]
        result = logits.transpose(1, 2).transpose(0, 1)
        result = torch.exp(torch.squeeze(result, 1)[-1]).cpu().detach().numpy()

        return result

    def print(self):
        print(self.model)
        print("Trainable params count: ",
              sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    @classmethod
    def load(self,
             serialization_dir: str,
             params_file: str=None,
             weights_file: str=None,
             cuda_device: int=-1):
        params_file = params_file or os.path.join(serialization_dir, _DEFAULT_PARAMS)
        params = Params.from_file(params_file)
        if params.get('train', None):
            params.pop('train')

        inner_model = Model._load(
            params,
            serialization_dir,
            weights_file=weights_file,
            cuda_device=cuda_device)
        params.pop('model')
        model = NNLanguageModel.from_params(params, model=inner_model, vocab=inner_model.vocab)
        return model


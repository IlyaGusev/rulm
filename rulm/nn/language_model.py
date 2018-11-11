import os
from typing import List, Tuple, Iterable
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from ignite.engine import Events
from ignite.metrics import Loss
from ignite.handlers import ModelCheckpoint
from allennlp.common.params import Params
from allennlp.training.optimizers import Optimizer

from rulm.utils import process_line
from rulm.nn.utils import create_lm_evaluator, create_lm_trainer, MaskedCategoricalAccuracy
from rulm.transform import Transform
from rulm.vocabulary import Vocabulary
from rulm.language_model import LanguageModel
from rulm.datasets.dataset import Dataset
from rulm.datasets.stream_dataset import StreamDataset, StreamFilesDataset
from rulm.datasets.chunk_dataset import ChunkDataset
from rulm.nn.models.lm import LMModule

use_cuda = torch.cuda.is_available()
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor


def preprocess_batch(batch):
    lengths = [np.count_nonzero(sample) for sample in batch]
    batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: x[1], reverse=True))
    batch = np.array(batch)[:, :max(lengths)]
    lengths = list(lengths)

    y = np.zeros((batch.shape[0], batch.shape[1]), dtype=batch.dtype)
    y[:, :-1] = batch[:, 1:]

    batch = torch.transpose(LongTensor(batch), 0, 1)
    y = LongTensor(y)
    return {"x": batch, "y": y, "lengths": lengths}


class NNLanguageModel(LanguageModel):
    def __init__(self,
                 vocabulary: Vocabulary,
                 model_params: Params,
                 transforms: Tuple[Transform]=tuple(),
                 reverse: bool=False,
                 max_length: int=50,
                 seed: int=42):
        LanguageModel.__init__(self, vocabulary, transforms, reverse)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.set_flags(True, False, True, True)

        self.max_length = max_length
        vocabulary_size = len(vocabulary)
        self.model = LMModule.from_params(model_params, vocabulary_size=vocabulary_size)

    def train(self, inputs: Iterable[List[str]], params: Params):
        params.pop("dataset")
        dataset = StreamDataset(self.process_line, inputs)
        self._train_on_dataset(dataset, params)

    def train_file(self, file_name: str, params: Params):
        assert os.path.exists(file_name)
        dataset = Dataset.from_params(params.pop("dataset"), input_files=[file_name],
                                      process_line=self.process_line)
        self._train_on_dataset(dataset, params)

    def process_line(self, line):
        return process_line(line, self.vocabulary, self.max_length, self.reverse)

    def _train_on_dataset(self, dataset: Dataset, params: Params):
        batch_size = params.pop("batch_size")
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=preprocess_batch)

        model_parameters = [[n, p] for n, p in self.model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(model_parameters, params.pop("optimizer"))
        criterion = nn.NLLLoss(ignore_index=self.vocabulary.get_pad())

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = create_lm_trainer(self.model, optimizer, criterion, device=device)
        evaluator = create_lm_evaluator(self.model, metrics={
            'loss': Loss(criterion),
            'accuracy': MaskedCategoricalAccuracy()
        })

        serialization_dir = params.pop("serialization_dir")
        model_name = "model"
        checkpoint_every = params.get("checkpoint_every", None)
        if checkpoint_every:
            checkpoint_every = params.pop("checkpoint_every")
            checkpointer = ModelCheckpoint(serialization_dir, model_name,
                                           save_interval=checkpoint_every, create_dir=True)
            trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {model_name: self.model})
        start_time = datetime.now()

        validate_every = params.pop("validate_every")

        @trainer.on(Events.ITERATION_COMPLETED)
        def validate(trainer):
            if trainer.state.iteration % validate_every == 0:
                evaluator.run(loader)
                metrics = evaluator.state.metrics
                print("Epoch: {}, iteration: {}, time: {}, loss: {}, accuracy: {}".format(
                    trainer.state.epoch,
                    trainer.state.iteration,
                    datetime.now()-start_time,
                    metrics["loss"],
                    metrics['accuracy']))

        print("Model:")
        print(self.model)
        print("Params count: ", sum(p.numel() for p in self.model.parameters()))
        print("Trainable params count: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        epochs = params.pop("epochs")
        params.assert_empty("train")
        trainer.run(loader, max_epochs=epochs)

    def predict(self, indices: List[int]) -> List[float]:
        self.model.eval()

        indices = LongTensor(indices)
        indices = torch.unsqueeze(indices, 1)
        lengths = [len(indices)]
        result = self.model.forward({"x": indices, "lengths": lengths})
        result = result.transpose(1, 2).transpose(0, 1)
        result = torch.exp(torch.squeeze(result, 1)[-1]).cpu().detach().numpy()
        return result

    def save(self, file_name):
        torch.save(self.model, file_name)

    def load(self, file_name):
        self.model = torch.load(file_name)


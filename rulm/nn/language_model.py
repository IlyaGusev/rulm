import os
from typing import List, Tuple, Generator, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader, default_collate
from ignite.engine import Events
from ignite.metrics import Loss
from ignite.handlers import ModelCheckpoint

from rulm.utils import process_line
from rulm.nn.utils import create_lm_evaluator, create_lm_trainer, MaskedCategoricalAccuracy
from rulm.transform import Transform
from rulm.vocabulary import Vocabulary
from rulm.language_model import LanguageModel
from rulm.datasets.chunk_dataset import ChunkDataset
from rulm.datasets.stream_dataset import StreamDataset, StreamFilesDataset

use_cuda = torch.cuda.is_available()
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor


def preprocess_batch(batch):
    lengths = [np.count_nonzero(sample) for sample in batch]
    batch, lengths  = zip(*sorted(zip(batch, lengths), key=lambda x: x[1], reverse=True))
    batch = np.array(batch)[:, :max(lengths)]
    lengths = list(lengths)

    y = np.zeros((batch.shape[0], batch.shape[1]), dtype=batch.dtype)
    y[:, :-1] = batch[:, 1:]

    batch = torch.transpose(LongTensor(batch), 0, 1)
    y = LongTensor(y)
    return {"x": batch, "y": y, "lengths": lengths}


class TrainConfig:
    def __init__(self,
                 intermediate_dir: str="./chunks",
                 epochs: int=20,
                 batch_size: int=64,
                 checkpoint_dir: str=None,
                 checkpoint_every: int=1,
                 report_every: int=50,
                 validate_every: int=1,
                 lr: float=0.001):
        self.intermediate_dir = intermediate_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.report_every = report_every
        self.validate_every = validate_every
        self.lr = lr


class NNLanguageModel(LanguageModel):
    def __init__(self,
                 vocabulary: Vocabulary,
                 transforms: Tuple[Transform]=tuple(),
                 reverse: bool=False,
                 max_length: int=50):
        LanguageModel.__init__(self, vocabulary, transforms, reverse)

        self.max_length = max_length
        self.model = None


    def train(self, inputs: Generator[List[str], Any, None], config: TrainConfig=TrainConfig()):
        dataset = StreamDataset(self._closed_process_line, inputs)
        self._train_on_dataset(dataset, config)

    def train_file(self, file_name: str, config: TrainConfig=TrainConfig()):
        assert os.path.exists(file_name)
        dataset = StreamFilesDataset([file_name], self._closed_process_line)
        self._train_on_dataset(dataset, config)

    def _closed_process_line(self, line):
        return process_line(line, self.vocabulary, self.max_length, self.reverse)

    def _train_on_dataset(self, dataset: Dataset, config: TrainConfig):
        loader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=preprocess_batch)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.lr)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        criterion = nn.NLLLoss(ignore_index=self.vocabulary.get_pad())

        trainer = create_lm_trainer(self.model, optimizer, criterion, device=device)
        evaluator = create_lm_evaluator(self.model, metrics={
            'loss': Loss(criterion),
            'accuracy': MaskedCategoricalAccuracy()
        })

        if config.checkpoint_dir:
            checkpointer = ModelCheckpoint(config.checkpoint_dir, "model",
                                           save_interval=config.checkpoint_every, create_dir=True)
            trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {"model": self.model})
        start_time = datetime.now()

        @trainer.on(Events.ITERATION_COMPLETED)
        def validate(trainer):
            if trainer.state.iteration % config.validate_every == 0:
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
        trainer.run(loader, max_epochs=config.epochs)

    def predict(self, indices: List[int]) -> List[float]:
        self.model.eval()
        use_cuda = torch.cuda.is_available()
        LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

        indices = LongTensor(indices)
        indices = torch.unsqueeze(indices, 1)
        result = self.model.forward(indices)
        result = result.transpose(1, 2).transpose(0, 1)
        result = torch.exp(torch.squeeze(result, 1)[-1]).cpu().detach().numpy()
        return result

    def save(self, file_name):
        torch.save(self.model, file_name)

    def load(self, file_name):
        self.model = torch.load(file_name)


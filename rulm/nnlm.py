import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader, default_collate
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import CategoricalAccuracy, Loss
from ignite.handlers import ModelCheckpoint

from rulm.transform import Transform
from rulm.vocabulary import Vocabulary
from rulm.language_model import LanguageModel
from rulm.datasets.chunk_dataset import ChunkDataset
from rulm.datasets.stream_dataset import StreamDataset

use_cuda = torch.cuda.is_available()
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor


def process_line(line, vocabulary, max_length, reverse):
    words = line.strip().split()
    indices = vocabulary.numericalize_inputs(words, reverse=reverse)
    indices += [vocabulary.get_eos()]
    indices = vocabulary.pad_indices(indices, max_length)
    return np.array(indices, dtype="int32")


def preprocess_batch(batch):
    lengths = [len([elem for elem in sample if elem != 0]) for sample in batch]
    max_length = max(lengths)
    pairs = sorted(zip(batch, lengths), key=lambda x: x[1], reverse=True)
    batch = [sample[:max_length] for sample, _ in pairs]
    lengths.sort(reverse=True)
    batch = default_collate(batch)
    batch = batch.numpy()

    y = np.zeros((batch.shape[0], batch.shape[1]), dtype=batch.dtype)
    y[:, :-1] = batch[:, 1:]

    batch = torch.transpose(LongTensor(batch), 0, 1)
    y = LongTensor(y)
    return batch, y


class NNLanguageModel(LanguageModel):
    def __init__(self, vocabulary: Vocabulary,
                 transforms: Tuple[Transform]=tuple(),
                 reverse: bool=False):
        LanguageModel.__init__(self, vocabulary, transforms, reverse)

        self.model = None

    def train_file(self, file_name: str, intermediate_dir: str="./chunks",
                   epochs: int=20, batch_size: int=64,
                   checkpoint_dir: str=None, checkpoint_every: int=1,
                   max_length: int=50, report_every: int=50, validate_every: int=1):
        assert os.path.exists(file_name)

        def closed_process_line(line):
            return process_line(line, self.vocabulary, max_length, self.reverse)

        dataset = StreamDataset([file_name], closed_process_line)
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=preprocess_batch)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        criterion = nn.NLLLoss()

        trainer = create_supervised_trainer(self.model, optimizer, criterion, device=device)
        evaluator = create_supervised_evaluator(self.model, metrics={
            'loss': Loss(criterion),
            'accuracy': CategoricalAccuracy()
        })

        if checkpoint_dir:
            checkpointer = ModelCheckpoint(checkpoint_dir, "model",
                                           save_interval=checkpoint_every, create_dir=True)
            trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {"model": self.model})

        @trainer.on(Events.ITERATION_COMPLETED)
        def validate(trainer):
            if trainer.state.iteration % validate_every == 0:
                evaluator.run(loader)
                metrics = evaluator.state.metrics
                print("Epoch: {}, iteration: {}, loss: {}, accuracy: {}".format(
                    trainer.state.epoch,
                    trainer.state.iteration,
                    metrics["loss"],
                    metrics['accuracy']))

        trainer.run(loader, max_epochs=epochs)

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


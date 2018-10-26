import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader, default_collate

from rulm.transform import Transform
from rulm.vocabulary import Vocabulary
from rulm.language_model import LanguageModel
from rulm.datasets.chunk_dataset import ChunkDataset
from rulm.datasets.stream_dataset import StreamDataset

use_cuda = torch.cuda.is_available()
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor


def process_line(line, vocabulary, max_length):
    words = line.strip().split()
    indices = vocabulary.numericalize_inputs(words)
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
    y = torch.transpose(LongTensor(y), 0, 1)
    return {
        'x': batch,
        'y': y,
        'lengths': lengths
    }


class NNLanguageModel(LanguageModel):
    def __init__(self, vocabulary: Vocabulary,
                 transforms: Tuple[Transform]=tuple(),
                 reverse: bool=False):
        LanguageModel.__init__(self, vocabulary, transforms, reverse)

        self.model = None
        self.optimizer = None

    def train(self, data: DataLoader, report_every: int=50):
        assert self.model
        for step, batch in enumerate(data):
            loss = self._process_batch(batch, self.optimizer)
            if step % 10 == 0:
                print("Step: {}, loss: {}".format(step, loss))
        self.save("model.pt")

    def train_file(self, file_name: str, intermediate_dir: str="./chunks",
                   epochs: int=20, batch_size: int=64,
                   max_length: int=50, report_every: int=50):
        assert os.path.exists(file_name)
        for epoch in range(epochs):
            print("Big epoch: {}".format(epoch))
            def closed_process_line(line):
                return process_line(line, self.vocabulary, max_length)
            dataset = StreamDataset([file_name], closed_process_line)
            # dataset = ChunkDataset([file_name], closed_process_line,
            #     intermediate_dir, max_sample_length=max_length, chunk_size=1000000)
            loader = DataLoader(dataset, batch_size=batch_size, collate_fn=preprocess_batch)
            self.train(loader, report_every=report_every)

    def predict(self, indices: List[int]) -> List[float]:
        self.model.eval()
        use_cuda = torch.cuda.is_available()
        LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

        indices = LongTensor(indices)
        indices = torch.unsqueeze(indices, 1)
        result = self.model.forward(indices, [len(indices)])
        result = torch.exp(torch.squeeze(result, 1)[-1]).cpu().detach().numpy()
        return result

    def save(self, file_name):
        torch.save(self.model, file_name)

    def load(self, file_name):
        self.model = torch.load(file_name)

    def _process_batch(self, batch, optimizer=None):
        if optimizer is not None:
            optimizer.zero_grad()

        result = self.model.forward(batch['x'], batch['lengths'])
        result = torch.transpose(result, 0, 1)
        result = torch.transpose(result, 1, 2)
        result = torch.unsqueeze(result, 2)

        target = batch['y']
        target = torch.t(target)
        target = torch.unsqueeze(target, 1)

        criterion = nn.NLLLoss()
        loss = criterion(result, target)

        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            optimizer.step()

        return loss.data.item()

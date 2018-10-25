import os
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from rulm.transform import Transform
from rulm.vocabulary import Vocabulary
from rulm.language_model import LanguageModel
from rulm.chunk_dataset import ChunkDataset, ChunkDataLoader


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
            dataset = ChunkDataset(self.vocabulary, [file_name], intermediate_dir,
                max_sentence_length=max_length, chunk_size=1000000)
            loader = ChunkDataLoader(dataset, batch_size=batch_size)
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

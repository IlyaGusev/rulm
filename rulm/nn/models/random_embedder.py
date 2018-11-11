from typing import Dict, Any

import torch
from torch.nn import Embedding, Dropout

from rulm.nn.models.embedder import Embedder


@Embedder.register("random")
class RandomEmbedder(Embedder):
    def __init__(self,
                 input_dim: int,
                 embedding_dim: int,
                 dropout: float=None):
        super().__init__(input_dim, embedding_dim)

        self._embedding = Embedding(self._input_dim, self._embedding_dim)
        self.dropout = dropout
        if dropout:
            self._dropout = Dropout(dropout)

    def forward(self, source: Dict[str, Any]):
        inputs = source["x"]
        inputs = self._embedding(inputs)
        if self.dropout:
            inputs = self._dropout(inputs)
        return inputs

    def get_weight(self) -> torch.nn.Parameter:
        return self._embedding.weight


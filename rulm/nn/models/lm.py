from typing import Dict, Any

import torch
from torch.nn import Dropout, Linear, LogSoftmax
from allennlp.common.registrable import Registrable

from rulm.nn.models.seq2seq_encoder import Seq2SeqEncoder
from rulm.nn.models.lstm_encoder import LstmEncoder
from rulm.nn.models.embedder import Embedder
from rulm.nn.models.random_embedder import RandomEmbedder


class LMModule(torch.nn.Module, Registrable):
    def __init__(self,
                 vocabulary_size: int,
                 embedder: Embedder,
                 contextualizer: Seq2SeqEncoder,
                 dropout: float = None,
                 tie_embeddings=True):
        super().__init__()

        self._embedder = embedder
        self._contextualizer = contextualizer
        self._context_dim = contextualizer.get_output_dim()

        if dropout:
            self._dropout = Dropout(dropout)
        else:
            self._dropout = lambda x: x

        self._softmax_linear = Linear(self._context_dim, vocabulary_size)
        if tie_embeddings:
            self._softmax_linear.weight = self._embedder.get_weight()

        self._softmax = LogSoftmax(dim=2)

    def forward(self, source: Dict[str, torch.LongTensor]):
        embeddings = self._embedder(source)
        source = {"x": embeddings, "lengths": source["lengths"]}
        contextual_embeddings = self._contextualizer(source)
        contextual_embeddings = self._dropout(contextual_embeddings)
        result = self._softmax(self._softmax_linear(contextual_embeddings))
        result = torch.transpose(result, 0, 1)
        result = torch.transpose(result, 1, 2)
        return result


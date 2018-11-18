from typing import Dict

import torch
from torch.nn import Dropout, Linear, LogSoftmax, NLLLoss, Softmax
from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary, DEFAULT_PADDING_TOKEN
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.nn.util import get_text_field_mask

from rulm.nn.models.seq2seq_encoder import Seq2SeqEncoder
from rulm.nn.models.lstm_encoder import LstmEncoder

@Model.register("unidirectional_language_model")
class UnidirectionalLanguageModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TokenEmbedder,
                 contextualizer: Seq2SeqEncoder,
                 dropout: float = None,
                 use_custom_embedder_weights: bool = False,
                 tie_embeddings: bool = True):
        super().__init__(vocab)

        self._embedder = embedder
        if use_custom_embedder_weights:
            self._embedder.weight.data.uniform_(-1., 1.)
        self._contextualizer = contextualizer
        self._context_dim = contextualizer.get_output_dim()

        if dropout:
            self._dropout = Dropout(dropout)
        else:
            self._dropout = lambda x: x

        self._softmax_linear = Linear(self._context_dim, vocab.get_vocab_size())
        if tie_embeddings:
            self._softmax_linear.weight = self._embedder.weight

        self._softmax = LogSoftmax(dim=2)

    def forward(self, input_tokens: Dict[str, torch.Tensor],
                      output_tokens: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, max_length)
        source = input_tokens["tokens"]
        source = source.flip(0)
        mask = source > 0

        # Shape: (batch_size, max_length, embedding_size)
        embeddings = self._embedder(source)

        # Shape: (batch_size, max_lenght, context_dim)
        contextual_embeddings = self._contextualizer(embeddings, mask)
        contextual_embeddings = self._dropout(contextual_embeddings)

        # Shape: (batch_size, max_length, vocab_size)
        linears = self._softmax_linear(contextual_embeddings)
        logits = self._softmax(linears)

        # Shape: (batch_size, vocab_size, max_length)
        logits = torch.transpose(logits, 1, 2)

        result = {"logits": logits}

        criterion = NLLLoss(ignore_index=self.vocab.get_token_index(DEFAULT_PADDING_TOKEN))
        if output_tokens:
            target = output_tokens["tokens"]
            target = target.flip(0)
            loss = criterion(logits, target)
            result["loss"] = loss
        return result


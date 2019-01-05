from typing import Dict

import torch
from torch.nn import Dropout, Linear, LogSoftmax, NLLLoss
from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary, DEFAULT_PADDING_TOKEN
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules import Seq2SeqEncoder


@Model.register("encoder_only")
class EncoderOnlyLanguageModel(Model):
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

    def forward(self,
                source_tokens: Dict[str, torch.Tensor],
                target_tokens: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, max_length)
        source = source_tokens["tokens"]
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

        result = {"logits": logits}

        criterion = NLLLoss(ignore_index=self.vocab.get_token_index(DEFAULT_PADDING_TOKEN))
        if target_tokens:
            target = target_tokens["tokens"]
            target = target.flip(0)
            loss = criterion(logits.transpose(1, 2), target)
            result["loss"] = loss
        return result


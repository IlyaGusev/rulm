from typing import Dict

import numpy as np
import torch
from torch.nn.functional import linear, log_softmax
from torch.nn import Dropout, LogSoftmax, NLLLoss
from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary, DEFAULT_PADDING_TOKEN
from allennlp.modules import TextFieldEmbedder
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss


class SoftmaxLoss(torch.nn.Module):
    def __init__(self,
                 num_words: int,
                 embedding_dim: int,
                 padding_index: int = 0) -> None:
        super().__init__()

        self.softmax_w = torch.nn.Parameter(torch.Tensor(num_words, embedding_dim))
        self.softmax_b = torch.nn.Parameter(torch.Tensor(num_words))
        self._softmax_func = LogSoftmax(dim=-1)
        self._padding_index = padding_index
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1. / np.sqrt(self.softmax_w.size(1))
        self.softmax_w.data.uniform_(-stdv, stdv)
        self.softmax_b.data.uniform_(-stdv, stdv)

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self._softmax_func(linear(embeddings, self.softmax_w, self.softmax_b))
        criterion = NLLLoss(ignore_index=self._padding_index, reduction="sum")
        return criterion(logits, targets.long())


@Model.register("encoder_only")
class EncoderOnlyLanguageModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 contextualizer: Seq2SeqEncoder,
                 dropout: float = None,
                 tie_embeddings: bool = True,
                 num_samples: int = None):
        super().__init__(vocab)

        self._embedder = embedder
        self._contextualizer = contextualizer
        self._context_dim = contextualizer.get_output_dim()
        self._dropout = Dropout(dropout) if dropout else lambda x: x

        vocab_size = self.vocab.get_vocab_size()
        padding_index = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN)
        if num_samples:
            self._softmax_loss = SampledSoftmaxLoss(vocab_size, self._context_dim, num_samples)
        else:
            self._softmax_loss = SoftmaxLoss(vocab_size, self._context_dim, padding_index)

        self._tie_embeddings = tie_embeddings
        if self._tie_embeddings:
            assert "token_embedder_tokens" in dict(self._embedder.named_children())
            source_token_embedder = dict(self._embedder.named_children())["token_embedder_tokens"]
            assert self._softmax_loss.softmax_w.size() == source_token_embedder.weight.size()
            self._softmax_loss.softmax_w = source_token_embedder.weight

    def forward(self,
                source_tokens: Dict[str, torch.Tensor],
                target_tokens: Dict[str, torch.Tensor]=None,
                **kwargs) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, max_length)
        source = source_tokens["tokens"]
        mask = source > 0

        # Shape: (batch_size, max_length, embedding_size)
        embeddings = self._embedder(source_tokens)

        # Shape: (batch_size, max_length, context_dim)
        contextual_embeddings = self._contextualizer(embeddings, mask)
        contextual_embeddings = self._dropout(contextual_embeddings)

        result = dict()
        if target_tokens:
            targets = target_tokens["tokens"]
            targets = targets.view(-1)
            mask = targets > 0
            masked_targets = targets.masked_select(mask)
            lined_embeddings = contextual_embeddings.view(-1, self._context_dim)
            masked_embeddings = lined_embeddings.masked_select(mask.unsqueeze(-1))
            masked_embeddings = masked_embeddings.view(-1, self._context_dim)
            if self.training:
                loss = self._softmax_loss(masked_embeddings, masked_targets)
                num_targets = torch.sum(mask.long())
                result["loss"] = loss / num_targets.float()
            else:
                logits = self._get_logits(masked_embeddings)
                criterion = NLLLoss(ignore_index=self.vocab.get_token_index(DEFAULT_PADDING_TOKEN))
                result["loss"] = criterion(logits, masked_targets.long())
        if not self.training:
            result["logits"] = self._get_logits(contextual_embeddings)
        return result

    def _get_logits(self, embeddings):
        linears = linear(embeddings, self._softmax_loss.softmax_w, self._softmax_loss.softmax_b)
        return log_softmax(linears, dim=-1)

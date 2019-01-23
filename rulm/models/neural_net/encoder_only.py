from typing import Dict

import numpy as np
import torch
from torch.nn.functional import linear, log_softmax, embedding
from torch.nn import Dropout, LogSoftmax, NLLLoss
from allennlp.common import Params
from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary, DEFAULT_PADDING_TOKEN
from allennlp.modules import TextFieldEmbedder, TimeDistributed, Seq2SeqEncoder
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.modules.token_embedders import Embedding, TokenEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.nn.util import combine_initial_dims, uncombine_initial_dims


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


@TokenEmbedder.register("embedding_with_dropout")
class EmbeddingWithDropout(Embedding):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 dropout: float = None,
                 projection_dim: int = None,
                 weight: torch.FloatTensor = None,
                 padding_index: int = None,
                 trainable: bool = True,
                 max_norm: float = None,
                 norm_type: float = 2.,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False) -> None:
        Embedding.__init__(self,
                           num_embeddings=num_embeddings,
                           embedding_dim=embedding_dim,
                           projection_dim=projection_dim,
                           weight=weight,
                           padding_index=padding_index,
                           trainable=trainable,
                           max_norm=max_norm,
                           norm_type=norm_type,
                           scale_grad_by_freq=scale_grad_by_freq,
                           sparse=sparse)
        self.dropout = dropout

    def forward(self, inputs):
        original_size = inputs.size()
        inputs = combine_initial_dims(inputs)

        if self.dropout and self.training:
            mask = self.weight.data.new().resize_((self.weight.size(0), 1)).bernoulli_(1 - self.dropout)\
                       .expand_as(self.weight) / (1 - self.dropout)
            masked_embed_weight = mask * self.weight
        else:
            masked_embed_weight = self.weight

        embedded = embedding(inputs, masked_embed_weight,
                             max_norm=self.max_norm,
                             norm_type=self.norm_type,
                             scale_grad_by_freq=self.scale_grad_by_freq,
                             sparse=self.sparse)

        embedded = uncombine_initial_dims(embedded, original_size)

        if self._projection:
            projection = self._projection
            for _ in range(embedded.dim() - 2):
                projection = TimeDistributed(projection)
            embedded = projection(embedded)
        return embedded

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'Embedding':
        num_embeddings = params.pop_int('num_embeddings', None)
        vocab_namespace = params.pop("vocab_namespace", "tokens")
        if num_embeddings is None:
            num_embeddings = vocab.get_vocab_size(vocab_namespace)
        embedding_dim = params.pop_int('embedding_dim')
        pretrained_file = params.pop("pretrained_file", None)
        projection_dim = params.pop_int("projection_dim", None)
        trainable = params.pop_bool("trainable", True)
        padding_index = params.pop_int('padding_index', None)
        max_norm = params.pop_float('max_norm', None)
        norm_type = params.pop_float('norm_type', 2.)
        scale_grad_by_freq = params.pop_bool('scale_grad_by_freq', False)
        sparse = params.pop_bool('sparse', False)
        dropout = params.pop_float('dropout', None)
        params.assert_empty(cls.__name__)
        weight = _read_pretrained_embeddings_file(pretrained_file, embedding_dim,
                                                  vocab, vocab_namespace) if pretrained_file else None
        return cls(num_embeddings=num_embeddings,
                   embedding_dim=embedding_dim,
                   projection_dim=projection_dim,
                   weight=weight,
                   padding_index=padding_index,
                   trainable=trainable,
                   max_norm=max_norm,
                   norm_type=norm_type,
                   scale_grad_by_freq=scale_grad_by_freq,
                   sparse=sparse,
                   dropout=dropout)


@Model.register("encoder_only")
class EncoderOnlyLanguageModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 contextualizer: Seq2SeqEncoder,
                 dropout: float = None,
                 tie_embeddings: bool = True,
                 num_samples: int = None,
                 use_variational_dropout: bool = False):
        super().__init__(vocab)

        self._embedder = embedder
        self._contextualizer = contextualizer
        self._context_dim = contextualizer.get_output_dim()

        if use_variational_dropout:
            self._dropout = InputVariationalDropout(dropout) if dropout else lambda x: x
        else:
            self._dropout = Dropout(dropout) if dropout else lambda x: x

        vocab_size = self.vocab.get_vocab_size()
        padding_index = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN)
        if num_samples:
            self._softmax_loss = SampledSoftmaxLoss(vocab_size, self._context_dim, num_samples)
        else:
            self._softmax_loss = SoftmaxLoss(vocab_size, self._context_dim, padding_index)

        self._tie_embeddings = tie_embeddings
        if self._tie_embeddings:
            embedder_children = dict(self._embedder.named_children())
            word_embedder = embedder_children["token_embedder_tokens"]
            assert self._softmax_loss.softmax_w.size() == word_embedder.weight.size()
            self._softmax_loss.softmax_w = word_embedder.weight

    def forward(self,
                source_tokens: Dict[str, torch.Tensor],
                target_tokens: Dict[str, torch.Tensor]=None,
                **kwargs) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, max_length)
        source = source_tokens["tokens"]
        mask = source > 0

        # Shape: (batch_size, max_length, embedding_size)
        embeddings = self._embedder(source_tokens)
        embeddings = self._dropout(embeddings)

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
            loss = self._softmax_loss(masked_embeddings, masked_targets)
            num_targets = torch.sum(mask.long())
            result["loss"] = loss / num_targets.float()
        if not self.training:
            result["logits"] = self._get_logits(contextual_embeddings)
        return result

    def _get_logits(self, embeddings):
        linears = linear(embeddings, self._softmax_loss.softmax_w, self._softmax_loss.softmax_b)
        return log_softmax(linears, dim=-1)

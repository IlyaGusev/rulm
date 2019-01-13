from typing import Dict

import torch
from torch.nn import Dropout, Linear, LogSoftmax, NLLLoss
from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary, DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
from allennlp.modules import TextFieldEmbedder
from allennlp.modules import Seq2SeqEncoder
# from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss


@Model.register("encoder_only")
class EncoderOnlyLanguageModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 contextualizer: Seq2SeqEncoder,
                 dropout: float = None,
                 tie_embeddings: bool=True):
        super().__init__(vocab)

        self._embedder = embedder
        self._contextualizer = contextualizer
        self._context_dim = contextualizer.get_output_dim()
        self._dropout = Dropout(dropout) if dropout else lambda x: x

        self._softmax_linear = Linear(self._context_dim, vocab.get_vocab_size())

        self._tie_embeddings = tie_embeddings
        if self._tie_embeddings:
            assert "token_embedder_tokens" in dict(self._embedder.named_children())
            source_token_embedder = dict(self._embedder.named_children())["token_embedder_tokens"]
            assert self._softmax_linear.weight.size() == source_token_embedder.weight.size()
            self._softmax_linear.weight = source_token_embedder.weight

        self._softmax = LogSoftmax(dim=2)

        # self._sampled_softmax_loss = SampledSoftmaxLoss(self.vocab.get_vocab_size(),
        #                                                 self._context_dim, 1024,
        #                                                 use_character_inputs=False)

    def forward(self,
                source_tokens: Dict[str, torch.Tensor],
                target_tokens: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, max_length)
        source = source_tokens["tokens"]
        mask = source > 0

        # Shape: (batch_size, max_length, embedding_size)
        embeddings = self._embedder(source_tokens)

        # Shape: (batch_size, max_length, context_dim)
        contextual_embeddings = self._contextualizer(embeddings, mask)
        contextual_embeddings = self._dropout(contextual_embeddings)

        result = dict()
        # use_sampled_softmax = True
        # if not use_sampled_softmax:
        # Shape: (batch_size, max_length, vocab_size)
        logits = self._softmax(self._softmax_linear(contextual_embeddings))
        result["logits"] = logits
        if target_tokens:
            padding_index = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN)
            criterion = NLLLoss(ignore_index=padding_index)
            target = target_tokens["tokens"]
            loss = criterion(logits.transpose(1, 2), target)
            result["loss"] = loss
        # else:
        #     targets = target_tokens["tokens"]
        #     targets = targets.view(-1)
        #     contextual_embeddings = contextual_embeddings.view(-1, self._context_dim)
        #     loss = self._sampled_softmax_loss(contextual_embeddings, targets)
        #     result["loss"] = loss
        return result


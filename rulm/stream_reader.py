from typing import Dict, List, Iterable
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.fields import TextField
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.language_modeling import LanguageModelingReader


@DatasetReader.register("lm_stream")
class LanguageModelingStreamReader(LanguageModelingReader):
    def __init__(self,
                 reverse: bool = False,
                 tokens_per_instance: int = None,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(tokens_per_instance, tokenizer, token_indexers, True)
        self.reverse = reverse

    @overrides
    def _read(self, file_path: str):
        for line in self._lines(file_path):
            if self._tokens_per_instance is None:
                yield self.text_to_instance(line)
                continue
            tokenized_text = self._tokenize(line)
            num_tokens = self._tokens_per_instance + 1
            if num_tokens >= len(tokenized_text):
                yield self._sample_to_instance(tokenized_text)
                continue
            for start in range(0, len(tokenized_text) - num_tokens, num_tokens - 1):
                end = start + num_tokens
                sample = tokenized_text[start:end]
                print(sample)
                yield self._sample_to_instance(sample)

    def text_to_instance(self,
                         text: str,
                         add_end: bool=True,
                         undo_reverse: bool=False,
                         add_begin: bool=True) -> Iterable[Instance]:
        tokens = self._tokenize(text, add_end=add_end, undo_reverse=undo_reverse, add_begin=add_begin)
        return self._sample_to_instance(tokens)

    def _tokenize(self,
                  text: str,
                  add_end: bool=True,
                  undo_reverse: bool=False,
                  add_begin: bool = True) -> List[Token]:
        tokenized_text = self._tokenizer.tokenize(text)
        tokenized_text = tokenized_text[::-1] if self.reverse and not undo_reverse else tokenized_text
        if add_begin:
            tokenized_text.insert(0, Token(START_SYMBOL))
        if add_end:
            tokenized_text.append(Token(END_SYMBOL))
        return tokenized_text

    def _sample_to_instance(self, sample: List[Token]) -> Instance:
        input_field = TextField(sample[:-1], self._token_indexers)
        output_field = TextField(sample[1:], self._token_indexers)
        return Instance({
            'source_tokens': input_field,
            'target_tokens': output_field
        })

    @staticmethod
    def _lines(file_path: str) -> Iterable[str]:
        file_path = cached_path(file_path)
        with open(file_path, "r") as text_file:
            for line in text_file:
                line = line.strip()
                yield line

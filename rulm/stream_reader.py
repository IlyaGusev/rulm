from typing import Dict, List, Iterable

from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.fields import TextField
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.language_modeling import LanguageModelingReader
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter


@DatasetReader.register("lm_stream")
class LanguageModelingStreamReader(LanguageModelingReader):
    def __init__(self,
                 reverse: bool = False,
                 tokens_per_instance: int = None,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        tokenizer = tokenizer or WordTokenizer(word_splitter=JustSpacesWordSplitter())
        super().__init__(tokens_per_instance, tokenizer, token_indexers, True)
        self._reverse = reverse

    def _read(self, file_path: str):
        for line in self._lines(file_path):
            if self._tokens_per_instance is None:
                yield self.text_to_instance(line)
                continue
            tokenized_text = self._tokenize(line)
            if self._reverse:
                tokenized_text = tokenized_text[::-1]
                tokenized_text[0] = Token(START_SYMBOL)
                tokenized_text[-1] = Token(END_SYMBOL)
            num_tokens = self._tokens_per_instance + 1
            if num_tokens >= len(tokenized_text):
                yield self._sample_to_instance(tokenized_text)
                continue
            for start in range(0, len(tokenized_text) - num_tokens, num_tokens - 1):
                end = start + num_tokens
                sample = tokenized_text[start:end]
                yield self._sample_to_instance(sample)

    def text_to_instance(self, text: str) -> Instance:
        tokens = self._tokenize(text)
        return self._sample_to_instance(tokens)

    def _tokenize(self, text: str) -> List[Token]:
        tokenized_text = self._tokenizer.tokenize(text)
        tokenized_text.insert(0, Token(START_SYMBOL))
        tokenized_text.append(Token(END_SYMBOL))
        return tokenized_text

    def _sample_to_instance(self, sample: List[Token]) -> Instance:
        result = dict()
        result['source_tokens'] = TextField(sample[:-1], self._token_indexers)
        result['target_tokens'] = TextField(sample[1:], self._token_indexers)
        result['all_tokens'] = TextField(sample, self._token_indexers)
        return Instance(result)

    @staticmethod
    def _lines(file_path: str) -> Iterable[str]:
        file_path = cached_path(file_path)
        with open(file_path, "r") as text_file:
            for line in text_file:
                line = line.strip()
                yield line

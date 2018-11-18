from typing import  Dict
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


@DatasetReader.register("stream")
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
        file_path = cached_path(file_path)
        with open(file_path, "r") as text_file:
            for line in text_file:
                line = line.strip()
                tokenized_line = self._tokenizer.tokenize(line)
                if self.reverse:
                    tokenized_line = tokenized_line[::-1]
                tokenized_line.insert(0, Token(START_SYMBOL))
                tokenized_line.append(Token(END_SYMBOL))
                if self._tokens_per_instance is not None:
                    num_tokens = self._tokens_per_instance + 1
                    for index in range(0, len(tokenized_line) - num_tokens, num_tokens - 1):
                        sample = tokenized_line[index:(index + num_tokens)]
                        yield self._sample_to_instance(sample)
                else:
                    yield self._sample_to_instance(tokenized_line)

    def _sample_to_instance(self, sample):
        y = sample[1:]
        y.append(Token(DEFAULT_PADDING_TOKEN))
        input_field = TextField(sample, self._token_indexers)
        output_field = TextField(y, self._token_indexers)
        return Instance({
            'input_tokens': input_field,
            'output_tokens': output_field
        })

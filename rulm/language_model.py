from typing import List, Callable

from rulm.vocabulary import Vocabulary


class LanguageModel:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary  # type: Vocabulary

    def train(self, inputs: List[List[str]]):
        raise NotImplementedError()

    def query(self, inputs: List[str], filter_func: Callable, map_func: Callable):
        raise NotImplementedError()

    def measure_perplexity(self, inputs: List[List[str]]):
        raise NotImplementedError()

    def numericalize_inputs(self, inputs: List[str]) -> List[int]:
        return [self.vocabulary.get_bos()] + [self.vocabulary.get_index_by_word(word) for word in inputs]

from collections import defaultdict
from typing import List, Callable, Counter, Dict, Tuple

import numpy as np

from rulm.language_model import  LanguageModel
from rulm.vocabulary import Vocabulary
from rulm.filter import AlphabetOrderFilter, Filter


class NGramLanguageModel(LanguageModel):
    def __init__(self, n: int, vocabulary: Vocabulary,
                 filter_func: Filter,
                 map_func: Callable=lambda x: x):
        self.transitions = defaultdict(Counter)  # type: Dict[Tuple, Dict[int, int]]
        self.n = n  # type: int
        LanguageModel.__init__(self, vocabulary, filter_func, map_func)

    def _collect_transitions(self, indices: List[int]) -> None:
        for i in range(len(indices) - self.n + 1):
            last_word_index = i + self.n - 2
            current_words = indices[i: last_word_index + 1]
            next_word = indices[last_word_index + 1]
            for shift in range(len(current_words)-1):
                self.transitions[tuple(current_words[shift:])][next_word] += 1

    def train(self, inputs: List[List[str]]):
        for sentence in inputs:
            indices = self._numericalize_inputs(sentence)
            indices = [self.vocabulary.get_bos()] * (self.n - 2) + indices
            indices.append(self.vocabulary.get_eos())
            self._collect_transitions(indices)

    def predict(self, indices: List[int]) -> List[float]:
        indices = [self.vocabulary.get_bos()] * (self.n - 2) + indices
        counts = None
        shift = 1
        while not counts:
            counts = self.transitions.get(tuple(indices[-self.n+shift:]), None)  # type: Dict[int, int]
            shift += 1
        counts = dict(filter(self.filter_func, counts.items()))
        counts = dict(map(self.map_func, counts.items()))
        s = sum(counts.values())
        probabilities = np.zeros(self.vocabulary.size(), dtype=np.float)
        for index, count in counts.items():
            probabilities[index] = count/s
        return probabilities

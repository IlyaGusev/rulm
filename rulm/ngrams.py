from collections import defaultdict
from typing import List, Callable, Counter, Dict, Tuple

from rulm.language_model import  LanguageModel
from rulm.vocabulary import Vocabulary


class NGramLanguageModel(LanguageModel):
    def __init__(self, n: int, vocabulary: Vocabulary):
        self.transitions = defaultdict(Counter)  # type: Dict[Tuple, Dict[int, int]]
        self.n = n  # type: int
        LanguageModel.__init__(self, vocabulary)

    def _collect_transitions(self, indices: List[int]) -> None:
        for i in range(len(indices) - self.n + 1):
            last_word_index = i + self.n - 2
            current_words = tuple(indices[i: last_word_index + 1])
            next_word = indices[last_word_index + 1]
            self.transitions[current_words][next_word] += 1

    def train(self, inputs: List[List[str]]):
        for sentence in inputs:
            indices = self.numericalize_inputs(sentence)
            indices.append(self.vocabulary.get_eos())
            self._collect_transitions(indices)

    def query(self, inputs: List[str], filter_func: Callable=lambda x: x,
              map_func: Callable=lambda x: x) -> Dict[str, float]:
        indices = self.numericalize_inputs(inputs)
        counts = self.transitions[tuple(indices[-self.n+1:])]  # type: Dict[int, int]
        counts = dict(filter(filter_func, counts.items()))
        counts = dict(map(map_func, counts.items()))
        s = sum(counts.values())
        counts = dict(map(lambda x: (self.vocabulary.get_word_by_index(x[0]), x[1]/s),
                          counts.items()))  # type: Dict[int, float]
        return counts

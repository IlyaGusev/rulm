from collections import defaultdict
from typing import List, Callable, Counter, Dict, Tuple

import numpy as np

from rulm.language_model import  LanguageModel
from rulm.vocabulary import Vocabulary
from rulm.transform import Transform


class NGramLanguageModel(LanguageModel):
    def __init__(self, n: int, vocabulary: Vocabulary,
                 transforms: Tuple[Transform]=tuple(),
                 interpolation_lambdas: Tuple[float]=None):
        self.transitions = defaultdict(Counter)  # type: Dict[Tuple, Dict[int, int]]
        self.n = n  # type: int
        self.interpolation_lambdas = interpolation_lambdas  # type: Tuple[float]
        LanguageModel.__init__(self, vocabulary, transforms)

    def _collect_transitions(self, indices: List[int]) -> None:
        for i in range(len(indices) - self.n + 1):
            last_word_index = i + self.n - 2
            current_words = indices[i: last_word_index + 1]
            next_word = indices[last_word_index + 1]
            for shift in range(len(current_words) + 1):
                context = tuple(current_words[shift:])
                self.transitions[context][next_word] += 1

    def train(self, inputs: List[List[str]]):
        for sentence in inputs:
            indices = self._numericalize_inputs(sentence)
            indices = [self.vocabulary.get_bos()] * (self.n - 2) + indices
            indices.append(self.vocabulary.get_eos())
            self._collect_transitions(indices)

    def predict(self, indices: List[int]) -> List[float]:
        indices = [self.vocabulary.get_bos()] * (self.n - 2) + indices
        probabilities = np.zeros(len(self.vocabulary), dtype=np.float)
        shift_range = range(1, self.n+1) if self.interpolation_lambdas else (1,)
        interpolation_lambdas = self.interpolation_lambdas if self.interpolation_lambdas else (1.0, )
        for shift in shift_range:
            context = tuple(indices[-self.n+shift:]) if shift != self.n else tuple()
            counts = self.transitions.get(context, None)  # type: Dict[int, int]
            s = sum(counts.values()) if counts else 0

            shift_probabilities = np.zeros(len(self.vocabulary), dtype=np.float)
            for index, count in counts.items():
                shift_probabilities[index] = count/s if s != 0 else 0
            probabilities += shift_probabilities * interpolation_lambdas[shift-1]
        return probabilities

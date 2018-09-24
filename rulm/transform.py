import numpy as np
from typing import Tuple, List

from rulm.vocabulary import Vocabulary

class Transform:
    def __call__(self, probabilities: List[float]):
        raise NotImplementedError()

    def advance(self, index: int):
        raise NotImplementedError()

class TopKTransform(Transform):
    def __init__(self, k):
        self.k = k

    def __call__(self, probabilities: List[float]) -> List[float]:
        indices = set(np.argpartition(probabilities, -self.k)[-self.k:])
        for i in range(len(probabilities)):
            if i not in indices:
                probabilities[i] = 0.
        return probabilities

    def advance(self, index: int):
        pass

class AlphabetOrderTransform(Transform):
    def __init__(self, vocabulary: Vocabulary, language: str="ru"):
        assert language in ("ru", "en"), "Bad language for filter"
        self.language = language
        if language == "ru":
            self.current_letter = "а"
        elif language == "en":
            self.current_letter = "a"
        self.vocabulary = vocabulary

    def __call__(self, probabilities: List[float]) -> List[float]:
        for index, prob in enumerate(probabilities):
            first_letter = self.vocabulary.get_word_by_index(index)[0].lower()
            if first_letter != self.current_letter and index >= 4:
                probabilities[index] = 0.
        return probabilities

    def advance(self, index: int) -> None:
        self.current_letter = chr(ord(self.current_letter) + 1)
        if self.language == "ru" and self.current_letter > 'я':
            self.current_letter = 'а'
        if self.language == "en" and self.current_letter > 'z':
            self.current_letter = 'a'

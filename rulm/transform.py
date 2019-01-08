import numpy as np
from typing import List, Iterable

from allennlp.data.vocabulary import Vocabulary
from allennlp.common.util import END_SYMBOL
from allennlp.common.registrable import Registrable


class Transform(Registrable):
    def __call__(self, probabilities: List[float]):
        raise NotImplementedError()

    def advance(self, index: int):
        raise NotImplementedError()


@Transform.register("top-k")
class TopKTransform(Transform):
    def __init__(self, k):
        self.k = k

    def __call__(self, probabilities: np.array) -> Iterable[float]:
        if probabilities.shape[0] < self.k:
            return probabilities
        indices = set(np.argpartition(probabilities, -self.k)[-self.k:])
        for i in range(len(probabilities)):
            if i not in indices:
                probabilities[i] = 0.
        return probabilities

    def advance(self, index: int):
        pass


@Transform.register("alphabet")
class AlphabetOrderTransform(Transform):
    def __init__(self, vocab: Vocabulary, language: str="ru", start_letter: str=None):
        assert language in ("ru", "en"), "Bad language for filter"
        self.language = language
        if start_letter:
            self.current_letter = start_letter
        else:
            if language == "ru":
                self.current_letter = "а"
            elif language == "en":
                self.current_letter = "a"
        self.vocab = vocab

    def __call__(self, probabilities: np.array) -> List[float]:
        for index, prob in enumerate(probabilities):
            first_letter = self.vocab.get_token_from_index(index)[0].lower()
            if first_letter != self.current_letter and not index == self.vocab.get_token_index(END_SYMBOL):
                probabilities[index] = 0.
        return probabilities

    def advance(self, index: int) -> None:
        self.current_letter = chr(ord(self.current_letter) + 1)
        if self.language == "ru" and self.current_letter > 'я':
            self.current_letter = 'а'
        if self.language == "en" and self.current_letter > 'z':
            self.current_letter = 'a'

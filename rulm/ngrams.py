from collections import defaultdict
from typing import List, Callable, Counter, Dict, Tuple
import gzip

import numpy as np

from rulm.language_model import  LanguageModel
from rulm.vocabulary import Vocabulary
from rulm.transform import Transform


class NGramLanguageModel(LanguageModel):
    def __init__(self, n: int, vocabulary: Vocabulary,
                 transforms: Tuple[Transform]=tuple(),
                 interpolation_lambdas: Tuple[float]=None,
                 reverse: bool=False):
        self.n_grams = [defaultdict(float) for _ in range(n+1)]  # type: List[Dict[int, int]]
        self.n = n  # type: int
        self.interpolation_lambdas = interpolation_lambdas  # type: Tuple[float]
        self.reverse = reverse  # type: bool
        LanguageModel.__init__(self, vocabulary, transforms)

    def _collect_n_grams(self, indices: List[int]) -> None:
        l = len(indices)
        for n in range(self.n + 1):
            for i in range(min(l - n + 1, l)):
                n_gram = tuple(indices[i:i+n])
                self.n_grams[n][n_gram] += 1.0

    def train(self, inputs: List[List[str]]):
        for sentence in inputs:
            if self.reverse:
                sentence = sentence[::-1]
            indices = self._numericalize_inputs(sentence)
            indices.append(self.vocabulary.get_eos())
            self._collect_n_grams(indices)
        self._normalize()

    def _normalize(self):
        for n in range(self.n, 0, -1):
            current_n_grams = self.n_grams[n]
            for words, count in current_n_grams.items():
                prev_order_n_gram_count = self.n_grams[n-1][words[:-1]]
                current_n_grams[words] = count / prev_order_n_gram_count
        self.n_grams[0][tuple()] = 1.0

    def predict(self, indices: List[int]) -> List[float]:
        probabilities = np.zeros(len(self.vocabulary), dtype=np.float)
        if not self.interpolation_lambdas:
            self.interpolation_lambdas = (1.0, ) + tuple((0. for _ in range(self.n-1)))
        context = tuple(indices[-self.n+1:])
        for shift in range(self.n):
            wanted_context_length = self.n-1-shift
            if wanted_context_length > len(context) or self.interpolation_lambdas[shift] == 0.:
                continue
            difference = len(context) - wanted_context_length
            context = tuple(context[difference:])
            n = len(context) + 1
            for index in range(probabilities.shape[0]):
                n_gram = context + (index,)
                p = self.n_grams[n][n_gram]
                probabilities[index] += p * self.interpolation_lambdas[shift]
        return probabilities

    def save(self, path:str) -> None:
        assert path.endswith(".arpa") or path.endswith(".arpa.gzip")
        file_open = gzip.open if path.endswith(".gzip") else open
        with file_open(path, "wt", encoding="utf-8") as w:
            w.write("\\data\\\n")
            for n in range(1, self.n+1):
                w.write("ngram {}={}\n".format(n, len(self.n_grams[n])))
            w.write("\n")
            for n in range(1, self.n+1):
                w.write("\\{}-grams:\n".format(n))
                for words, p in self.n_grams[n].items():
                    words = " ".join(map(self.vocabulary.get_word_by_index, words))
                    w.write("{:.4f}\t{}\n".format(np.log10(p), words))
                w.write("\n")
            w.write("\\end\\\n")

    def load(self, path: str) -> None:
        self.n_grams[0][tuple()] = 1.
        file_open = gzip.open if path.endswith(".gzip") else open
        with file_open(path, "rt", encoding="utf-8") as r:
            line = next(r)
            assert line.strip() == "\\data\\", "Invalid ARPA: missing \\data\\"
            max_n = 0
            for line in r:
                if not line.startswith("ngram"):
                    break
                n = int(line.strip().split()[1].split("=")[0])
                max_n = max(max_n, n)
            assert max_n == self.n, "Invalid ARPA: wrong max n"
            for n in range(1, self.n+1):
                self.n_grams[n] = defaultdict(float)
                while not line.strip():
                    line = next(r)
                assert line.strip() == "\\{}-grams:".format(n), "Invalid ARPA: wrong {}-gram start".format(n)
                for line in r:
                    if not line.strip():
                        break
                    tokens = line.strip().split()
                    p = float(tokens[0])
                    words = tuple(map(self.vocabulary.get_index_by_word, tokens[1:n+1]))
                    self.n_grams[n][words] = np.power(10, p)
            while not line.strip():
                line = next(r)
            assert line.strip() == "\\end\\", "Invalid ARPA: \\end\\ invalid or missing"

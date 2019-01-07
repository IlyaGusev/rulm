import os
from collections import defaultdict
from typing import List, Tuple, Type, Iterable
import gzip

import pygtrie
import numpy as np
from scipy.optimize import minimize
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.util import END_SYMBOL
from allennlp.common.params import Params

from rulm.language_model import LanguageModel
from rulm.transform import Transform
from rulm.settings import DEFAULT_N_GRAM_WEIGHTS


class NGramContainer:
    def __getitem__(self, n_gram: Iterable[int]):
        raise NotImplementedError()

    def __setitem__(self, n_gram: Iterable[int], value: float):
        raise NotImplementedError()

    def __delitem__(self, n_gram: Iterable[int]):
        raise NotImplementedError()

    def __contains__(self, n_gram: Iterable[int]):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()

    def items(self):
        raise NotImplementedError()


class DictNGramContainer(NGramContainer):
    def __init__(self):
        self.data = defaultdict(float)

    def __getitem__(self, n_gram: Iterable[int]):
        return self.data[tuple(n_gram)]

    def __setitem__(self, n_gram: Iterable[int], value: float):
        self.data[tuple(n_gram)] = value

    def __delitem__(self, n_gram: Iterable[int]):
        del self.data[tuple(n_gram)]

    def __contains__(self, n_gram: Iterable[int]):
        return tuple(n_gram) in self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.data)

    def items(self):
        return self.data.items()


class TrieNGramContainer(NGramContainer):
    def __init__(self):
        self.data = pygtrie.Trie()
        self.n = None

    def __getitem__(self, n_gram: List[int]):
        return self.data[tuple(n_gram)] if n_gram in self.data else 0.

    def __setitem__(self, n_gram: List[int], value: float):
        self.data[tuple(n_gram)] = value

    def __delitem__(self, n_gram: Iterable[int]):
        del self.data[tuple(n_gram)]

    def __contains__(self, n_gram: Iterable[int]):
        return tuple(n_gram) in self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.data)

    def items(self):
        return self.data.items()


@LanguageModel.register("n_gram")
class NGramLanguageModel(LanguageModel):
    def __init__(self,
                 n: int,
                 vocab: Vocabulary,
                 transforms: Tuple[Transform]=tuple(),
                 reverse: bool=False,
                 cutoff_count: int=None,
                 interpolation_lambdas: Tuple[float, ...]=None,
                 container: Type[NGramContainer]=DictNGramContainer):
        self.n_grams = tuple(container() for _ in range(n+1))  # type: List[NGramContainer]
        self.n = n  # type: int
        self.cutoff_count = cutoff_count  # type: int

        self.interpolation_lambdas = interpolation_lambdas  # type: Tuple[float]
        if not interpolation_lambdas:
            self.interpolation_lambdas = np.zeros(self.n + 1, dtype=np.float64)
            self.interpolation_lambdas[-1] = 1.0
        else:
            self.interpolation_lambdas = np.array(interpolation_lambdas)
        assert n + 1 == len(self.interpolation_lambdas)

        LanguageModel.__init__(self, vocab, transforms, reverse)

    def _collect_n_grams(self, indices: List[int]) -> None:
        count = len(indices)
        for n in range(self.n + 1):
            for i in range(min(count - n + 1, count)):
                n_gram = indices[i:i+n]
                self.n_grams[n][n_gram] += 1.0

    def train(self,
              inputs: Iterable[List[str]],
              train_params: Params=Params({}),
              serialization_dir: str = None,
              report_every: int=10000):
        sentence_number = 0
        for sentence in inputs:
            indices = self._numericalize_inputs(sentence)
            eos_index = self.vocab.get_token_index(END_SYMBOL)
            indices.append(eos_index)
            self._collect_n_grams(indices)
            sentence_number += 1
            if sentence_number % report_every == 0:
                print("Train: {} sentences processed".format(sentence_number))
        if serialization_dir:
            self.save_weights(os.path.join(serialization_dir, DEFAULT_N_GRAM_WEIGHTS))

    def train_file(self,
                   file_name: str,
                   train_params: Params=Params({}),
                   serialization_dir: str=None,
                   **kwargs):
        assert os.path.exists(file_name)
        sentences = self._parse_file_for_sentences(file_name)
        self.train(sentences, train_params, serialization_dir=None)
        print("Train: normalizng...")
        self.normalize()
        if serialization_dir:
            self.save_weights(os.path.join(serialization_dir, DEFAULT_N_GRAM_WEIGHTS))

    def normalize(self):
        if self.cutoff_count:
            for n in range(1, self.n+1):
                current_n_grams = self.n_grams[n]
                for words, count in tuple(current_n_grams.items()):
                    if count < self.cutoff_count:
                        del current_n_grams[words]
        for n in range(self.n, 0, -1):
            current_n_grams = self.n_grams[n]
            for words, count in current_n_grams.items():
                prev_order_n_gram_count = self.n_grams[n-1][words[:-1]]
                current_n_grams[words] = count / prev_order_n_gram_count
        self.n_grams[0][tuple()] = 1.0

    def predict(self, indices: List[int]) -> np.ndarray:
        step_probabilities = self._get_step_probabilities(indices)
        probabilities = self.interpolation_lambdas.dot(step_probabilities)
        norm_proba = probabilities / np.sum(probabilities)
        return norm_proba

    def _get_step_probabilities(self, indices: List[int]) -> np.ndarray:
        vocab_size = self.vocab.get_vocab_size()
        context = tuple(indices[-self.n+1:])
        step_probabilities = np.zeros((self.n + 1, vocab_size), dtype=np.float64)
        for shift in range(self.n):
            current_n = self.n - shift
            wanted_context_length = current_n - 1
            if wanted_context_length > len(context):
                continue
            start_index = len(context) - wanted_context_length
            wanted_context = context[start_index:]
            for index in range(vocab_size):
                n_gram = wanted_context + (index,)
                p = self.n_grams[current_n][n_gram] if n_gram in self.n_grams[current_n] else 0.
                step_probabilities[current_n, index] = p
        step_probabilities[0].fill(1.0 / vocab_size)
        return step_probabilities

    def estimate_parameters(self, inputs: Iterable[str]):
        samples = []
        for sentence in inputs:
            words = sentence.strip().split()
            sentence_indices = self._numericalize_inputs(words)
            sentence_indices.append(self.vocab.get_token_index(END_SYMBOL))
            for i in range(1, len(sentence_indices) + 1):
                indices = sentence_indices[:i]
                true_index = indices[-1]
                context = indices[:-1]
                step_probabilities = self._get_step_probabilities(context)
                samples.append((step_probabilities, true_index))

        def sum_log_likelihood(interpolation_lambdas):
            s = 0
            for step_probabilities, true_index in samples:
                probabilities = interpolation_lambdas.dot(step_probabilities)
                norm_proba = probabilities / np.sum(probabilities)
                s -= np.log(norm_proba[true_index])
            return s

        def sum_one_constraint(x):
            return np.sum(x) - 1

        opt = minimize(
            sum_log_likelihood,
            self.interpolation_lambdas,
            constraints=[{'type': 'eq', 'fun': sum_one_constraint}],
            bounds=[(0, 1) for _ in self.interpolation_lambdas]
        )

        print(opt.x, np.exp(sum_log_likelihood(opt.x)/len(samples)), len(samples))
        self.interpolation_lambdas = opt.x

    def save_weights(self, path: str) -> None:
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
                    words = " ".join(map(self.vocab.get_token_from_index, words))
                    w.write("{:.4f}\t{}\n".format(np.log10(p), words))
                w.write("\n")
            w.write("\\end\\\n")

    @classmethod
    def _load(cls,
              params: Params,
              vocab: Vocabulary,
              serialization_dir: str,
              weights_file: str = None,
              cuda_device: int = -1):
        model = NGramLanguageModel.from_params(params, vocab=vocab)
        weights_file = weights_file or os.path.join(serialization_dir, DEFAULT_N_GRAM_WEIGHTS)
        model.load_weights(weights_file)
        return model

    def load_weights(self, path: str):
        self.n_grams[0][tuple()] = 1.
        file_open = gzip.open if path.endswith(".gzip") else open
        with file_open(path, "rt", encoding="utf-8") as r:
            line = next(r)
            while not line.strip():
                line = next(r)
            assert line.strip() == "\\data\\", "Invalid ARPA: missing \\data\\"
            max_n = 0
            for line in r:
                if not line.startswith("ngram"):
                    break
                n = int(line.strip().split()[1].split("=")[0])
                max_n = max(max_n, n)
            assert max_n == self.n, "Invalid ARPA: wrong max n"
            for n in range(1, self.n + 1):
                while not line.strip():
                    line = next(r)
                assert line.strip() == "\\{}-grams:".format(n), "Invalid ARPA: wrong {}-gram start".format(n)
                for line in r:
                    if not line.strip():
                        break
                    tokens = line.strip().split()
                    p = float(tokens[0])
                    words = tuple(map(self.vocab.get_token_index, tokens[1:n + 1]))
                    self.n_grams[n][words] = np.power(10, p)
            while not line.strip():
                line = next(r)
            assert line.strip() == "\\end\\", "Invalid ARPA: \\end\\ invalid or missing"
        print("Load finished")

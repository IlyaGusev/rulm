from typing import List, Dict, Tuple
import os

import numpy as np

from rulm.vocabulary import Vocabulary
from rulm.transform import Transform, TopKTransform
from rulm.beam import BeamSearch


class PerplexityState:
    def __init__(self):
        self.word_count = 0
        self.zeroprobs_count = 0
        self.unknown_count = 0
        self.avg_log_perplexity = 0.

    def add(self, word_index: int, probability: float, is_including_unk: bool, unk_index: int) -> None:
        old_word_count = self.word_count - self.zeroprobs_count - (self.unknown_count if not is_including_unk else 0)
        self.word_count += 1

        if word_index == unk_index:
            self.unknown_count += 1
            if not is_including_unk:
                return

        if probability == 0.:
            self.zeroprobs_count += 1
            return

        log_prob = -np.log(probability)
        true_word_count = self.word_count - self.zeroprobs_count - (self.unknown_count if not is_including_unk else 0)

        prev_avg = self.avg_log_perplexity * old_word_count / true_word_count
        self.avg_log_perplexity = prev_avg + log_prob / true_word_count
        return

    def __repr__(self):
        return "Avg ppl: {}, zeroprobs: {}, unk: {}".format(
            np.exp(self.avg_log_perplexity), self.zeroprobs_count, self.unknown_count)


class LanguageModel:
    def __init__(self, vocabulary: Vocabulary, transforms: Tuple[Transform]):
        self.vocabulary = vocabulary  # type: Vocabulary
        self.transforms = transforms  # type: List[Transform]

    def train(self, inputs: List[List[str]]):
        raise NotImplementedError()

    def normalize(self):
        raise NotImplementedError()

    def predict(self, inputs: List[int]) -> List[float]:
        raise NotImplementedError()

    def query(self, inputs: List[str]) -> Dict[str, float]:
        indices = self._numericalize_inputs(inputs)
        next_index_prediction = self.predict(indices)
        return {self.vocabulary.get_word_by_index(index): prob
                for index, prob in enumerate(next_index_prediction)}

    def train_file(self, file_name, batch_size: int=10000):
        assert os.path.exists(file_name)
        sentences = []
        batch_number = 0
        with open(file_name, "r", encoding="utf-8") as r:
            for line in r:
                words = line.strip().split()
                sentences.append(words)
                if len(sentences) == batch_size:
                    self.train(sentences)
                    batch_number += 1
                    print("Train: {} sentences processed".format(batch_number*batch_size))
                    sentences = []
        if sentences:
            self.train(sentences)
            print("Train: {} sentences processed".format(batch_number * batch_size + len(sentences)))
        print("Train: normalizng...")
        self.normalize()

    def beam_decoding(self, inputs: List[str], beam_width: int=5,
                      max_length: int=50, length_reward: float=0.0) -> List[str]:
        current_state = self._numericalize_inputs(inputs)
        beam = BeamSearch(
            eos_index=self.vocabulary.get_eos(),
            predict_func=self.predict,
            transforms=self.transforms,
            beam_width=beam_width,
            max_length=max_length,
            length_reward=length_reward)
        best_guess = beam.decode(current_state)
        return self._decipher_outputs(best_guess)

    def sample_decoding(self, inputs: List[str], k: int=5, max_length: int=30) -> List[str]:
        if k > len(self.vocabulary):
            k = len(self.vocabulary)
        current_state = self._numericalize_inputs(inputs)
        last_index = current_state[-1] if current_state else self.vocabulary.get_bos()
        while last_index != self.vocabulary.get_eos() and len(current_state) < max_length:
            next_word_probabilities = self.predict(current_state)
            for transform in self.transforms:
                next_word_probabilities = transform(next_word_probabilities)
            next_word_probabilities = TopKTransform(k)(next_word_probabilities)
            last_index = self._choose(next_word_probabilities)[0]
            for transform in self.transforms:
                transform.advance(last_index)
            current_state.append(last_index)
        outputs = self._decipher_outputs(current_state)
        return outputs

    def measure_perplexity(self, inputs: List[List[str]], state: PerplexityState,
                           is_including_unk: bool=True) -> PerplexityState:
        for sentence in inputs:
            indices = self._numericalize_inputs(sentence)
            indices.append(self.vocabulary.get_eos())
            for i, word_index in enumerate(indices[1:]):
                context = indices[:i+1]

                prediction = self.predict(context)
                state.add(word_index, prediction[word_index], is_including_unk, self.vocabulary.get_unk())
        return state

    def measure_perplexity_file(self, file_name, batch_size: int=100):
        assert os.path.exists(file_name)
        sentences = []
        ppl_state = PerplexityState()
        batch_number = 0
        with open(file_name, "r", encoding="utf-8") as r:
            for line in r:
                words = line.strip().split()
                sentences.append(words)
                if len(sentences) == batch_size:
                    ppl_state = self.measure_perplexity(sentences, ppl_state)
                    batch_number += 1
                    print("Measure_perplexity: {} sentences processed, {}".format(
                        batch_number * batch_size, ppl_state))
                    sentences = []
            if sentences:
                ppl_state = self.measure_perplexity(sentences, ppl_state)
        return ppl_state

    def _numericalize_inputs(self, words: List[str]) -> List[int]:
        return [self.vocabulary.get_bos()] + [self.vocabulary.get_index_by_word(word) for word in words]

    def _decipher_outputs(self, indices: List[int]) -> List[str]:
        return [self.vocabulary.get_word_by_index(index) for index in indices[1:-1]]

    @staticmethod
    def _choose(model: np.array, k: int=1):
        norm_model = model / np.sum(model)
        return np.random.choice(range(norm_model.shape[0]), k, p=norm_model, replace=False)


class EquiprobableLanguageModel(LanguageModel):
    def __init__(self, vocabulary: Vocabulary, transforms: Tuple[Transform]=tuple()):
        LanguageModel.__init__(self, vocabulary, transforms)

    def train(self, inputs: List[List[str]]):
        pass

    def normalize(self):
        pass

    def predict(self, inputs: List[int]):
        vocab_size = len(self.vocabulary)
        probabilities = np.full((vocab_size,), 1./(vocab_size-2))
        probabilities[self.vocabulary.get_bos()] = 0.
        probabilities[self.vocabulary.get_pad()] = 0.
        return probabilities


class VocabularyChainLanguageModel(LanguageModel):
    def __init__(self, vocabulary: Vocabulary, transforms: Tuple[Transform]=tuple()):
        LanguageModel.__init__(self, vocabulary, transforms)

    def train(self, inputs: List[List[str]]):
        pass

    def normalize(self):
        pass

    def predict(self, inputs: List[int]):
        probabilities = np.zeros(len(self.vocabulary))
        last_index = inputs[-1]
        if last_index == self.vocabulary.get_bos():
            probabilities[4] = 1.
        elif last_index != len(self.vocabulary) - 1:
            probabilities[last_index + 1] = 1.
        return probabilities

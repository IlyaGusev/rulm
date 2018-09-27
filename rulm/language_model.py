from typing import List, Callable, Dict, Tuple
import copy
import os
from collections import namedtuple

import numpy as np

from rulm.vocabulary import Vocabulary
from rulm.transform import Transform, TopKTransform


class LanguageModel:
    def __init__(self, vocabulary: Vocabulary, transforms: Tuple[Transform]):
        self.vocabulary = vocabulary  # type: Vocabulary
        self.transforms = transforms  # type: Callable

    def train(self, inputs: List[List[str]]):
        raise NotImplementedError()

    def predict(self, inputs: List[int]) -> List[float]:
        raise NotImplementedError()

    def query(self, inputs: List[str]) -> Dict[str, float]:
        indices = self._numericalize_inputs(inputs)
        next_index_prediction = self.predict(indices)
        return {self.vocabulary.get_word_by_index(index): prob
                for index, prob in enumerate(next_index_prediction)}

    def train_file(self, file_name):
        assert os.path.exists(file_name)
        sentences = []
        with open(file_name, "r", encoding="utf-8") as r:
            for line in r:
                words = line.strip().split()
                sentences.append(words)
        self.train(sentences)

    def beam_decoding(self, inputs: List[str], beam_width: int=5,
                      max_length: int=50, length_reward: float=0.0) -> List[str]:
        current_state = self._numericalize_inputs(inputs)
        BeamState = namedtuple("BeamState", "indices log_prob transforms")
        all_candidates = [BeamState(current_state, 0., self.transforms)]
        best_guess = None
        while all_candidates:
            new_candidates = []
            finished_count = 0
            for candidate in all_candidates:
                is_eos = candidate.indices[-1] == self.vocabulary.get_eos()
                is_max_length = len(candidate.indices) >= max_length
                if is_max_length and not is_eos:
                    candidate.indices.append(self.vocabulary.get_eos())
                if is_eos or is_max_length:
                    new_candidates.append(candidate)
                    finished_count += 1
                    continue
                next_word_prediction = self.predict(candidate.indices)
                for transform in candidate.transforms:
                    next_word_prediction = transform(next_word_prediction)
                top_k_prediction = TopKTransform(beam_width)(next_word_prediction)
                for index, p in enumerate(top_k_prediction):
                    if p == 0.:
                        continue
                    new_indices = candidate.indices + [index]
                    new_log_prob = candidate.log_prob - np.log(p)
                    new_state = BeamState(new_indices, new_log_prob, candidate.transforms)
                    new_candidates.append(new_state)
            new_candidates.sort(key=lambda state: state.log_prob - len(state.indices) * length_reward)
            new_candidates = new_candidates[:beam_width]
            for i, candidate in enumerate(new_candidates):
                new_transforms = copy.deepcopy(candidate.transforms)
                for transform in new_transforms:
                    transform.advance(candidate.indices[-1])
                new_candidates[i] = BeamState(candidate.indices, candidate.log_prob, new_transforms)
            if finished_count >= beam_width:
                best_guess = new_candidates[0].indices
                break
            all_candidates = new_candidates
        return self._decipher_outputs(best_guess)

    def sample_decoding(self, inputs: List[str], k: int=5) -> List[str]:
        if k > len(self.vocabulary):
            k = len(self.vocabulary)
        current_state = self._numericalize_inputs(inputs)
        last_index = current_state[-1] if current_state else self.vocabulary.get_bos()
        while last_index != self.vocabulary.get_eos():
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

    def measure_perplexity(self, inputs: List[List[str]]):
        avg_log_perplexity = 0
        sentence_count = 0
        for sentence in inputs:
            indices = self._numericalize_inputs(sentence)
            indices.append(self.vocabulary.get_eos())
            sum_log_prob = 0.
            for i, word_index in enumerate(indices[1:]):
                context = indices[:i+1]
                prediction = self.predict(context)
                if prediction[word_index] == 0.:
                    print(context, word_index)
                log_prob = -np.log(prediction[word_index])
                sum_log_prob += log_prob
            sentence_log_perplexity = sum_log_prob/(len(indices) - 1) # <bos> excluded
            avg_log_perplexity = avg_log_perplexity * sentence_count / (sentence_count + 1) + \
                                 sentence_log_perplexity / (sentence_count + 1)
            avg_perplexity = np.exp(avg_log_perplexity)
            print(avg_perplexity)
            sentence_count += 1
        return avg_perplexity

    def measure_perplexity_file(self, file_name):
        assert os.path.exists(file_name)
        sentences = []
        with open(file_name, "r", encoding="utf-8") as r:
            for line in r:
                words = line.strip().split()
                sentences.append(words)
        return self.measure_perplexity(sentences)

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

    def predict(self, inputs: List[int]):
        l = len(self.vocabulary)
        probabilities = np.full((l,), 1./(l-2))
        probabilities[self.vocabulary.get_bos()] = 0.
        probabilities[self.vocabulary.get_pad()] = 0.
        return probabilities

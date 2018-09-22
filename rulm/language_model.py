from typing import List, Callable, Dict
import copy
from collections import namedtuple

import numpy as np

from rulm.vocabulary import Vocabulary
from rulm.filter import Filter


class LanguageModel:
    def __init__(self, vocabulary: Vocabulary,
                 filter_func: Filter, map_func: Callable):
        self.vocabulary = vocabulary  # type: Vocabulary
        self.filter_func = filter_func  # type: Callable
        self.map_func = map_func  # type: Callable

    def train(self, inputs: List[List[str]]):
        raise NotImplementedError()

    def predict(self, inputs: List[int]) -> List[float]:
        raise NotImplementedError()

    def query(self, inputs: List[str]) -> Dict[str, float]:
        indices = self._numericalize_inputs(inputs)
        next_index_prediction = self.predict(indices)
        return {self.vocabulary.get_word_by_index(index): prob
                for index, prob in enumerate(next_index_prediction)}

    def beam_decoding(self, inputs: List[str], beam_width: int=5,
                      max_length: int=50, length_reward: float=0.0) -> List[str]:
        current_state = self._numericalize_inputs(inputs)
        BeamState = namedtuple("BeamState", "indices log_prob")
        all_candidates = [BeamState(current_state, 0.)]
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
                top_k = self._top_k(next_word_prediction, beam_width).items()
                for index, p in top_k:
                    if p == 0.:
                        continue
                    new_indices = candidate.indices + [index]
                    new_log_prob = candidate.log_prob - np.log(p)
                    new_state = BeamState(new_indices, new_log_prob)
                    new_candidates.append(new_state)
            new_candidates.sort(key=lambda state: state.log_prob - len(state.indices) * length_reward)
            new_candidates = new_candidates[:beam_width]
            if finished_count >= beam_width:
                best_guess = new_candidates[0].indices
                break
            all_candidates = new_candidates
        return self._decipher_outputs(best_guess)

    def sample_decoding(self, inputs: List[str], k: int=5) -> List[str]:
        current_state = self._numericalize_inputs(inputs)
        last_index = current_state[-1]
        while last_index != self.vocabulary.get_eos():
            next_word_prediction = self.predict(current_state)
            correct_indices = list(filter(self.filter_func, range(len(next_word_prediction))))
            next_word_prediction = next_word_prediction[correct_indices]
            top_k = self._top_k(next_word_prediction, k).items()
            probabilities = np.zeros(self.vocabulary.size(), dtype=np.float)
            for index, p in top_k:
                probabilities[index] = p
            last_index = self._choose(probabilities)[0]
            self.filter_func.advance(last_index)
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
                context = indices[:i + 1]
                prediction = self.predict(context)
                if prediction[word_index] == 0.:
                    print(context, word_index)
                log_prob = -np.log(prediction[word_index])
                sum_log_prob += log_prob
            sentence_log_perplexity = sum_log_prob/(len(indices) - 1) # <bos> excluded
            avg_log_perplexity = avg_log_perplexity * sentence_count / (sentence_count + 1) + \
                                 sentence_log_perplexity / (sentence_count + 1)
            avg_perplexity = np.exp(avg_log_perplexity)
            sentence_count += 1
        return avg_perplexity

    def _numericalize_inputs(self, words: List[str]) -> List[int]:
        return [self.vocabulary.get_bos()] + [self.vocabulary.get_index_by_word(word) for word in words]

    def _decipher_outputs(self, indices: List[int]) -> List[str]:
        return [self.vocabulary.get_word_by_index(index) for index in indices[1:-1]]

    @staticmethod
    def _top_k(prediction, k: int=1):
        indices = np.argpartition(prediction, -k)[-k:]
        return {index: prediction[index] for index in indices}

    @staticmethod
    def _choose(model: np.array, k: int=1):
        norm_model = model / np.sum(model)
        return np.random.choice(range(len(norm_model)), k, p=norm_model, replace=False)

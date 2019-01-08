import copy
from typing import Callable, List, Iterable

import numpy as np

from rulm.transform import Transform, TopKTransform


class BeamState:
    def __init__(self, indices, log_prob, transforms):
        self.indices = indices
        self.log_prob = log_prob
        self.transforms = transforms
        self.is_finished = False

    def score(self, length_reward: float=0.):
        return self.log_prob - len(self.indices) * length_reward

    def __repr__(self):
        return "Indices: {}".format(self.indices)


class BeamSearch:
    def __init__(self,
                 eos_index: int,
                 predict_func: Callable,
                 transforms: Iterable[Transform],
                 beam_width: int=5,
                 max_length: int=50,
                 length_reward: float=0.):
        self.predict = predict_func  # type: Callable
        self.eos_index = eos_index  # type: int
        self.transforms = transforms  # type: Iterable[Transform]
        self.beam_width = beam_width  # type: int
        self.max_length = max_length  # type: int
        self.length_reward = length_reward  # type: float

    def decode(self, inputs: List[int]) -> List[int]:
        candidates = [BeamState(inputs, 0., self.transforms)]
        finished_count = 0
        while finished_count < self.beam_width:
            finished_count = 0
            new_candidates = []
            for candidate in candidates:
                is_finished = self._is_finished_state(candidate)
                if is_finished:
                    new_candidates.append(candidate)
                    finished_count += 1
                else:
                    new_candidates += self._beam_process_candidate(candidate)

            new_candidates.sort(key=lambda x: x.score(self.length_reward))
            new_candidates = new_candidates[:self.beam_width]
            for i, candidate in enumerate(new_candidates):
                if candidate.is_finished:
                    continue
                new_transforms = copy.copy(candidate.transforms)
                for transform in new_transforms:
                    transform.advance(candidate.indices[-1])
                new_candidates[i] = BeamState(candidate.indices, candidate.log_prob, new_transforms)

            candidates = new_candidates

        assert candidates
        return candidates[0].indices

    def _is_finished_state(self, candidate: BeamState):
        if candidate.is_finished:
            return True
        is_eos = candidate.indices[-1] == self.eos_index
        is_max_length = len(candidate.indices) >= self.max_length
        if is_max_length and not is_eos:
            candidate.indices.append(self.eos_index)
        candidate.is_finished = is_max_length or is_eos
        return candidate.is_finished

    def _beam_process_candidate(self, candidate: BeamState):
        next_word_prediction = self.predict(candidate.indices)
        for transform in candidate.transforms:
            next_word_prediction = transform(next_word_prediction)
        top_k_prediction = TopKTransform(self.beam_width)(next_word_prediction)

        new_candidates = []
        for index, p in enumerate(top_k_prediction):
            if p == 0.:
                continue
            new_indices = candidate.indices + [index]
            new_log_prob = candidate.log_prob - np.log(p)
            new_state = BeamState(new_indices, new_log_prob, candidate.transforms)
            new_candidates.append(new_state)
        return new_candidates

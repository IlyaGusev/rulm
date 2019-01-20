import copy
from typing import Callable, Iterable

import numpy as np

from rulm.transform import Transform, TopKTransform


class BeamState:
    def __init__(self, text: str, log_prob: float, transforms: Iterable[Transform], step_number: int):
        self.text = text
        self.log_prob = log_prob
        self.step_number = step_number
        self.transforms = transforms
        self.is_finished = False

    def score(self, length_reward: float=0.):
        return self.log_prob - self.step_number * length_reward

    def __repr__(self):
        return "Text: {}, step_number: {}, is_finished: {}".format(self.text, self.step_number, self.is_finished)


class BeamSearch:
    def __init__(self,
                 eos_index: int,
                 predict_func: Callable,
                 index_to_text_func: Callable,
                 transforms: Iterable[Transform],
                 beam_width: int=5,
                 max_length: int=50,
                 length_reward: float=0.):
        self.predict = predict_func  # type: Callable
        self.index_to_text = index_to_text_func  # type: Callable
        self.eos_index = eos_index  # type: int
        self.transforms = transforms  # type: Iterable[Transform]
        self.beam_width = beam_width  # type: int
        self.max_length = max_length  # type: int
        self.length_reward = length_reward  # type: float

    def decode(self, text: str) -> str:
        candidates = [BeamState(text, 0., self.transforms, 0)]
        finished_count = 0
        while finished_count < self.beam_width:
            finished_count = 0
            new_candidates = []
            for candidate in candidates:
                if candidate.is_finished:
                    new_candidates.append(candidate)
                    finished_count += 1
                else:
                    new_candidates += self._beam_process_candidate(candidate)
            if len(new_candidates) == finished_count:
                break

            new_candidates.sort(key=lambda x: x.score(self.length_reward))
            candidates = new_candidates[:self.beam_width]

        assert candidates
        return candidates[0].text

    def _beam_process_candidate(self, candidate: BeamState):
        next_word_prediction = self.predict(candidate.text)
        for transform in candidate.transforms:
            next_word_prediction = transform(next_word_prediction)
        top_k_prediction = TopKTransform(self.beam_width)(next_word_prediction)

        new_candidates = []
        step_number = candidate.step_number
        is_max_length = step_number + 1 >= self.max_length
        for index, p in enumerate(top_k_prediction):
            if p == 0.:
                continue
            is_eos = index == self.eos_index
            new_text = candidate.text + " " + self.index_to_text(index) if not is_eos else candidate.text
            new_log_prob = candidate.log_prob - np.log(p)
            new_transforms = copy.copy(candidate.transforms)
            for transform in new_transforms:
                transform.advance(index)
            new_state = BeamState(new_text, new_log_prob, new_transforms, step_number + 1)
            if is_eos or is_max_length:
                new_state.is_finished = True
            new_candidates.append(new_state)
        return new_candidates

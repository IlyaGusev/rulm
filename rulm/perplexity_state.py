import numpy as np


class PerplexityState:
    def __init__(self, unk_index: int, is_including_unk: bool=True):
        self.unk_index = unk_index
        self.is_including_unk = is_including_unk
        self.word_count = 0
        self.zeroprobs_count = 0
        self.unknown_count = 0
        self.avg_log_perplexity = 0.
        self.time = 0.

    def add(self, word_index: int, probability: float) -> None:
        old_word_count = self.true_word_count
        self.word_count += 1

        if word_index == self.unk_index:
            self.unknown_count += 1
            if not self.is_including_unk:
                return

        if probability == 0.:
            self.zeroprobs_count += 1
            return

        log_prob = -np.log(probability)
        prev_avg = self.avg_log_perplexity * old_word_count / self.true_word_count
        self.avg_log_perplexity = prev_avg + log_prob / self.true_word_count

    @property
    def true_word_count(self):
        unknown_count = self.unknown_count if not self.is_including_unk else 0
        return self.word_count - self.zeroprobs_count - unknown_count

    @property
    def avg_perplexity(self):
        return np.exp(self.avg_log_perplexity)

    def __repr__(self):
        return "Avg ppl: {}, zeroprobs: {}, unk: {}, time: {}".format(
            self.avg_perplexity,
            self.zeroprobs_count,
            self.unknown_count,
            self.time
        )

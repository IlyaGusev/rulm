from typing import List, Dict, Tuple, Iterable
import os
from timeit import default_timer as timer
import logging

import numpy as np
import torch
from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.common.registrable import Registrable
from allennlp.common.params import Params
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField

from rulm.transform import Transform, TopKTransform
from rulm.beam import BeamSearch
from rulm.settings import DEFAULT_PARAMS, DEFAULT_VOCAB_DIR
from rulm.stream_reader import LanguageModelingStreamReader

logger = logging.getLogger(__name__)


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


class LanguageModel(Registrable):
    def __init__(self,
                 vocab: Vocabulary,
                 transforms: Tuple[Transform]=None,
                 reader: DatasetReader=None,
                 seed: int = 42):
        self.vocab = vocab  # type: Vocabulary
        self.transforms = transforms or tuple()  # type: Iterable[Transform]
        self.reader = reader or LanguageModelingStreamReader(reverse=False)
        self.set_seed(seed)

    def train(self,
              file_name: str,
              train_params: Params,
              serialization_dir: str=None,
              **kwargs):
        raise NotImplementedError()

    def predict(self, inputs: Iterable[int]) -> List[float]:
        raise NotImplementedError()

    @classmethod
    def _load(cls,
              params: Params,
              vocab: Vocabulary,
              serialization_dir: str,
              weights_file: str,
              cuda_device: int = -1) -> 'LanguageModel':
        raise NotImplementedError()

    @classmethod
    def load(cls,
             serialization_dir: str,
             params_file: str = None,
             weights_file: str = None,
             vocabulary_dir: str = None,
             cuda_device: int = -1) -> 'LanguageModel':
        params_file = params_file or os.path.join(serialization_dir, DEFAULT_PARAMS)
        params = Params.from_file(params_file)
        params.pop("vocab", None)

        vocabulary_dir = vocabulary_dir or os.path.join(serialization_dir, DEFAULT_VOCAB_DIR)
        vocabulary = Vocabulary.from_files(vocabulary_dir)

        model_type = params.pop("type")
        return cls.by_name(model_type)._load(params, vocabulary, serialization_dir,
                                             weights_file, cuda_device)

    def query(self, inputs: List[str]) -> Dict[str, float]:
        indices = self._numericalize_inputs(inputs)
        next_index_prediction = self.predict(indices)
        return {self.vocab.get_token_from_index(index): prob
                for index, prob in enumerate(next_index_prediction)}

    def beam_decoding(self, inputs: List[str], beam_width: int=5,
                      max_length: int=50, length_reward: float=0.0) -> List[str]:
        current_state = self._numericalize_inputs(inputs)
        beam = BeamSearch(
            eos_index=self.vocab.get_token_index(END_SYMBOL),
            predict_func=self.predict,
            transforms=self.transforms,
            beam_width=beam_width,
            max_length=max_length,
            length_reward=length_reward)
        best_guess = beam.decode(current_state)
        return self._decipher_outputs(best_guess)

    def sample_decoding(self, inputs: List[str], k: int=5, max_length: int=30) -> List[str]:
        vocab_size = self.vocab.get_vocab_size()
        if k > vocab_size:
            k = vocab_size
        current_state = self._numericalize_inputs(inputs)
        bos_index = self.vocab.get_token_index(START_SYMBOL)
        eos_index = self.vocab.get_token_index(END_SYMBOL)
        last_index = current_state[-1] if current_state else bos_index
        while last_index != eos_index and len(current_state) < max_length:
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

    def measure_perplexity(self, file_name: str, batch_size: int=100, is_including_unk: bool=True):
        unk_index = self.vocab.get_token_index(DEFAULT_OOV_TOKEN)
        ppl_state = PerplexityState(unk_index, is_including_unk)
        batch_number = 0
        batch = []
        for instance in self.reader.read(file_name):
            batch.append(instance)
            if len(batch) == batch_size:
                ppl_state = self._measure_perplexity_on_batch(batch, ppl_state)
                batch_number += 1
                logger.info("Measure_perplexity: {} sentences processed, {}".format(
                    batch_number * batch_size, ppl_state))
                batch = []
        if batch:
            ppl_state = self._measure_perplexity_on_batch(batch, ppl_state)
        return ppl_state

    def _measure_perplexity_on_batch(self, inputs: List[Instance], state: PerplexityState) -> PerplexityState:
        start_time = timer()
        for instance in inputs:
            instance.index_fields(self.vocab)
            text_field = instance["source_tokens"]
            sentence_indices = text_field.as_tensor(text_field.get_padding_lengths())["tokens"].tolist()
            for i in range(1, len(sentence_indices) + 1):
                indices = sentence_indices[:i]
                true_index = indices[-1]
                context = indices[:-1]
                prediction = self.predict(context)
                p = prediction[true_index]
                state.add(true_index, p)
        end_time = timer()
        state.time += end_time - start_time
        return state

    def _numericalize_inputs(self, words: List[str]) -> List[int]:
        words.insert(0, START_SYMBOL)
        return [self.vocab.get_token_index(word) for word in words]

    def _decipher_outputs(self, indices: List[int]) -> List[str]:
        return [self.vocab.get_token_from_index(index) for index in indices[1:-1]]

    @staticmethod
    def _choose(model: np.array, k: int=1):
        norm_model = model / np.sum(model)
        return np.random.choice(range(norm_model.shape[0]), k, p=norm_model, replace=False)

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.set_flags(True, False, True, True)



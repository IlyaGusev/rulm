from typing import List, Dict, Tuple, Iterable
import os
from timeit import default_timer as timer
import logging

import numpy as np
import torch
from torch import Tensor
from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from allennlp.common.util import END_SYMBOL
from allennlp.common.registrable import Registrable
from allennlp.common.params import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.basic_iterator import BasicIterator

from rulm.transform import Transform, TopKTransform, ExcludeTransform
from rulm.beam import BeamSearch
from rulm.settings import DEFAULT_PARAMS, DEFAULT_VOCAB_DIR
from rulm.stream_reader import LanguageModelingStreamReader
from rulm.perplexity_state import PerplexityState

logger = logging.getLogger(__name__)


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

    def predict(self, batch: Dict[str, Dict[str, Tensor]], **kwargs) -> np.ndarray:
        raise NotImplementedError()

    @classmethod
    def _load(cls,
              params: Params,
              vocab: Vocabulary,
              serialization_dir: str,
              weights_file: str,
              cuda_device: int = -1,
              **kwargs) -> 'LanguageModel':
        raise NotImplementedError()

    def predict_texts(self, texts: List[str], batch_size: int=64, **kwargs) -> np.ndarray:
        instances = [self.reader.text_to_instance(text) for text in texts]
        for instance in instances:
            instance.index_fields(self.vocab)
        iterator = BasicIterator(batch_size=batch_size)
        batches = iterator(instances, num_epochs=1)
        predictions = None
        for batch in batches:
            batch_predictions = self.predict(batch, **kwargs)
            predictions = batch_predictions if not predictions else np.concatenate((predictions, batch_predictions))
        return predictions

    def predict_text(self, text: str, **kwargs) -> np.ndarray:
        return self.predict_texts([text], **kwargs)[0]

    def query(self, text: str) -> Dict[str, float]:
        next_index_prediction = self.predict_text(text)
        for transform in self.transforms:
            next_index_prediction = transform(next_index_prediction)
        next_index_prediction = next_index_prediction / np.sum(next_index_prediction)
        return {self.vocab.get_token_from_index(index): float(prob)
                for index, prob in enumerate(next_index_prediction)}

    @classmethod
    def load(cls,
             serialization_dir: str,
             params_file: str = None,
             weights_file: str = None,
             vocabulary_dir: str = None,
             cuda_device: int = -1,
             **kwargs) -> 'LanguageModel':
        params_file = params_file or os.path.join(serialization_dir, DEFAULT_PARAMS)
        params = Params.from_file(params_file)
        params.pop("vocab", None)

        vocabulary_dir = vocabulary_dir or os.path.join(serialization_dir, DEFAULT_VOCAB_DIR)
        vocabulary = Vocabulary.from_files(vocabulary_dir)

        model_type = params.pop("type")
        return cls.by_name(model_type)._load(params, vocabulary, serialization_dir,
                                             weights_file, cuda_device, **kwargs)

    def sample_decoding(self,
                        input_text: str,
                        k: int=5,
                        max_length: int=30,
                        exclude_unk: bool=False,
                        **kwargs) -> str:
        vocab_size = self.vocab.get_vocab_size()
        if k > vocab_size:
            k = vocab_size
        eos_index = self.vocab.get_token_index(END_SYMBOL)
        indices = self.text_to_indices(input_text)[:-1]
        last_index = indices[-1]
        current_text = input_text

        while last_index != eos_index and len(indices) < max_length:
            next_word_probabilities = self.predict_text(current_text, **kwargs)
            if exclude_unk:
                unk_index = self.vocab.get_token_index(DEFAULT_OOV_TOKEN)
                next_word_probabilities = ExcludeTransform((unk_index, ))(next_word_probabilities)
            for transform in self.transforms:
                next_word_probabilities = transform(next_word_probabilities)
            next_word_probabilities = TopKTransform(k)(next_word_probabilities)
            last_index = self._choose(next_word_probabilities)[0]
            for transform in self.transforms:
                transform.advance(last_index)
            if last_index != eos_index:
                current_text = current_text + " " + self.vocab.get_token_from_index(last_index)
                indices = self.text_to_indices(current_text)[:-1]
        return current_text.strip()

    def beam_decoding(self,
                      input_text: str,
                      beam_width: int=5,
                      max_length: int=50,
                      length_reward: float=0.0,
                      **kwargs) -> str:
        beam = BeamSearch(
            eos_index=self.vocab.get_token_index(END_SYMBOL),
            predict_func=lambda x : self.predict_text(x, **kwargs),
            index_to_text_func=self.vocab.get_token_from_index,
            transforms=self.transforms,
            beam_width=beam_width,
            max_length=max_length,
            length_reward=length_reward)
        best_guess = beam.decode(input_text)
        return best_guess

    def measure_perplexity(self,
                           file_name: str,
                           batch_size: int=50,
                           is_including_unk: bool=True):
        unk_index = self.vocab.get_token_index(DEFAULT_OOV_TOKEN)
        ppl_state = PerplexityState(unk_index, is_including_unk)
        batch_number = 0

        iterator = BasicIterator(batch_size=batch_size)
        iterator.index_with(self.vocab)
        dataset = self.reader.read(file_name)
        batches = iterator(dataset, num_epochs=1)
        for batch in batches:
            ppl_state = self._measure_perplexity_on_batch(batch, ppl_state)
            batch_number += 1
            logger.info("Measure_perplexity: {} sentences processed, {}".format(
                batch_number * batch_size, ppl_state))
        return ppl_state

    def _measure_perplexity_on_batch(self,
                                     batch: Dict[str, Tensor],
                                     state: PerplexityState) -> PerplexityState:
        start_time = timer()

        all_field = batch["all_tokens"]
        tokens = all_field["tokens"]
        batch_size = tokens.size(0)
        sentence_len = tokens.size(1)
        for i in range(1, sentence_len):
            field = {}
            for key, value in all_field.items():
                field[key] = value[:, :i]
            inputs = {"source_tokens": field}
            true_indices = tokens[:, i]
            predictions = self.predict(inputs)
            for b in range(batch_size):
                true_index = true_indices[b]
                if true_index == self.vocab.get_token_index(DEFAULT_PADDING_TOKEN):
                    continue
                prediction = predictions[b]
                state.add(true_index, prediction[true_index])
        end_time = timer()
        state.time += end_time - start_time
        return state

    def text_to_indices(self, input_text: str) -> List[int]:
        instance = self.reader.text_to_instance(input_text)
        instance.index_fields(self.vocab)
        text_field = instance["all_tokens"]
        tensor = text_field.as_tensor(text_field.get_padding_lengths())
        indices = tensor["tokens"].tolist()
        return indices

    def _decipher_outputs(self, indices: List[int]) -> List[str]:
        return [self.vocab.get_token_from_index(index) for index in indices[1:-1]]

    @staticmethod
    def _choose(model: np.array, k: int=1):
        norm_model = model / np.sum(model)
        non_zero_count = int(np.sum(model > 0))
        k = min(k, non_zero_count)
        return np.random.choice(range(norm_model.shape[0]), k, p=norm_model, replace=False)

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.set_flags(True, False, True, True)



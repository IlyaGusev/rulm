import unittest
import os
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from rulm.settings import REMEMBERING_EXAMPLE, ENCODER_ONLY_MODEL_PARAMS, DEFAULT_VOCAB_DIR
from rulm.language_model import LanguageModel
from rulm.models.neural_net import NeuralNetLanguageModel


class TestRNNLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configs = (ENCODER_ONLY_MODEL_PARAMS,)
        cls.params_sets = [Params.from_file(config) for config in configs]

        cls.vocabularies = []
        for params in cls.params_sets:
            vocabulary_params = params.pop("vocabulary", default=Params({}))
            reader_params = params.duplicate().pop("reader", default=Params({}))
            cls.reader = DatasetReader.from_params(reader_params)
            dataset = cls.reader.read(REMEMBERING_EXAMPLE)
            cls.vocabularies.append(Vocabulary.from_params(vocabulary_params, instances=dataset))

        cls.sentences = []
        with open(REMEMBERING_EXAMPLE, "r", encoding="utf-8") as r:
            for line in r:
                cls.sentences.append(line.strip())

    def _test_model_predictions(self, model, reverse=False):
        for sentence in self.sentences:
            sentence = sentence.split()
            if reverse:
                sentence = sentence[::-1]
            for i in range(1, len(sentence)-1):
                if i == 1 and (sentence[0] in "Я!.?"):
                    continue
                context = sentence[:i]
                prediction = model.sample_decoding(context, k=1)
                self.assertListEqual(prediction, sentence)

    @staticmethod
    def _test_model_equality(model1: NeuralNetLanguageModel, model2: NeuralNetLanguageModel):
        old_params = list([param.detach().cpu().numpy() for param in model1.model.parameters()])
        new_params = list([param.detach().cpu().numpy() for param in model2.model.parameters()])
        for o, n in zip(old_params, new_params):
            np.testing.assert_array_almost_equal(o, n)

    def test_model_from_file(self):
        for params, vocabulary in zip(self.params_sets, self.vocabularies):
            params = params.duplicate()
            train_params = params.pop("train")
            model = LanguageModel.from_params(params, vocab=vocabulary)
            model.train(REMEMBERING_EXAMPLE, train_params)
            self._test_model_predictions(model)

    def test_reversed_model(self):
        for params, vocabulary in zip(self.params_sets, self.vocabularies):
            params = params.duplicate()
            train_params = params.pop('train')

            params["reader"]["reverse"] = True
            model_reversed = LanguageModel.from_params(params, vocab=vocabulary)

            model_reversed.train(REMEMBERING_EXAMPLE, train_params)
            self._test_model_predictions(model_reversed, reverse=True)

    def test_save_load(self):
        with TemporaryDirectory() as dirpath:
            for params, vocabulary in zip(self.params_sets, self.vocabularies):
                vocab_dir = os.path.join(dirpath, DEFAULT_VOCAB_DIR)
                os.mkdir(vocab_dir)
                vocabulary.save_to_files(vocab_dir)

                params = params.duplicate()
                train_params = params.pop('train')
                model = LanguageModel.from_params(params, vocab=vocabulary)
                model.train(REMEMBERING_EXAMPLE, train_params, dirpath)

                loaded_model = LanguageModel.load(dirpath,
                                                  params_file=ENCODER_ONLY_MODEL_PARAMS,
                                                  vocabulary_dir=vocab_dir)

                self.assertTrue(isinstance(model, NeuralNetLanguageModel))
                self.assertTrue(isinstance(loaded_model, NeuralNetLanguageModel))
                self._test_model_equality(cast(NeuralNetLanguageModel, model),
                                          cast(NeuralNetLanguageModel, loaded_model))

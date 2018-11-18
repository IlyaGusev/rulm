import unittest
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary

from rulm.nn.language_model import NNLanguageModel
from rulm.settings import RNNLM_REMEMBER_EXAMPLE, RNNLM_MODEL_PARAMS
from rulm.stream_reader import LanguageModelingStreamReader

class TestRNNLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = LanguageModelingStreamReader()
        dataset = cls.reader.read(RNNLM_REMEMBER_EXAMPLE)
        cls.vocabulary = Vocabulary.from_instances(dataset)

        cls.sentences = []
        with open(RNNLM_REMEMBER_EXAMPLE, "r", encoding="utf-8") as r:
            for line in r:
                cls.sentences.append(line.strip())

        cls.params = Params.from_file(RNNLM_MODEL_PARAMS)
        params = cls.params.duplicate()
        train_params = params.pop("train")
        cls.model = NNLanguageModel.from_params(params, vocab=cls.vocabulary)
        cls.model.train_file(RNNLM_REMEMBER_EXAMPLE, train_params)

    def _test_model_predictions(self, model, reverse=False):
        for sentence in self.sentences:
            sentence = sentence.split()
            if reverse:
                sentence = sentence[::-1]
            for i in range(1, len(sentence)-1):
                if i == 1 and sentence[0] == "Я" or reverse and sentence[0] != "!":
                    continue
                context = sentence[:i]
                if reverse:
                    context = context[::-1]
                prediction = model.sample_decoding(context, k=1)
                self.assertListEqual(prediction, sentence)

    @staticmethod
    def _test_model_equality(model1, model2):
        old_params = list([param.detach().cpu().numpy() for param in model1.model.parameters()])
        new_params = list([param.detach().cpu().numpy() for param in model2.model.parameters()])
        for o, n in zip(old_params, new_params):
            np.testing.assert_array_almost_equal(o, n)

    def test_model_from_file(self):
        self._test_model_predictions(self.model)

    #def test_train_from_python(self):
    #    params = self.params.duplicate()
    #    model = NNLanguageModel(self.vocabulary, params.pop("model"))
    #    model.train(self.sentences, params.pop("train"))
    #    self._test_model_predictions(model)
    #    self._test_model_equality(model, self.model)

    def test_reversed_model(self):
        params = self.params.duplicate()
        train_params = params.pop('train')
        model_reversed = NNLanguageModel.from_params(params, vocab=self.vocabulary, reverse=True)
        model_reversed.train_file(RNNLM_REMEMBER_EXAMPLE, train_params)
        self._test_model_predictions(model_reversed, reverse=True)

    def test_save_load(self):
        with TemporaryDirectory() as dirpath:
            params = self.params.duplicate()
            train_params = params.pop('train')
            model = NNLanguageModel.from_params(params, vocab=self.vocabulary)
            model.train_file(RNNLM_REMEMBER_EXAMPLE, train_params, dirpath)
            loaded_model = NNLanguageModel.load(dirpath, RNNLM_MODEL_PARAMS)
            self._test_model_equality(model, loaded_model)


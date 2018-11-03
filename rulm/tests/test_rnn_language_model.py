import unittest
import os
from tempfile import NamedTemporaryFile

import numpy as np

from rulm.vocabulary import Vocabulary
from rulm.nn.language_model import TrainConfig
from rulm.nn.rnn_language_model import RNNLanguageModel
from rulm.settings import RNNLM_REMEMBER_EXAMPLE


class TestRNNLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocabulary = Vocabulary()
        cls.vocabulary.add_file(RNNLM_REMEMBER_EXAMPLE)

        cls.sentences = []
        with open(RNNLM_REMEMBER_EXAMPLE, "r", encoding="utf-8") as r:
            for line in r:
                cls.sentences.append(line.strip())

        cls.config = TrainConfig()
        cls.config.epochs = 20

        cls.model = RNNLanguageModel(cls.vocabulary)
        cls.model.train_file(RNNLM_REMEMBER_EXAMPLE, cls.config)

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

    def _test_model_equality(self, model1, model2):
        old_params = list([param.detach().cpu().numpy() for param in model1.model.parameters()])
        new_params = list([param.detach().cpu().numpy() for param in model2.model.parameters()])
        for o, n in zip(old_params, new_params):
            np.testing.assert_array_almost_equal(o, n)

    def test_model_from_file(self):
        self._test_model_predictions(self.model)

    def test_train_from_python(self):
        model = RNNLanguageModel(self.vocabulary)
        model.train(self.sentences, self.config)
        self._test_model_predictions(model)
        self._test_model_equality(model, self.model)

    def test_reversed_model(self):
        model_reversed = RNNLanguageModel(self.vocabulary, reverse=True)
        model_reversed.train_file(RNNLM_REMEMBER_EXAMPLE, self.config)
        self._test_model_predictions(model_reversed, reverse=True)

    def test_save_load(self):
        f = NamedTemporaryFile(delete=False)
        self.model.save(f.name)
        loaded_model = RNNLanguageModel(self.vocabulary)
        loaded_model.load(f.name)
        os.unlink(f.name)
        self.assertFalse(os.path.exists(f.name))
        self._test_model_equality(loaded_model, self.model)


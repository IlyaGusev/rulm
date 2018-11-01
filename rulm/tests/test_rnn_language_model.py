import unittest
import os
from tempfile import NamedTemporaryFile

from rulm.vocabulary import Vocabulary
from rulm.nn.language_model import TrainConfig
from rulm.nn.rnn_language_model import RNNLanguageModel
from rulm.settings import RNNLM_REMEMBER_EXAMPLE


class TestRNNLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocabulary = Vocabulary()
        cls.vocabulary.add_file(RNNLM_REMEMBER_EXAMPLE)
        cls.config = TrainConfig()
        cls.config.epochs = 20

        cls.model1 = RNNLanguageModel(cls.vocabulary)
        cls.model1.train_file(RNNLM_REMEMBER_EXAMPLE, cls.config)

        cls.model_reversed = RNNLanguageModel(cls.vocabulary, reverse=True)
        cls.model_reversed.train_file(RNNLM_REMEMBER_EXAMPLE, cls.config)

        cls.model2 = RNNLanguageModel(cls.vocabulary)
        with open(RNNLM_REMEMBER_EXAMPLE, encoding="utf-8") as r:
            lines = list(map(lambda x: x.strip(), r.readlines()))
            cls.model2.train(lines, cls.config)

    def test_print(self):
        sentences = []
        for model in (self.model1, self.model2):
            with open(RNNLM_REMEMBER_EXAMPLE, "r", encoding="utf-8") as r:
                for line in r:
                    sentences.append(line.strip().split())
            for sentence in sentences:
                for i in range(1, len(sentence)-1):
                    if i == 1 and sentence[0] == "Я":
                        continue
                    context = sentence[:i]
                    self.assertListEqual(model.sample_decoding(context, k=1), sentence)
        for sentence in sentences:
            if sentence[-1] != '!':
                continue
            for i in range(len(sentence)-1, -1, -1):
                context = sentence[i:]
                self.assertListEqual(self.model_reversed.sample_decoding(context, k=1), sentence[::-1])


    def test_save_load(self):
        f = NamedTemporaryFile(delete=False)
        self.model1.save(f.name)
        loaded_model = RNNLanguageModel(self.vocabulary)
        loaded_model.load(f.name)
        os.unlink(f.name)
        self.assertFalse(os.path.exists(f.name))


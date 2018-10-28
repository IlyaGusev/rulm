import unittest
import os
from tempfile import NamedTemporaryFile

from rulm.vocabulary import Vocabulary
from rulm.nn.rnn_language_model import RNNLanguageModel
from rulm.settings import RNNLM_REMEMBER_EXAMPLE


class TestRNNLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocabulary = Vocabulary()
        cls.vocabulary.add_file(RNNLM_REMEMBER_EXAMPLE)
        cls.model = RNNLanguageModel(cls.vocabulary)
        cls.model.train_file(RNNLM_REMEMBER_EXAMPLE, epochs=20)

    def test_print(self):
        sentences = []
        with open(RNNLM_REMEMBER_EXAMPLE, "r", encoding="utf-8") as r:
            for line in r:
                sentences.append(line.strip().split())
        for sentence in sentences:
            for i in range(1, len(sentence)-1):
                if i == 1 and sentence[0] == "Я":
                    continue
                context = sentence[:i]
                self.assertListEqual(self.model.sample_decoding(context, k=1), sentence)

    def test_save_load(self):
        f = NamedTemporaryFile(delete=False)
        self.model.save(f.name)
        loaded_model = RNNLanguageModel(self.vocabulary)
        loaded_model.load(f.name)
        os.unlink(f.name)
        self.assertFalse(os.path.exists(f.name))


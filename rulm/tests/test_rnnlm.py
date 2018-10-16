import unittest

from rulm.vocabulary import Vocabulary
from rulm.rnnlm import RNNLanguageModel
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

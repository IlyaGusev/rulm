import unittest

import numpy as np

from rulm.vocabulary import Vocabulary
from rulm.language_model import EquiprobableLanguageModel

class TestLanguageModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocabulary = Vocabulary()
        cls.vocabulary.add_word("я")
        cls.vocabulary.add_word("не")
        cls.vocabulary.add_word("ты")
        cls.eq_model = EquiprobableLanguageModel(cls.vocabulary)

    def test_measure_perplexity(self):
        self.assertAlmostEqual(self.eq_model.measure_perplexity([["я", "не", "ты"]]), 5.)

    def test_query(self):
        predictions = self.eq_model.query([])
        self.assertEqual(predictions, {
            "я": 1./5.,
            "не": 1./5.,
            "ты": 1./5.,
            Vocabulary.UNK: 1./5.,
            Vocabulary.EOS: 1./5.,
            Vocabulary.PAD: 0.,
            Vocabulary.BOS: 0.
        })

    def test_sample_decoding(self):
        np.random.seed(1045966)
        self.assertListEqual(self.eq_model.sample_decoding([], k=5), ["не", "я", "ты", "я"])


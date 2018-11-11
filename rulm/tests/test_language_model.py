import unittest

import numpy as np

from rulm.vocabulary import Vocabulary
from rulm.language_model import EquiprobableLanguageModel, VocabularyChainLanguageModel, PerplexityState

# TODO: test reverse


class TestLanguageModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocabulary = Vocabulary()
        cls.vocabulary.add_word("я")
        cls.vocabulary.add_word("не")
        cls.vocabulary.add_word("ты")
        cls.eq_model = EquiprobableLanguageModel(cls.vocabulary)
        cls.chain_model = VocabularyChainLanguageModel(cls.vocabulary)

    def test_measure_perplexity(self):
        eq_state = PerplexityState()
        eq_state = self.eq_model.measure_perplexity([["я", "не", "ты"]], eq_state)
        self.assertAlmostEqual(np.exp(eq_state.avg_log_perplexity), 5.)
        self.assertEqual(eq_state.zeroprobs_count, 0)

        chain_state = PerplexityState()
        chain_state = self.chain_model.measure_perplexity([["я", "не", "ты"]], chain_state)
        self.assertEqual(chain_state.zeroprobs_count, 1)
        self.assertAlmostEqual(np.exp(chain_state.avg_log_perplexity), 1.)

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

    def test_beam_decoding(self):
        self.assertListEqual(self.eq_model.beam_decoding(["не"], beam_width=10), ["не"])

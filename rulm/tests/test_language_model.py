import unittest

import numpy as np
from allennlp.data.vocabulary import Vocabulary, DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from rulm.language_model import EquiprobableLanguageModel, VocabularyChainLanguageModel, PerplexityState

# TODO: test reverse


class TestLanguageModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocabulary = Vocabulary()
        cls.vocabulary.add_token_to_namespace(START_SYMBOL)
        cls.vocabulary.add_token_to_namespace(END_SYMBOL)
        cls.vocabulary.add_token_to_namespace("я")
        cls.vocabulary.add_token_to_namespace("не")
        cls.vocabulary.add_token_to_namespace("ты")
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
            DEFAULT_OOV_TOKEN: 1./5.,
            END_SYMBOL: 1./5.,
            DEFAULT_PADDING_TOKEN: 0.,
            START_SYMBOL: 0.
        })

    def test_sample_decoding(self):
        np.random.seed(13370)
        self.assertListEqual(self.eq_model.sample_decoding([], k=5), ['не', 'не'])

    def test_beam_decoding(self):
        self.assertListEqual(self.eq_model.beam_decoding(["не"], beam_width=10), ["не"])

import unittest
import os
from tempfile import NamedTemporaryFile

import numpy as np
from allennlp.data.vocabulary import Vocabulary, DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from rulm.models.vocabulary_chain import VocabularyChainLanguageModel
from rulm.models.equiprobable import EquiprobableLanguageModel


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
        val_file = NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
        val_file.write("я не ты")
        val_file.close()

        eq_state = self.eq_model.measure_perplexity(val_file.name)
        self.assertAlmostEqual(np.exp(eq_state.avg_log_perplexity), 5.)
        self.assertEqual(eq_state.zeroprobs_count, 0)

        chain_state = self.chain_model.measure_perplexity(val_file.name)
        self.assertEqual(chain_state.zeroprobs_count, 0)
        self.assertAlmostEqual(np.exp(chain_state.avg_log_perplexity), 1.)

        os.unlink(val_file.name)

    def test_query(self):
        predictions = self.eq_model.query("")
        self.assertDictEqual(predictions, {
            "я": 1./5.,
            "не": 1./5.,
            "ты": 1./5.,
            DEFAULT_OOV_TOKEN: 1./5.,
            END_SYMBOL: 1./5.,
            DEFAULT_PADDING_TOKEN: 0.,
            START_SYMBOL: 0.
        })

    def test_sample_decoding(self):
        self.eq_model.set_seed(13370)
        self.assertEqual(self.eq_model.sample_decoding("", k=5), 'не не')

    def test_beam_decoding(self):
        self.assertEqual(self.eq_model.beam_decoding("не", beam_width=10), "не")
        self.assertEqual(self.chain_model.beam_decoding("я", beam_width=10), "я не ты")

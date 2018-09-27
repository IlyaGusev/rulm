import unittest
import tempfile
import os
import numpy as np

from rulm.vocabulary import Vocabulary
from rulm.ngrams import NGramLanguageModel
from rulm.settings import TRAIN_EXAMPLE, TRAIN_VOCAB_EXAMPLE, DATA_DIR

class TestNGrams(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocabulary = Vocabulary()
        cls.vocabulary.add_word("я")
        cls.vocabulary.add_word("не")
        cls.vocabulary.add_word("ты")
        cls.model = NGramLanguageModel(n=3, vocabulary=cls.vocabulary)
        cls.model.train([["я", "не", "я"], ["ты", "не", "ты"], ["я", "не", "ты"]])

    def test_save_load(self):
        vocabulary = Vocabulary()
        vocabulary.load(TRAIN_VOCAB_EXAMPLE)
        model1 = NGramLanguageModel(n=3, vocabulary=vocabulary, interpolation_lambdas=(1.0, 0.0, 0.0))
        model1.train_file(TRAIN_VOCAB_EXAMPLE)

        def assert_ngrams_equal(n_grams_1, n_grams_2):
            for words, p1 in n_grams_1.items():
                self.assertIn(words, n_grams_2)
                p2 = n_grams_2[words]
                self.assertEqual("{:.4f}".format(np.log10(p1)), "{:.4f}".format(np.log10(p2)))

        def assert_models_equal(m1, m2):
            self.assertEqual(m1.n, m2.n)
            for n in range(1, m1.n+1):
                assert_ngrams_equal(m1.n_grams[n], m2.n_grams[n])

        model_path_text = os.path.join(DATA_DIR, "model.arpa")
        model1.save(model_path_text)
        model2 = NGramLanguageModel(n=3, vocabulary=vocabulary, interpolation_lambdas=(1.0, 0.0, 0.0))
        model2.load(model_path_text)
        assert_models_equal(model1, model2)
        os.remove(model_path_text)

        model_path_gzip = os.path.join(DATA_DIR, "model.arpa.gzip")
        model1.save(model_path_gzip)
        model3 = NGramLanguageModel(n=3, vocabulary=vocabulary, interpolation_lambdas=(1.0, 0.0, 0.0))
        model3.load(model_path_gzip)
        assert_models_equal(model1, model3)
        os.remove(model_path_gzip)

    def test_predict_big(self):
        vocabulary = Vocabulary()
        vocabulary.load(TRAIN_VOCAB_EXAMPLE)
        model = NGramLanguageModel(n=2, vocabulary=vocabulary)
        model.train_file(TRAIN_EXAMPLE)
        prediction = model.predict([vocabulary.get_bos()])
        non_zero_indices = list(filter(lambda x: x != 0., prediction))
        self.assertEqual(len(non_zero_indices), 19457)

    def test_train(self):
        def assert_n_gram_prob(n_gram, p):
            self.assertEqual(self.model.n_grams[len(n_gram)][n_gram], p)
        assert_n_gram_prob(tuple(), 1.)
        assert_n_gram_prob((4, ), 3./15.)
        assert_n_gram_prob((6, ), 3./15.)
        assert_n_gram_prob((5, ), 3./15.)
        assert_n_gram_prob((self.vocabulary.get_eos(), ), 3./15.)
        assert_n_gram_prob((self.vocabulary.get_bos(), ), 3./15.)
        assert_n_gram_prob((4, 5), 2./3.)
        assert_n_gram_prob((self.vocabulary.get_bos(), self.vocabulary.get_eos()), 0.0)
        assert_n_gram_prob((4, 5, 4), 0.5)
        assert_n_gram_prob((4, 5, 5), 0.0)

    def test_predict(self):
        def pack_context(context):
            return tuple(map(self.vocabulary.get_index_by_word, context))

        def assert_prediction(context, prediction):
            self.assertListEqual(list(self.model.predict(pack_context(context))), prediction)

        assert_prediction((Vocabulary.BOS, "я"), [0., 0., 0., 0., 0., 1., 0.])
        assert_prediction(("я", "не"), [0., 0., 0., 0., 0.5, 0., 0.5])
        assert_prediction(("не", "ты"), [0., 0., 1., 0., 0., 0., 0.])

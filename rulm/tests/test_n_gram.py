import unittest
import os
import time
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import cast

import numpy as np
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.common.params import Params

from rulm.models.n_gram import NGramLanguageModel
from rulm.settings import TRAIN_EXAMPLE, TRAIN_VOCAB_EXAMPLE, TEST_EXAMPLE, N_GRAM_PARAMS, DEFAULT_VOCAB_DIR
from rulm.stream_reader import LanguageModelingStreamReader
from rulm.language_model import LanguageModel


class TestNGrams(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocabulary = Vocabulary()
        cls.vocabulary.add_token_to_namespace(START_SYMBOL)
        cls.vocabulary.add_token_to_namespace(END_SYMBOL)
        cls.vocabulary.add_token_to_namespace("я")
        cls.vocabulary.add_token_to_namespace("не")
        cls.vocabulary.add_token_to_namespace("ты")

        cls.reader = LanguageModelingStreamReader()

        cls.model = NGramLanguageModel(n=3, vocab=cls.vocabulary)
        cls.model.train([["я", "не", "я"], ["ты", "не", "ты"], ["я", "не", "ты"]])
        cls.model.normalize()

    def _assert_ngrams_equal(self, n_grams_1, n_grams_2):
        for words, p1 in n_grams_1.items():
            p2 = n_grams_2[words]
            self.assertEqual("{:.4f}".format(np.log10(p1)), "{:.4f}".format(np.log10(p2)))

    def _assert_models_equal(self, m1, m2):
        self.assertEqual(m1.n, m2.n)
        for n in range(1, m1.n + 1):
            self._assert_ngrams_equal(m1.n_grams[n], m2.n_grams[n])

    def test_save_load_weights(self):
        vocabulary = Vocabulary.from_files(TRAIN_VOCAB_EXAMPLE)
        model1 = NGramLanguageModel(n=3, vocab=vocabulary, interpolation_lambdas=(1.0, 0.0, 0.0))
        model1.train_file(TRAIN_EXAMPLE)

        model1_file = NamedTemporaryFile(delete=False, suffix=".arpa")
        model1.save_weights(model1_file.name)
        model2 = NGramLanguageModel(n=3, vocab=vocabulary, interpolation_lambdas=(1.0, 0.0, 0.0))
        model2.load_weights(model1_file.name)
        self._assert_models_equal(model1, model2)
        os.unlink(model1_file.name)

        model1_file_gzip = NamedTemporaryFile(delete=False, suffix=".arpa.gzip")
        model1.save_weights(model1_file_gzip.name)
        model3 = NGramLanguageModel(n=3, vocab=vocabulary, interpolation_lambdas=(1.0, 0.0, 0.0))
        model3.load_weights(model1_file_gzip.name)
        self._assert_models_equal(model1, model3)
        os.unlink(model1_file_gzip.name)

    def test_predict_big(self):
        vocabulary = Vocabulary.from_files(TRAIN_VOCAB_EXAMPLE)
        model = NGramLanguageModel(n=2, vocab=vocabulary)
        model.train_file(TRAIN_EXAMPLE)

        prediction = model.predict([vocabulary.get_token_index(START_SYMBOL)])
        non_zero_indices = list(filter(lambda x: x != 0., prediction))
        self.assertEqual(len(non_zero_indices), 502)

    def test_predict_time(self):
        dataset = self.reader.read(TRAIN_EXAMPLE)
        vocabulary = Vocabulary.from_instances(dataset, max_vocab_size=3000)
        model = NGramLanguageModel(n=3, vocab=vocabulary, interpolation_lambdas=(1.0, 0.1, 0.01))
        model.train_file(TRAIN_EXAMPLE)

        ts = time.time()
        for i in range(10):
            model.predict([7, 28])
        te = time.time()
        self.assertLess(te - ts, 1.0)

    def test_train(self):
        def assert_n_gram_prob(n_gram, p):
            self.assertEqual(self.model.n_grams[len(n_gram)][n_gram], p)
        assert_n_gram_prob(tuple(), 1.)
        assert_n_gram_prob((4, ), 3./15.)
        assert_n_gram_prob((6, ), 3./15.)
        assert_n_gram_prob((5, ), 3./15.)
        assert_n_gram_prob((self.vocabulary.get_token_index(END_SYMBOL), ), 3./15.)
        assert_n_gram_prob((self.vocabulary.get_token_index(START_SYMBOL), ), 3./15.)
        assert_n_gram_prob((4, 5), 2./3.)
        assert_n_gram_prob((self.vocabulary.get_token_index(START_SYMBOL),
                            self.vocabulary.get_token_index(END_SYMBOL)), 0.0)
        assert_n_gram_prob((4, 5, 4), 0.5)
        assert_n_gram_prob((4, 5, 5), 0.0)

    def test_predict(self):
        def pack_context(context):
            return tuple(map(self.vocabulary.get_token_index, context))

        def assert_prediction(context, prediction):
            self.assertListEqual(list(self.model.predict(pack_context(context))), prediction)

        assert_prediction((START_SYMBOL, "я"), [0., 0., 0., 0., 0., 1., 0.])
        assert_prediction(("я", "не"), [0., 0., 0., 0., 0.5, 0., 0.5])
        assert_prediction(("не", "ты"), [0., 0., 0., 1., 0., 0., 0.])

    def test_perplexity(self):
        dataset = self.reader.read(TRAIN_EXAMPLE)
        vocabulary = Vocabulary.from_instances(dataset, max_vocab_size=500)
        model = NGramLanguageModel(n=3, vocab=vocabulary, interpolation_lambdas=(1.0, 0.1, 0.01))
        model.train_file(TRAIN_EXAMPLE)
        ppl_state = model.measure_perplexity_file(TEST_EXAMPLE)
        self.assertLess(np.exp(ppl_state.avg_log_perplexity), 30.)

    def test_from_params(self):
        params = Params.from_file(N_GRAM_PARAMS)
        vocabulary_params = params.pop("vocab")
        dataset = self.reader.read(TRAIN_EXAMPLE)
        vocabulary = Vocabulary.from_params(vocabulary_params, instances=dataset)
        model = LanguageModel.from_params(params, vocab=vocabulary)
        self.assertTrue(isinstance(model, NGramLanguageModel))

    def test_save_load(self):
        with TemporaryDirectory() as dirpath:
            params = Params.from_file(N_GRAM_PARAMS)
            vocabulary_params = params.pop("vocab")
            dataset = self.reader.read(TRAIN_EXAMPLE)
            vocabulary = Vocabulary.from_params(vocabulary_params, instances=dataset)

            vocab_dir = os.path.join(dirpath, DEFAULT_VOCAB_DIR)
            os.mkdir(vocab_dir)
            vocabulary.save_to_files(vocab_dir)

            model = LanguageModel.from_params(params, vocab=vocabulary)
            model.train_file(TRAIN_EXAMPLE, Params({}), serialization_dir=dirpath)

            loaded_model = LanguageModel.load(dirpath,
                                              params_file=N_GRAM_PARAMS,
                                              vocabulary_dir=vocab_dir)

            self.assertTrue(isinstance(model, NGramLanguageModel))
            self.assertTrue(isinstance(loaded_model, NGramLanguageModel))
            self._assert_models_equal(cast(NGramLanguageModel, model),
                                      cast(NGramLanguageModel, loaded_model))

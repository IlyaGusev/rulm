import unittest

from rulm.vocabulary import Vocabulary
from rulm.ngrams import NGramLanguageModel
from rulm.settings import TRAIN_EXAMPLE, TRAIN_VOCAB_EXAMPLE

class TestNGrams(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocabulary = Vocabulary()
        cls.vocabulary.add_word("я")
        cls.vocabulary.add_word("не")
        cls.vocabulary.add_word("ты")
        cls.model = NGramLanguageModel(n=3, vocabulary=cls.vocabulary)
        cls.model.train([["я", "не", "я"], ["ты", "не", "ты"]])

    def test_canon(self):
        vocabulary = Vocabulary()
        vocabulary.load(TRAIN_VOCAB_EXAMPLE)
        model = NGramLanguageModel(n=3, vocabulary=vocabulary)
        model.train_file(TRAIN_VOCAB_EXAMPLE)

    def test_predict_big(self):
        vocabulary = Vocabulary()
        vocabulary.load(TRAIN_VOCAB_EXAMPLE)
        model = NGramLanguageModel(n=2, vocabulary=vocabulary)
        model.train_file(TRAIN_EXAMPLE)
        prediction = model.predict([vocabulary.get_bos()])
        non_zero_indices = list(filter(lambda x: x != 0., prediction))
        self.assertEqual(len(non_zero_indices), 19457)

    def test_train(self):
        self.assertEqual(self.model.n_grams[0][tuple()], 10.0)
        self.assertEqual(self.model.n_grams[1][(4, )], 0.2)
        self.assertEqual(self.model.n_grams[1][(6, )], 0.2)
        self.assertEqual(self.model.n_grams[1][(5, )], 0.2)
        self.assertEqual(self.model.n_grams[1][(self.vocabulary.get_eos(), )], 0.2)
        self.assertEqual(self.model.n_grams[1][(self.vocabulary.get_bos(), )], 0.2)
        self.assertEqual(self.model.n_grams[2][(4, 5)], 0.5)
        self.assertEqual(self.model.n_grams[2][(self.vocabulary.get_bos(), self.vocabulary.get_eos())], 0.0)
        self.assertEqual(self.model.n_grams[3][(4, 5, 4)], 1.0)
        self.assertEqual(self.model.n_grams[3][(4, 5, 5)], 0.0)


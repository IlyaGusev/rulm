import unittest

from rulm.vocabulary import Vocabulary
from rulm.ngrams import NGramLanguageModel
from rulm.settings import TRAIN_EXAMPLE, TRAIN_VOCAB_EXAMPLE

class TestNGrams(unittest.TestCase):
    def test_canon(self):
        vocabulary = Vocabulary()
        vocabulary.load(TRAIN_VOCAB_EXAMPLE)
        model = NGramLanguageModel(n=3, vocabulary=vocabulary)
        model.train_file(TRAIN_VOCAB_EXAMPLE)
        assert len(model.transitions) == 490624

    def test_predict(self):
        vocabulary = Vocabulary()
        vocabulary.load(TRAIN_VOCAB_EXAMPLE)
        model = NGramLanguageModel(n=3, vocabulary=vocabulary)
        model.train_file(TRAIN_EXAMPLE)
        prediction = model.predict([vocabulary.get_bos()])
        non_zero_indices = list(filter(lambda x: x != 0., prediction))
        self.assertEqual(len(non_zero_indices), 19457)

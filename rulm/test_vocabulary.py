import unittest

from rulm.vocabulary import Vocabulary
from rulm.settings import TRAIN_EXAMPLE, TRAIN_VOCAB_EXAMPLE

class TestsVocabulary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocabulary = Vocabulary()
        cls.vocabulary.load(TRAIN_VOCAB_EXAMPLE)
        assert len(cls.vocabulary) > 4
        assert cls.vocabulary.get_word_by_index(cls.vocabulary.get_bos()) == "<bos>"

    @classmethod
    def tearDownClass(cls):
        del cls.vocabulary

    def test_canon(self):
        vocabulary = Vocabulary()
        vocabulary.add_file(TRAIN_EXAMPLE)
        vocabulary.sort()
        self.assertGreater(len(vocabulary), 4)
        self.assertEqual(vocabulary.get_word_by_index(vocabulary.get_bos()), "<bos>")
        self.assertEqual(vocabulary.__dict__, self.vocabulary.__dict__)

    def test_add_word(self):
        vocabulary = Vocabulary()
        vocabulary.add_word("a")
        vocabulary.add_word("b")
        vocabulary.add_word("a")
        self.assertTrue(vocabulary.has_word("a"))
        self.assertTrue(vocabulary.has_word("b"))
        self.assertEqual(vocabulary.get_count_by_word("a"), 2)
        self.assertEqual(vocabulary.get_count_by_word("b"), 1)
        self.assertEqual(vocabulary.get_index_by_word("a"), 4)
        self.assertEqual(vocabulary.get_index_by_word("b"), 5)

    def test_sort(self):
        vocabulary = Vocabulary()
        vocabulary.add_file(TRAIN_EXAMPLE)
        size = len(vocabulary)
        vocabulary.sort()
        self.assertEqual(len(vocabulary), size)
        for index, word in enumerate(vocabulary.index_to_word[:-1]):
            if vocabulary.is_special(index):
                self.assertEqual(vocabulary.index_to_count[index], 0)
                continue
            next_count = vocabulary.index_to_count[index + 1]
            current_count = vocabulary.index_to_count[index]
            self.assertGreaterEqual(current_count, next_count)


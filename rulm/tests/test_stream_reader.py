import unittest

from allennlp.data.vocabulary import Vocabulary

from rulm.settings import TRAIN_EXAMPLE
from rulm.stream_reader import LanguageModelingStreamReader


class TestStreamReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = LanguageModelingStreamReader()

    def test_one_file(self):
        dataset = list(self.reader.read(TRAIN_EXAMPLE))
        self.assertNotEqual(len(dataset), 0)
        for _ in range(2):
            prev_sample = None
            for sample in dataset:
                self.assertIsNotNone(sample)
                self.assertNotEqual(len(sample), 0)
                if prev_sample:
                    self.assertNotEqual(sample["source_tokens"], prev_sample["source_tokens"])
                prev_sample = sample

    def test_vocabulary(self):
        dataset = self.reader.read(TRAIN_EXAMPLE)
        vocabulary = Vocabulary.from_instances(dataset)
        self.assertNotEqual(vocabulary.get_vocab_size(), 0)

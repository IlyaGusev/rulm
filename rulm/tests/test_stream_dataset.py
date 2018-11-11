import unittest

from rulm.settings import TRAIN_EXAMPLE
from rulm.datasets.stream_dataset import StreamFilesDataset


class TestStreamDataset(unittest.TestCase):
    def test_one_file(self):
        ds = StreamFilesDataset([TRAIN_EXAMPLE], lambda x: x.strip().split())
        self.assertNotEqual(len(ds), 0)
        for _ in range(2):
            prev_sample = None
            for i in range(len(ds)):
                sample = ds[i]
                self.assertIsNotNone(sample)
                self.assertNotEqual(len(sample), 0)
                if prev_sample:
                    self.assertNotEqual(sample, prev_sample)
                prev_sample = sample


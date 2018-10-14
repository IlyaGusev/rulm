import os
from pkg_resources import resource_filename

DATA_DIR = resource_filename(__name__, "data/")
TRAIN_EXAMPLE = os.path.join(DATA_DIR, "rdt.example.txt")
TRAIN_VOCAB_EXAMPLE = os.path.join(DATA_DIR, "vocab.rdt.example.txt")
TEST_EXAMPLE = os.path.join(DATA_DIR, "rdt.example.test.txt")
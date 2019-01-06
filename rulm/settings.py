import os
from pkg_resources import resource_filename

DATA_DIR = resource_filename(__name__, "data")
CONFIGS_DIR = resource_filename(__name__, "configs")
TRAIN_EXAMPLE = os.path.join(DATA_DIR, "rdt.example.txt")
TRAIN_VOCAB_EXAMPLE = os.path.join(DATA_DIR, "rdt.example.vocab")
TEST_EXAMPLE = os.path.join(DATA_DIR, "rdt.example.test.txt")
REMEMBERING_EXAMPLE = os.path.join(DATA_DIR, "remember.txt")
ENCODER_ONLY_MODEL_PARAMS = os.path.join(CONFIGS_DIR, "encoder_only.json")
N_GRAM_PARAMS = os.path.join(CONFIGS_DIR, "n_gram.json")

DEFAULT_PARAMS = "config.json"
DEFAULT_VOCAB_DIR = "vocabulary"
DEFAULT_N_GRAM_WEIGHTS = "weights.arpa"

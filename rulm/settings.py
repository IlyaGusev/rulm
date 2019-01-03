import os
from pkg_resources import resource_filename

DATA_DIR = resource_filename(__name__, "data")
CONFIGS_DIR = resource_filename(__name__, "configs")
TRAIN_EXAMPLE = os.path.join(DATA_DIR, "rdt.example.txt")
TRAIN_VOCAB_EXAMPLE = os.path.join(DATA_DIR, "rdt.example.vocab")
TEST_EXAMPLE = os.path.join(DATA_DIR, "rdt.example.test.txt")
RNNLM_REMEMBER_EXAMPLE = os.path.join(DATA_DIR, "rnnlm.remember.txt")
RNNLM_MODEL_PARAMS = os.path.join(CONFIGS_DIR, "encoder_only.json")

from rulm.vocabulary import Vocabulary
from rulm.settings import TRAIN_EXAMPLE, TRAIN_VOCAB_EXAMPLE, TEST_EXAMPLE, RNNLM_REMEMBER_EXAMPLE
from rulm.nn.rnn_language_model import RNNLanguageModel

vocabulary = Vocabulary()
vocabulary.load(TRAIN_VOCAB_EXAMPLE)
vocabulary.sort(1000)
#vocabulary.add_file(RNNLM_REMEMBER_EXAMPLE)
model = RNNLanguageModel(vocabulary)
#model.train_file(RNNLM_REMEMBER_EXAMPLE, epochs=20)
model.train_file(TRAIN_EXAMPLE, epochs=10)

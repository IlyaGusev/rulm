from rulm.vocabulary import Vocabulary
from rulm.settings import TRAIN_EXAMPLE, TRAIN_VOCAB_EXAMPLE, TEST_EXAMPLE
from rulm.rnnlm import RNNLanguageModel

vocabulary = Vocabulary()
vocabulary.load(TRAIN_VOCAB_EXAMPLE)
vocabulary.sort(10000)
model = RNNLanguageModel(vocabulary)
# model.train_file(TRAIN_EXAMPLE)
# model.save("model.pt")
model.load("model.pt")
print(model.sample_decoding(["После"]))
# model.measure_perplexity_file(TEST_EXAMPLE)
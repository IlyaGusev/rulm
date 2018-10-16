from rulm.vocabulary import Vocabulary
from rulm.settings import TRAIN_EXAMPLE, TRAIN_VOCAB_EXAMPLE, TEST_EXAMPLE
from rulm.rnnlm import RNNLanguageModel

TRAIN = "/media/yallen/My Passport/Datasets/rdt_very_clean/rdt/rdt.train.01.txt"
vocabulary = Vocabulary()
vocabulary.add_file(TRAIN)
vocabulary.save("vocab.01.txt")
vocabulary.sort(50000)
model = RNNLanguageModel(vocabulary)
model.train_file(TRAIN, epochs=2)
print(model.sample_decoding(["После"]))
# model.measure_perplexity_file(TEST_EXAMPLE)
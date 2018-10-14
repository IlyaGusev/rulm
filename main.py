from rulm.vocabulary import Vocabulary
from rulm.ngrams import DictNGramContainer, TrieNGramContainer, NGramLanguageModel
from rulm.transform import AlphabetOrderTransform
from nltk import WordPunctTokenizer

# file_name = "../../rdt/rdt.train.01.txt"
# vocab_file_name = "vocab.rdt.train.01.txt"
# #vocab_file_name = "sphinx.vocab.txt"
# #model_file_name = "ru.lm"
# #test_file_name = "rulm/data/rdt.example.test.txt"
#
# vocabulary = Vocabulary()
# vocabulary.load(vocab_file_name)
# print("Vocabulary ok")
#
# model = NGramLanguageModel(3, vocabulary, tuple(), (1.0, 0.1, 0.01))
# model.train_file(file_name)
# model.save("model.train.01.arpa")
#
# #model = NGramLanguageModel(3, vocabulary, tuple(), (1.0, 0.1, 0.01), container=DictNGramContainer)
# #model.load("ru.lm")
#
# #print(" ".join(model.sample_decoding([], k=10)))
# #s = input()
# #print(model.measure_perplexity_file(test_file_name))
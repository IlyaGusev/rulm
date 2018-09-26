from rulm.vocabulary import Vocabulary
from rulm.ngrams import NGramLanguageModel
from rulm.transform import AlphabetOrderTransform
from nltk import WordPunctTokenizer

file_name = "rulm/data/rdt.example.txt"
vocab_file_name = "rulm/data/vocab.rdt.example.txt"
test_file_name = "rulm/data/rdt.example.test.txt"

vocabulary = Vocabulary()
vocabulary.load(vocab_file_name)
vocabulary.sort(100000)

f = AlphabetOrderTransform(vocabulary)
model = NGramLanguageModel(4, vocabulary, tuple(), (1.0, 0.1, 0.01, 0.000001))
model.train_file(file_name)
model.save("model.arpa")

print(" ".join(model.sample_decoding([], k=5)))
#print(model.measure_perplexity_file(test_file_name))

from rulm.vocabulary import Vocabulary
from rulm.ngrams import NGramLanguageModel
from rulm.transform import AlphabetOrderTransform
from nltk import WordPunctTokenizer

file_name = "rulm/data/rdt.example.txt"
vocab_file_name = "rulm/data/vocab.rdt.example.txt"

vocabulary = Vocabulary()
vocabulary.load(vocab_file_name)

f = AlphabetOrderTransform(vocabulary)
model = NGramLanguageModel(3, vocabulary, (f, ), (1.0, 0.01, 0.0001))
model.train_file(file_name)

print(model.beam_decoding([], beam_width=20))
#model.measure_perplexity(sentences)

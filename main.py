from rulm.vocabulary import Vocabulary
from rulm.ngrams import NGramLanguageModel
from rulm.filter import AlphabetOrderFilter
from nltk import WordPunctTokenizer

file_name = "rdt.example.txt"
vocabulary = Vocabulary()
#i = 0
#with open(file_name, "r") as r:
#    for line in r:
#        i += 1
#        for word in line.strip().split():
#            vocabulary.add_word(word)
#        if i % 100000 == 0:
#            print("Vocab:", i)
#vocabulary.sort()
vocabulary.load("vocab_rdt_example.txt")

sentences = []
with open(file_name, "r") as r:
    for line in r:
        words = line.strip().split()
        sentences.append(words)
f = AlphabetOrderFilter(vocabulary)
model = NGramLanguageModel(4, vocabulary, f)
model.train(sentences)
print(model.sample_decoding(["В"], k=5))
#model.measure_perplexity(sentences)

from rulm.vocabulary import Vocabulary
from rulm.ngrams import NGramLanguageModel
from nltk import WordPunctTokenizer

vocabulary = Vocabulary()
with open("wp.txt", "r") as r:
    for line in r:
        for word in WordPunctTokenizer().tokenize(line):
            vocabulary.add_word(word)
vocabulary.sort()
vocabulary.save("vocab_wp.txt")

model = NGramLanguageModel(3, vocabulary)
with open("wp.txt", "r") as r:
    for line in r:
        words = WordPunctTokenizer().tokenize(line)
        model.train([words])
print(model.query(["<bos>", "В"]))
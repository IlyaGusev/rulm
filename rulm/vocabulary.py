import json
from typing import List, Dict, Counter


class Vocabulary:
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self):
        self.specials = (Vocabulary.PAD, Vocabulary.BOS, Vocabulary.EOS, Vocabulary.UNK)
        self.index_to_word = list()  # type: List[str]
        self.word_to_index = dict()  # type: Dict[str, int]
        self.index_to_count = Counter()   # type: Dict[int, int]
        self.size = 0  # type: int
        self._reset()

    def _reset(self):
        self.index_to_word = list(self.specials)  # type: List[str]
        self.word_to_index = {word: i for i, word in enumerate(self.specials)}  # type: Dict[str, int]
        self.index_to_count = Counter()  # type: Dict[int, int]
        self.size = len(self.specials)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as w:
            for index, word in enumerate(self.index_to_word):
                if index < len(self.specials):
                    continue
                count = self.index_to_count[index]
                w.write(word + "\t" + str(count) + "\n")

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as r:
            self._reset()
            for line in r:
                word, count = line.strip().split("\t")
                self._insert_word_with_count(word, int(count))

    def get_unk(self) -> int:
        return self.specials.index(Vocabulary.UNK)

    def get_pad(self) -> int:
        return self.specials.index(Vocabulary.PAD)

    def get_bos(self) -> int:
        return self.specials.index(Vocabulary.BOS)

    def get_eos(self) -> int:
        return self.specials.index(Vocabulary.EOS)

    def is_special(self, index: int) -> bool:
        return index < len(self.specials)

    def get_index_by_word(self, word: str) -> int:
        word = self.word_to_index.get(word, None)
        return word if word else self.get_unk()

    def get_word_by_index(self, index: int) -> str:
        return self.index_to_word[index]

    def get_count_by_word(self, word: str) -> int:
        return self.index_to_count[self.get_index_by_word(word)]

    def __len__(self) -> int:
        return self.size

    def _insert_word_with_count(self, word: str, count: int):
        self.index_to_word.append(word)
        self.word_to_index[word] = self.size
        self.index_to_count[self.size] = count
        self.size += 1

    def add_word(self, word: str) -> bool:
        index = self.word_to_index.get(word, None)
        if index:
            self.index_to_count[index] += 1
            return False
        self._insert_word_with_count(word, 1)
        return True

    def has_word(self, word: str) -> bool:
        return word in self.word_to_index

    def sort(self, n_best: int=None):
        n_best = n_best if n_best else self.size
        word_to_count = {self.index_to_word[index]: count for index, count in self.index_to_count.items()
                         if index >= len(self.specials)}
        self._reset()
        for i, (word, count) in enumerate(sorted(word_to_count.items(), key=lambda x: -x[1])):
            self._insert_word_with_count(word, count)
            if i >= n_best:
                break

    def add_file(self, file_name):
        with open(file_name, "r", encoding="utf-8") as r:
            for line in r:
                for word in line.strip().split():
                    self.add_word(word)


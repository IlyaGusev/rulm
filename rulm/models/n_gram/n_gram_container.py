from typing import Iterable, List
from collections import defaultdict

import pygtrie
from allennlp.common.registrable import Registrable


class NGramContainer(Registrable):
    def __getitem__(self, n_gram: Iterable[int]):
        raise NotImplementedError()

    def __setitem__(self, n_gram: Iterable[int], value: float):
        raise NotImplementedError()

    def __delitem__(self, n_gram: Iterable[int]):
        raise NotImplementedError()

    def __contains__(self, n_gram: Iterable[int]):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()

    def items(self):
        raise NotImplementedError()


@NGramContainer.register("dict")
class DictNGramContainer(NGramContainer):
    def __init__(self):
        self.data = defaultdict(float)

    def __getitem__(self, n_gram: Iterable[int]):
        return self.data.get(tuple(n_gram), 0.)

    def __setitem__(self, n_gram: Iterable[int], value: float):
        self.data[tuple(n_gram)] = value

    def __delitem__(self, n_gram: Iterable[int]):
        del self.data[tuple(n_gram)]

    def __contains__(self, n_gram: Iterable[int]):
        return tuple(n_gram) in self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.data)

    def items(self):
        return self.data.items()


@NGramContainer.register("trie")
class TrieNGramContainer(NGramContainer):
    def __init__(self):
        self.data = pygtrie.Trie()

    def __getitem__(self, n_gram: List[int]):
        return self.data[tuple(n_gram)] if n_gram in self.data else 0.

    def __setitem__(self, n_gram: List[int], value: float):
        self.data[tuple(n_gram)] = value

    def __delitem__(self, n_gram: Iterable[int]):
        del self.data[tuple(n_gram)]

    def __contains__(self, n_gram: Iterable[int]):
        return tuple(n_gram) in self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.data)

    def items(self):
        return self.data.items()
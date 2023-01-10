from itertools import tee
from typing import List, Text
import unicodedata
import json


def gen_batch(records, batch_size):
    batch_start = 0
    while batch_start < len(records):
        batch_end = batch_start + batch_size
        batch = records[batch_start: batch_end]
        batch_start = batch_end
        yield batch


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ")
    text = text.replace("&quot;", '"')
    return text


def remove_non_printable(text):
    return "".join(c for c in text if c.isprintable())


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


class PlainArchive:
    def __init__(self, file_path):
        self.file_path = file_path
        self.fh = open(file_path, "w")

    def add_data(self, text, meta={}):
        self.fh.write(json.dumps({"text": text, "meta": meta}, ensure_ascii=False).strip() + "\n")

    def commit(self):
        self.fh.flush()


def ngrams(sequence: List[Text], n: int):
    """
    Return the ngrams generated from a sequence of items, as an iterator.
    This is a modified version of nltk.util.ngrams.
    """
    iterables = tee(iter(sequence), n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)



class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        self.parent[px] = self.parent[py] = min(px, py)

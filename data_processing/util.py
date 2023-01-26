from itertools import tee
from typing import List, Text
import unicodedata
import json
import random
import zstandard
import io
import jsonlines
import simdjson

from data_processing.lang_detector import FasttextLanguageDetector

parser = simdjson.Parser()

def parse_json(x):
    try:
        return parser.parse(x).as_dict()
    except ValueError:
        return


BAD_SUBSTRINGS = (
    "+79",
    "@gmail",
    "var ",
    "<a ",
    "<p ",
    ".jpg",
    "http:",
    "https:"
)

lang_detector = FasttextLanguageDetector()

def gen_batch(records, batch_size):
    batch_start = 0
    while batch_start < len(records):
        batch_end = batch_start + batch_size
        batch = records[batch_start: batch_end]
        batch_start = batch_end
        yield batch


def gen_batch_iter(records, batch_size):
    batch = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class TextProcessor:
    def __init__(
        self,
        languages=("ru", ),
        join_lines=False,
        normalization="NFKC",
        min_chars=30
    ):
        self.languages = languages
        self.join_lines = join_lines
        self.normalization = normalization
        self.min_chars = min_chars

    def remove_non_printable(self, text):
        return "".join(c for c in text if c.isprintable())

    def fix_punct(self, line):
        line = " ".join(line.split()).strip()
        line = line.strip("*").strip("=").strip("~").strip("•")
        line = line.replace(" ,", ",")
        line = line.replace(" .", ". ")
        line = line.replace(",", ", ")
        line = " ".join(line.split()).strip()
        line = line.replace(". ,", ".,")
        return line

    def normalize(self, text):
        text = unicodedata.normalize(self.normalization, text)
        text = text.replace("\xa0", " ")
        text = text.replace("&quot;", '"')
        text = text.replace("&gt;", ">")
        text = text.replace("&lt;", "<")
        text = text.replace("&ge;", ">=")
        text = text.replace("&le;", "<=")
        text = text.replace("&amp;", "&")
        text = text.replace("&nbsp;", " ")
        lines = [self.remove_non_printable(line) for line in text.split("\n")]
        lines = [self.fix_punct(line) for line in lines]
        lines = [" ".join(line.split()).strip() for line in lines]
        lines = [l for l in lines if len(set(l.replace(" ", "").strip())) > 1]
        if self.join_lines:
            text = " ".join(lines)
        else:
            text = "\n".join(lines)
        return text

    def has_bad_ss(self, text):
        return any(ss in text for ss in BAD_SUBSTRINGS)

    def has_bad_language(self, text):
        return lang_detector(text)[0] not in self.languages

    def __call__(self, text):
        text = self.normalize(text)
        if len(text) < self.min_chars:
            return None
        if self.has_bad_ss(text):
            return None
        if self.has_bad_language(text):
            return None
        return text


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield parse_json(line)


class PlainArchive:
    def __init__(self, file_path, mode="w"):
        self.file_path = file_path
        self.fh = open(file_path, mode)
        self.mode = mode

    def __iter__(self):
        assert self.mode == "r"
        for line in self.fh:
            yield parse_json(line)

    def add_data(self, text, meta={}):
        assert self.mode == "w"
        self.fh.write(json.dumps({"text": text, "meta": meta}, ensure_ascii=False).strip() + "\n")

    def commit(self):
        assert self.mode == "w"
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

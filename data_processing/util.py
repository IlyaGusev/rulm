import string
from itertools import tee
from typing import List, Text
import unicodedata
import json
import random
import io
import re

import zstandard
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
    "https:",
    "www."
)

STOP_BEFORE_LETTER = re.compile(r'\.(\w)')

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
        min_chars=30,
        min_text_part=0.9,
        fix_punct=True,
        fix_spaces=True,
        fix_short_lines=True,
        check_languages=True,
        check_bad_ss=True
    ):
        self.languages = languages
        self.join_lines = join_lines
        self.normalization = normalization
        self.min_chars = min_chars
        self.min_text_part = min_text_part
        self.fix_punct = fix_punct
        self.fix_spaces = fix_spaces
        self.fix_short_lines = fix_short_lines
        self.check_languages = check_languages
        self.check_bad_ss = check_bad_ss

    def remove_non_printable(self, text):
        return "".join(c for c in text if c.isprintable())

    def fix_line_punct(self, line):
        line = " ".join(line.split()).strip()
        line = line.strip("*").strip("=").strip("~").strip("•")
        line = line.replace(" ,", ",")
        line = line.replace(" .", ". ")
        line = STOP_BEFORE_LETTER.sub(r'. \1', line)
        line = line.replace(" ?", "?")
        line = line.replace(" !", "!")
        line = line.replace(" %", "%")
        line = line.replace(" ;", ";")
        line = line.replace(" :", ":")
        line = line.replace(":", ": ")
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

        lines = text.split("\n")
        lines = [self.remove_non_printable(line) for line in lines]

        if self.fix_punct:
            lines = [self.fix_line_punct(line) for line in lines]
        if self.fix_spaces:
            lines = [" ".join(line.split()).strip() for line in lines]
        if self.fix_short_lines:
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

    def count_text_part(self, sentence):
        text_count = 0.0
        all_count = 0.0
        for ch in sentence:
            all_count += 1.0
            if ch in string.punctuation:
                continue
            if ch.isnumeric():
                continue
            if ch in string.ascii_letters:
                continue
            text_count += 1.0
        return text_count / all_count

    def __call__(self, text):
        text = self.normalize(text)
        if len(text) < self.min_chars:
            return None
        if self.check_bad_ss and self.has_bad_ss(text):
            return None
        if self.check_languages and self.has_bad_language(text):
            return None
        if self.count_text_part(text) < self.min_text_part:
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

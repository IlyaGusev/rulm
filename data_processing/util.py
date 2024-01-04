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

SIMPLE_EMAIL_RE = re.compile(r"\S+@\S+")

PII_SUBSTRINGS = (
    "+79",
    "+74",
    "+7 (",
    "+7(",
    "@mail",
    "@gmail",
    "@yandex"
)

CODE_SUBSTRINGS = (
    "var ",
    "<a",
    "<p",
    "<h",
    "<th",
    "<tr",
    "<div",
)

LINKS_SUBSTRINGS = (
    ".jpg",
    "http:",
    "https:",
    "www.",
    ".com",
    ".ru",
    ".mp3"
)


STOP_BEFORE_LETTER = re.compile(r'\.(\w)')
RE_SQUARE_BRACKETS = re.compile(r"\[[^\]]*\]", flags=re.MULTILINE)

UNICODE_PUNCTUATION = {
    "，": ",",
    "。": ".",
    "、": ",",
    "„": '"',
    "”": '"',
    "“": '"',
    "«": '"',
    "»": '"',
    "１": '"',
    "」": '"',
    "「": '"',
    "《": '"',
    "》": '"',
    "´": "'",
    "∶": ":",
    "：": ":",
    "？": "?",
    "！": "!",
    "（": "(",
    "）": ")",
    "；": ";",
    "–": "-",
    "—": " - ",
    "．": ". ",
    "～": "~",
    "’": "'",
    "…": "...",
    "━": "-",
    "〈": "<",
    "〉": ">",
    "【": "[",
    "】": "]",
    "％": "%",
    "►": "-",
}

HTML_ARTEFACTS = {
    "\xa0": " ",
    "&quot;": '"',
    "&gt;": ">",
    "&lt;": "<",
    "&ge;": ">=",
    "&le;": "<=",
    "&amp;": "&",
    "&apos;": "'",
    "&nbsp;": " ",
    "&approx;": "≈",
    "&lbrace;": "{",
    "&rbrace;": "}",
    "&lbrack;": "[",
    "&rbrack;": "]"
}

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
        join_lines: bool = False,
        normalization: str = "NFKC",
        min_chars: int = 30,
        min_text_part: float = 0.85,
        fix_punct: bool = True,
        fix_spaces: bool = True,
        fix_short_lines: bool = True,
        check_languages: bool = True,
        check_pii: bool = True,
        check_code: bool = True,
        check_links: bool = True,
        check_email: bool = True,
        check_text_part: bool = True
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
        self.check_pii = check_pii
        self.check_code = check_code
        self.check_links = check_links
        self.check_email = check_email
        self.check_text_part = check_text_part

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
        line = " ".join(line.split()).strip()
        line = line.replace(". ,", ".,")
        return line

    def normalize(self, text):
        text = unicodedata.normalize(self.normalization, text)

        for old, new in HTML_ARTEFACTS.items():
            text = text.replace(old, new)
        for old, new in UNICODE_PUNCTUATION.items():
            text = text.replace(old, new)

        lines = text.split("\n")
        lines = [self.remove_non_printable(line) for line in lines]
        lines = [line for line in lines if line.strip()]

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
        has_email = self.check_email and SIMPLE_EMAIL_RE.search(text)
        has_pii = self.check_pii and any(ss in text for ss in PII_SUBSTRINGS)
        has_code = self.check_code and any(ss in text.lower() for ss in CODE_SUBSTRINGS)
        has_links = self.check_links and any(ss in text.lower() for ss in LINKS_SUBSTRINGS)
        return has_pii or has_code or has_links or has_email

    def has_bad_language(self, text):
        return lang_detector(text)[0] not in self.languages

    def count_text_part(self, sentence):
        all_count = len(sentence)
        text_count = sum(1 for ch in sentence if '\u0400' <= ch <= '\u04FF' or ch.isspace())
        return text_count / all_count

    def remove_square_brackets(self, text):
        brackets = RE_SQUARE_BRACKETS.finditer(text)
        for bracket in brackets:
            bracket = bracket.group()
            text = text.replace(bracket, " ")
        return text

    def __call__(self, text):
        text = self.normalize(text)
        if len(text) < self.min_chars:
            return None
        if self.has_bad_ss(text):
            return None
        if self.check_languages and self.has_bad_language(text):
            return None
        if self.check_text_part and self.count_text_part(text) < self.min_text_part:
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

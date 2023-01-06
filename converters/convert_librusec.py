import sys
import re
import json
import unicodedata

import razdel
from tqdm import tqdm

from converters.lang_detector import FasttextLanguageDetector

RE_ID = re.compile(r'^(\d+)\.fb2')
RE_BRACKETS = re.compile(r"\([^\)]*\)", flags=re.MULTILINE)
RE_SQUARE_BRACKETS = re.compile(r"\[[^\]]*\]", flags=re.MULTILINE)


def gen_batch(records, batch_size):
    batch_start = 0
    while batch_start < len(records):
        batch_end = batch_start + batch_size
        batch = records[batch_start: batch_end]
        batch_start = batch_end
        yield batch


def preprocess_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ")
    text = "".join(c for c in text if c.isprintable())

    brackets = RE_BRACKETS.finditer(text)
    for bracket in brackets:
        bracket = bracket.group()
        if len(bracket) > 20:
            continue
        text = text.replace(bracket, " ")

    brackets = RE_SQUARE_BRACKETS.finditer(text)
    for bracket in brackets:
        bracket = bracket.group()
        if len(bracket) > 20:
            continue
        text = text.replace(bracket, " ")

    if text.count("//") > 20:
        return
    sentences = [s.text for s in razdel.sentenize(text)]
    for s in sentences:
        if len(s) > 1500:
            return

    text = " ".join(text.split())
    return text.strip()

lang_detector = FasttextLanguageDetector()
input_path = sys.argv[1]
output_path = sys.argv[2]

with open(input_path, "r") as r, open(output_path, "w") as w:
    def flush(text_id, fragments):
        text = " ".join(fragments)
        sentences = [s.text for s in razdel.sentenize(text)]
        if text.count("...") > 100:
            return
        if text.count("!!!") > 100:
            return
        for fragment_num, fragment_sentences in enumerate(gen_batch(sentences, 500)):
            fragment = " ".join(fragment_sentences)
            if lang_detector(fragment)[0] != "ru":
                continue
            fragment = preprocess_text(fragment)
            if not fragment:
                continue
            record = {
                "text_id": text_id,
                "fragment_num": fragment_num,
                "text": fragment
            }
            w.write(json.dumps(record, ensure_ascii=False) + "\n")

    text_id = None
    fragments = []
    for line in tqdm(r):
        match = RE_ID.match(line)
        if match:
            if text_id:
                flush(text_id, fragments)
                fragments = []
            text_id = match.group(1)
            line = line[match.end() + 1:]
        fragments.append(line)
    flush(text_id, fragments)

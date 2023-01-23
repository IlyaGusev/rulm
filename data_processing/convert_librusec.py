import sys
import re
import json
import unicodedata

import razdel
from tqdm import tqdm

from data_processing.util import gen_batch, PlainArchive, TextProcessor


RE_ID = re.compile(r'^(\d+)\.fb2')
RE_BRACKETS = re.compile(r"\([^\)]*\)", flags=re.MULTILINE)
RE_SQUARE_BRACKETS = re.compile(r"\[[^\]]*\]", flags=re.MULTILINE)
TEXT_PROCESSOR = TextProcessor(join_lines=True)

def preprocess_text(text):
    text = TEXT_PROCESSOR(text)
    if text is None:
        return

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

    return " ".join(text.split()).strip()

input_path = sys.argv[1]
output_path = sys.argv[2]

archive = PlainArchive(output_path)

with open(input_path, "r") as r:
    def flush(text_id, fragments):
        text = " ".join(fragments)
        if text.count("...") > 100:
            return
        if text.count("!!!") > 100:
            return

        sentences = [s.text for s in razdel.sentenize(text)]
        for fragment_num, fragment_sentences in enumerate(gen_batch(sentences, 500)):
            fragment = " ".join(fragment_sentences)
            fragment = preprocess_text(fragment)
            if not fragment:
                continue
            if len(fragment) < 300:
                continue
            archive.add_data(
                text=fragment,
                meta={
                    "source": "librusec",
                    "text_id": text_id,
                    "fragment_num": fragment_num,
                }
            )

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

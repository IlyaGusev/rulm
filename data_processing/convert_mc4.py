import argparse
import string
from collections import Counter
from urllib.parse import urlparse

import razdel
from datasets import load_dataset
from tqdm import tqdm

from data_processing.util import TextProcessor, PlainArchive

PLAUSIBLE_ENDINGS = ".!?"
SKIP_SUBSTRINGS = (
    "загрузка",
    "посетители",
    "регистрация",
    "порно",
    "хуй",
    "пизда",
    "ебать"
)

def main(
    output_path
):
    output_archive = PlainArchive(output_path)
    text_processor = TextProcessor(min_chars=200)
    dataset = load_dataset("mc4", languages=["ru"], split="train", streaming=True)

    for record in tqdm(dataset):
        text = text_processor(record["text"])
        if not text:
            continue
        url = record["url"]
        host = urlparse(record["url"]).netloc
        fixed_paragraphs = []
        paragraphs = text.split("\n")
        for paragraph in paragraphs:
            fixed_sentences = []
            for sentence in razdel.sentenize(paragraph):
                sentence = sentence.text
                if text_processor.has_bad_language(sentence):
                    continue
                if text_processor.count_text_part(sentence) < 0.75:
                    continue
                sentence = sentence.strip()
                if not sentence:
                    continue
                words = sentence.split()
                if len(words) < 3:
                    continue
                if any(len(word) > 50 for word in words):
                    continue
                if sentence[-1] not in PLAUSIBLE_ENDINGS:
                    continue
                if sentence[0] in string.punctuation:
                    continue
                fixed_sentences.append(sentence)
            paragraph = " ".join(fixed_sentences).strip()
            if not paragraph:
                continue
            fixed_paragraphs.append(paragraph)

        if len(fixed_paragraphs) >= 2 and fixed_paragraphs[0] == fixed_paragraphs[1]:
            fixed_paragraphs = fixed_paragraphs[1:]

        text = "\n".join(fixed_paragraphs)

        if not text.strip() or len(text) < 200:
            continue
        if any(ss in text for ss in SKIP_SUBSTRINGS):
            continue
        if len(text) > 200000:
            continue

        output_archive.add_data(
            text=text,
            meta={
                "source": "mc4",
                "url": url,
                "host": host
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    main(**vars(args))

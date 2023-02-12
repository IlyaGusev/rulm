import argparse
import string
from collections import Counter
from urllib.parse import urlparse

import razdel
from datasets import load_dataset
from tqdm import tqdm

from data_processing.util import TextProcessor, PlainArchive

PLAUSIBLE_ENDINGS = ".!?"

with open("resources/mc4_bad_hosts.txt") as r:
    BAD_HOSTS = {line.strip() for line in r}

with open("resources/mc4_news_hosts.txt") as r:
    NEWS_HOSTS = {line.strip() for line in r}

with open("resources/mc4_good_hosts.txt") as r:
    GOOD_HOSTS = {line.strip() for line in r}

with open("resources/mc4_ru_bad_words.txt") as r:
    SKIP_WORDS = {line.strip() for line in r}

HOST_CNT = Counter()
HOST_EXAMPLE = dict()

def clean_text(text, text_processor):
    text = text_processor(text)
    if not text:
        return

    paragraphs = text.split("\n")
    fixed_paragraphs = []
    set_paragraphs = set()
    for p in paragraphs:
        p = p.strip()
        if p not in set_paragraphs:
            set_paragraphs.add(p)
            fixed_paragraphs.append(p)
    paragraphs = fixed_paragraphs

    fixed_paragraphs = []
    for paragraph in paragraphs:
        sentences = [s.text for s in razdel.sentenize(paragraph)]
        if any(len(s.strip()) >= 1000 for s in sentences):
            return
        if len(paragraph) >= 70 and text_processor.count_text_part(paragraph) < 0.8:
            return
        fixed_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if text_processor.has_bad_language(sentence):
                continue
            if text_processor.count_text_part(sentence) < 0.75:
                continue

            words = sentence.split()
            if len(words) <= 2:
                continue
            if any(len(word) >= 50 for word in words):
                return
            if any(len(word) >= 30 and "-" not in word and "," not in word for word in words):
                return
            if sentence[-1] not in PLAUSIBLE_ENDINGS:
                continue
            if sentence[0] in string.punctuation:
                continue

            tokens = [t.text for t in razdel.tokenize(sentence)]
            if any(token in SKIP_WORDS for token in tokens):
                return
            if any(word in SKIP_WORDS for word in words):
                return
            fixed_sentences.append(sentence)
        paragraph = " ".join(fixed_sentences).strip()
        if not paragraph:
            continue
        fixed_paragraphs.append(paragraph)
    text = "\n".join(fixed_paragraphs)
    if not text.strip() or len(text) < 200:
        return
    if len(text) > 200000:
        return
    return text


def main(
    output_path,
    news_output_path
):
    output_archive = PlainArchive(output_path)
    news_output_archive = PlainArchive(news_output_path)
    text_processor = TextProcessor(min_chars=200, min_text_part=0.85)
    dataset = load_dataset("mc4", languages=["ru"], split="train", streaming=True)
    skipped_count = 0
    for i, record in enumerate(tqdm(dataset)):
        url = record["url"]
        host = urlparse(record["url"]).netloc
        if host in BAD_HOSTS or "map" in host or ".ua" in host:
            skipped_count += 1
            continue

        is_news = False
        if host in NEWS_HOSTS or "news" in host or "smi" in host or "press" in host:
            is_news = True

        text = clean_text(record["text"], text_processor)
        if not text:
            skipped_count += 1
            continue

        if host not in GOOD_HOSTS and not is_news:
            HOST_CNT[host] += 1
            HOST_EXAMPLE[host] = " ".join(text.split()[:10]) + "..."

        if i % 10000 == 0 and i != 0:
            for pos, (host, cnt) in enumerate(HOST_CNT.most_common(40)):
                print(pos + 1, host, "\t", HOST_EXAMPLE[host])
            print("Skipped count:", skipped_count)
            print()

        archive = news_output_archive if is_news else output_archive
        archive.add_data(
            text=text,
            meta={
                "source": "mc4_news" if is_news else "mc4",
                "url": url,
                "host": host
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=str)
    parser.add_argument("news_output_path", type=str)
    args = parser.parse_args()
    main(**vars(args))

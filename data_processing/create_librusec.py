import argparse
import re
import json
import unicodedata

import razdel
from tqdm import tqdm

from data_processing.util import TextProcessor

RE_ID = re.compile(r'^(\d+)\.fb2')
TEXT_PROCESSOR = TextProcessor(
    min_chars=500,
    min_text_part=0.0,
    fix_punct=False,
    fix_spaces=False,
    fix_short_lines=False,
    check_code=True,
    check_pii=True,
    check_links=True,
    check_languages=True,
    check_email=False,
    check_text_part=False
)


def preprocess_text(text, text_id):
    text = TEXT_PROCESSOR(text)
    if text is None:
        print("Skipping id {}, pii/code/language/links".format(text_id))
        return

    # See https://otvet.mail.ru/question/97696484
    sentences = [s.text for s in razdel.sentenize(text)]
    for s in sentences:
        if len(s) > 1700:
            print("Skipping id {}, too long sentences".format(text_id))
            return
        if TEXT_PROCESSOR.count_text_part(s) < 0.7 and len(s) > 1000:
            print("Skipping id {}, bad sentences".format(text_id))
            return
        words = s.split()
        if any(len(word) > 70 for word in words):
            print("Skipping id {}, too long words".format(text_id))
            return

    return text


def main(input_path, output_path):
    with open(input_path, "r") as r, open(output_path, "w") as w:
        def flush(text_id, fragments):
            text = " ".join(fragments)
            text = preprocess_text(text, text_id)
            if not text:
                return
            w.write(json.dumps({
                "text": text,
                "id": int(text_id)
            }, ensure_ascii=False).strip() + "\n")

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
            fragments.append(line.strip())
        flush(text_id, fragments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    main(**vars(args))

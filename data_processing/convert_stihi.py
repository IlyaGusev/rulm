import argparse
import json

from tqdm import tqdm
from corus import load_taiga_stihi_metas, load_taiga_stihi

from data_processing.util import PlainArchive, TextProcessor


def main(
    input_path,
    output_path
):
    output_archive = PlainArchive(output_path)
    text_processor = TextProcessor(min_chars=100)
    metas = load_taiga_stihi_metas(input_path)

    metas_dict = dict()
    for meta in metas:
        metas_dict[str(meta.id)] = meta
    print(f"Metas count: {len(metas_dict)}")

    records = load_taiga_stihi(input_path, metas)
    for record in tqdm(records):
        rid = str(record.id)
        meta = metas_dict.get(rid)
        author = None
        title = None
        if meta is not None:
            author = meta.author.name
            title = meta.title
        text = record.text
        text = text_processor(text)
        if not text:
            continue

        lines = text.split("\n")
        fixed_lines = []
        bad_lines_count = 0
        short_lines_count = 0
        caps_lines_count = 0
        lines_count = len(lines)
        for line in lines:
            if text_processor.count_text_part(line) < 0.7:
                bad_lines_count += 1
                continue
            line = line.strip("-").strip("*").strip()
            if not line:
                continue
            if line[0] == "[" or line[-1] == "]":
                continue
            if line[0] == "(" or line[-1] == ")":
                continue
            if len(line) < 10:
                short_lines_count += 1
            upper_ch_count = sum(ch.isalpha() and ch.isupper() for ch in line)
            all_ch_count = sum(ch.isalpha() for ch in line)
            if upper_ch_count / all_ch_count > 0.5:
                caps_lines_count += 1
            line = line.replace(". ..", "...")
            line = line.replace("--", "-")
            fixed_lines.append(line)
        text = "\n".join(fixed_lines)

        if not text:
            continue
        if "PS" in text or "P.S." in text:
            continue
        if bad_lines_count / lines_count > 0.2:
            continue
        if short_lines_count / lines_count > 0.2:
            continue
        if caps_lines_count / lines_count > 0.1:
            continue

        char_count = len(text)
        bad_seq_count = (text.count("!!!") + text.count("...")) * 3
        if bad_seq_count / char_count > 0.01:
            continue

        output_archive.add_data(
            text=text,
            meta={
                "source": "stihi",
                "id": rid,
                "author": author,
                "title": title
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    main(**vars(args))

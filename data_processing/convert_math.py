import sys
import zipfile
import os
import json
from collections import Counter

from tqdm import tqdm
from bs4 import BeautifulSoup

from data_processing.util import gen_batch, PlainArchive


input_path = sys.argv[1]
output_path = sys.argv[2]


def get_txt_filepaths_from_zip(archive):
    txt_files = []
    for file_path in archive.namelist():
        if file_path.endswith('.txt') and file_path.startswith("whole_corpus/train"):
            print(file_path)
            txt_files.append(file_path)
    return txt_files


def parse_single_txt(path_to_txt, zip_file):
    with zip_file.open(path_to_txt) as f:
        lines = f.readlines()
    lines = [line.decode("utf-8").strip() for line in lines]
    lines = [line for line in lines if line]
    assert len(lines) % 2 == 0, "\n".join(lines)

    examples = []
    for i in range(len(lines) // 2):
        question = lines[2 * i]
        answer = lines[2 * i + 1]
        assert len(answer) < len(question), f"{answer} vs {question}"
        example =  question + " Ответ: " + answer
        examples.append(example)
    return "\n".join(examples)


if __name__ == "__main__":
    zip_file = zipfile.ZipFile(input_path)
    txt_paths = get_txt_filepaths_from_zip(zip_file)
    archive = PlainArchive(output_path)
    for path in tqdm(txt_paths, total=len(txt_paths)):
        full_text = parse_single_txt(path, zip_file)
        lines = full_text.split("\n")
        for batch_num, batch in enumerate(gen_batch(lines, 1000)):
            text = "\n".join(batch)
            archive.add_data(
                text=text,
                meta={
                    "source": "math",
                    "path": path,
                    "part_num": batch_num
                }
            )

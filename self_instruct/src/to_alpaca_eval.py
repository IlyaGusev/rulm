import json
from typing import Optional, List

import fire

from src.util.io import read_jsonl


def to_alpaca_eval(
    input_files: str,
    output_path: str,
):
    input_files = input_files.split(",")
    print(input_files)
    records = []
    for file_path in input_files:
        file_records = list(read_jsonl(file_path))
        generator = file_path.split("/")[-1].replace(".jsonl", "").replace("tasks_", "").replace("_answers", "")
        for r in file_records:
            r["generator"] = generator
        records.extend(file_records)
    with open(output_path, "w") as w:
        json.dump([{
            "instruction": r["instruction"],
            "output": r["answer"],
            "generator": r["generator"]
        } for r in records], w, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(to_alpaca_eval)

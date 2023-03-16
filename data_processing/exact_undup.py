import argparse
import json
import hashlib
from tqdm import tqdm

from data_processing.util import read_jsonl


def sha256str(s):
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def main(
    input_path,
    output_path,
    field
):
    seen = set()
    dups_count = 0
    with open(output_path, "w") as w:
        for record in tqdm(read_jsonl(input_path)):
            value = record[field]
            hash_st = sha256str(str(value)[:1000])
            if hash_st in seen:
                dups_count += 1
                continue
            seen.add(hash_st)
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
    print("Exact duplicates:", dups_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--field", type=str, default="text")
    args = parser.parse_args()
    main(**vars(args))

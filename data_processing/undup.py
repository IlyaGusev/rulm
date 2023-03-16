import argparse
import json
import re
import os
import hashlib
import fcntl
import multiprocessing
from multiprocessing.pool import ThreadPool
import zstandard
from tqdm import tqdm
from collections import defaultdict

from datasets import load_dataset
from datasketch import MinHash, MinHashLSH, LeanMinHash

from data_processing.util import read_jsonl, PlainArchive, ngrams


def re_tokenize(text):
    return re.findall(r'[а-яё-]+|[a-z-]+|\d+|\S', text, re.I)


def calc_fingerprint(record, ngram_size: int = 1, num_perm: int = 128):
    tokens = re_tokenize(record["text"])
    if ngram_size > 1:
        tokens = {" ".join(t) for t in ngrams(tokens, ngram_size)}
    tokens = [token.encode('utf-8') for token in tokens]

    minhash = MinHash(num_perm=num_perm)
    minhash.update_batch(tokens)

    lean_minhash = LeanMinHash(minhash)
    buf = bytearray(lean_minhash.bytesize())
    lean_minhash.serialize(buf)

    return {"minhash": buf}


def main(
    input_path,
    output_path,
    num_perm
):
    dataset = load_dataset("rulm/jsonl_loader.py", data_files={"train": [input_path]})["train"]
    dataset = dataset.map(
        function=calc_fingerprint,
        fn_kwargs={
            "num_perm": num_perm,
            "ngram_size": 1,
        },
        num_proc=os.cpu_count(),
        desc="Fingerprinting..."
    )

    archive = PlainArchive(output_path)
    #out = open(output_path, "w")

    threshold = 0.95
    false_positive_weight = 0.05
    lsh = MinHashLSH(
        threshold=threshold,
        weights=(false_positive_weight, 1 - false_positive_weight),
        num_perm=num_perm,
    )

    for idx, record in tqdm(enumerate(dataset)):
        minhash = LeanMinHash.deserialize(record["minhash"])

        is_dup = False
        for other_idx in lsh.query(minhash):
            other_record = dataset[other_idx]
            other_minhash = LeanMinHash.deserialize(other_record["minhash"])
            if minhash.jaccard(other_minhash) > threshold:
                is_dup = True
                break

        if not is_dup:
            record.pop("minhash")
            #out.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
            text = record["text"]
            meta = record["meta"]
            archive.add_data(text=text, meta=meta)

        lsh.insert(idx, minhash)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--num-perm", type=int, default=128)
    args = parser.parse_args()
    main(**vars(args))

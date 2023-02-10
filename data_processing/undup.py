import argparse

import os
import hashlib
import fcntl
import multiprocessing
from multiprocessing.pool import ThreadPool
import zstandard
from tqdm import tqdm
from collections import defaultdict

import razdel
from datasets import load_dataset
from datasketch import MinHash, MinHashLSH, LeanMinHash

from data_processing.util import read_jsonl, PlainArchive, ngrams


def calc_fingerprint(record, ngram_size: int = 1, num_perm: int = 128):
    tokens = [token.text for token in razdel.tokenize(record["text"])]
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
    num_perm,
    hashes_path
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
    lsh = MinHashLSH(threshold=0.95, num_perm=num_perm)
    for idx, record in tqdm(enumerate(dataset)):
        minhash = LeanMinHash.deserialize(record["minhash"])
        result = lsh.query(minhash)
        if not result:
            lsh.insert(str(idx), minhash)
            text = record["text"]
            meta = record["meta"]
            archive.add_data(text=text, meta=meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--hashes-path", type=str, default="hashes.txt")
    parser.add_argument("--num-perm", type=int, default=128)
    args = parser.parse_args()
    main(**vars(args))

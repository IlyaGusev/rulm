import sys
import os
import hashlib
import zstandard
from tqdm import tqdm
from collections import defaultdict

import razdel
from datasketch.minhash import MinHash, MinHashLSH

from data_processing.util import read_jsonl, PlainArchive, ngrams, UnionFind


B = 64

def calc_fingerprint(text):
    tokens = [token.text for token in razdel.tokenize(text)]
    tokens = {" ".join(t) for t in ngrams(tokens, 11)}
    tokens = [token.encode('utf-8') for token in tokens]
    minhash = MinHash(num_perm=B)
    minhash.update_batch(tokens)
    return minhash

input_path = sys.argv[1]
output_path = sys.argv[2]

archive = PlainArchive(output_path)
uf = UnionFind()
hash_tables = [defaultdict(list) for _ in range(B)]
for idx, record in tqdm(enumerate(read_jsonl(input_path))):
    text = record["text"]
    meta = record["meta"]
    if meta["source"] in ("math", ):
        continue
    minhash = calc_fingerprint(text)
    for hv, hash_table in zip(minhash.hashvalues, hash_tables):
        hash_table[int(hv)].append(idx)

for table in hash_tables:
    for cluster in table.values():
        if len(cluster) <= 1:
            continue
        idx = min(cluster)
        for x in cluster:
            uf.union(x, idx)

duplicates = list()
for idx, record in tqdm(enumerate(read_jsonl(input_path))):
    text = record["text"]
    meta = record["meta"]
    if meta["source"] in ("math", ):
        archive.add_data(text=text, meta=meta)
        continue

    if uf.find(idx) == idx:
        archive.add_data(text=text, meta=meta)
    else:
        duplicates.append((idx, uf.find(idx)))


with open("duplicates.txt", "w") as w:
    for idx1, idx2 in duplicates:
        w.write(f"{idx1}\t{idx2}\n")

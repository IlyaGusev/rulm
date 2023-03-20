import argparse
import random
import hashlib
from collections import defaultdict

from tqdm import tqdm

from data_processing.util import read_jsonl, PlainArchive


def sha256str(s):
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    return h.hexdigest()


sample_rates = defaultdict(lambda: 1.0)
#sample_rates["librusec"] = 1.0

parser = argparse.ArgumentParser()
parser.add_argument('-f','--files', nargs='+', dest='files', type=str, required=True)
parser.add_argument('--output-path', type=str, required=True)
args = parser.parse_args()

archive = PlainArchive(args.output_path)
seen = set()
for f in args.files:
    print(f)
    dups_count = 0
    for r in tqdm(read_jsonl(f)):
        meta = r["meta"]
        source = meta["source"]
        if random.random() > sample_rates[source]:
            continue
        text = r["text"]
        hash_st = sha256str(text[:1000])
        if hash_st in seen:
            dups_count += 1
            continue
        seen.add(hash_st)
        archive.add_data(text=text, meta=meta)
    print("Exact duplicates:", dups_count)

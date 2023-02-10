import sys
from collections import Counter

from tqdm import tqdm

from data_processing.util import ngrams, read_jsonl

input_path = sys.argv[1]
cnt = Counter()
for i, record in enumerate(tqdm(read_jsonl(input_path))):
    words = record["text"].split()
    cnt.update(list(set(ngrams(words, 2))))
    if i % 10000 == 0 and i != 0:
        for seq, _ in cnt.most_common(50):
            print(" ".join(seq))
        cnt = Counter()
        print()

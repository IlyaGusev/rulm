import sys
from typing import List
from collections import Counter

import razdel
from tqdm import tqdm

from data_processing.util import read_jsonl

input_path = sys.argv[1]
nrows = int(sys.argv[2])

def update_n_grams(n_grams: Counter, tokens: List[str], n: int = 13) -> None:
    local_n_grams = Counter()
    count = len(tokens)
    for i in range(min(count - n + 1, count)):
        n_gram = " ".join(tokens[i:i+n])
        local_n_grams[n_gram] += 1
    for n_gram, _ in local_n_grams.items():
        n_grams[n_gram] += 1


n_grams = Counter()
for i, record in enumerate(tqdm(read_jsonl(input_path))):
    text = record["text"]
    tokens = [token.text for token in razdel.tokenize(text)]
    update_n_grams(n_grams, tokens)
    if i == nrows:
        break

print(n_grams.most_common(20))

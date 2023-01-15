import sys

from tokenizers import Tokenizer
from transformers import AutoTokenizer
from tqdm import tqdm

from data_processing.util import read_jsonl, gen_batch_iter, PlainArchive

tokenizer_path = sys.argv[1]
input_path = sys.argv[2]
output_path = sys.argv[3]
max_id = 25000

archive = PlainArchive(output_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
for batch in tqdm(gen_batch_iter(read_jsonl(input_path), 1000)):
    batch_texts = [r["text"].replace("*", "").replace("=", "").replace("~", "").replace("\n", " ") for r in batch]
    batch_ids = tokenizer(batch_texts).input_ids
    for ids, record in zip(batch_ids, batch):
        bad_tokens = [i for i in ids if i > max_id or i == tokenizer.unk_token_id]
        count = len(bad_tokens)
        if count > 1:
            #print(record["text"])
            #print(tokenizer.decode(ids))
            #print(tokenizer.decode(bad_tokens))
            continue
        archive.add_data(text=record["text"], meta=record["meta"])

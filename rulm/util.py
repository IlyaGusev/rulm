import json


def gen_batch(records, batch_size):
    batch_start = 0
    while batch_start < len(records):
        batch_end = batch_start + batch_size
        batch = records[batch_start: batch_end]
        batch_start = batch_end
        yield batch


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)

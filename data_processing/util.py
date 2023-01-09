import unicodedata
import json


def gen_batch(records, batch_size):
    batch_start = 0
    while batch_start < len(records):
        batch_end = batch_start + batch_size
        batch = records[batch_start: batch_end]
        batch_start = batch_end
        yield batch


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ")
    text = text.replace("&quot;", '"')
    return text


def remove_non_printable(text):
    return "".join(c for c in text if c.isprintable())


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


class PlainArchive:
    def __init__(self, file_path):
        self.file_path = file_path
        self.fh = open(file_path, "w")

    def add_data(self, text, meta={}):
        self.fh.write(json.dumps({"text": text, "meta": meta}, ensure_ascii=False).strip() + "\n")

    def commit(self):
        self.fh.flush()

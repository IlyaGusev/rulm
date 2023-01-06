import sys
import json

from tqdm import tqdm
from corus import load_taiga_stihi_metas, load_taiga_stihi
from converters.lang_detector import FasttextLanguageDetector

input_path = sys.argv[1]
output_path = sys.argv[2]

lang_detector = FasttextLanguageDetector()
metas = load_taiga_stihi_metas(input_path)

metas_dict = dict()
for meta in metas:
    metas_dict[str(meta.id)] = meta
print(f"Metas count: {len(metas_dict)}")

records = load_taiga_stihi(input_path, metas)
with open(output_path, "w") as w:
    for record in tqdm(records):
        rid = str(record.id)
        meta = metas_dict.get(rid)
        author = None
        title = None
        if meta is not None:
            author = meta.author.name
            title = meta.title

        text = record.text
        if lang_detector(text)[0] != "ru":
            continue

        new_record = {
            "id": rid,
            "text": text,
            "author": author,
            "title": title
        }
        w.write(json.dumps(new_record, ensure_ascii=False) + "\n")

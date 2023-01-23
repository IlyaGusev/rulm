import sys
import json

from tqdm import tqdm
from corus import load_taiga_stihi_metas, load_taiga_stihi

from data_processing.util import PlainArchive, TextProcessor

input_path = sys.argv[1]
output_path = sys.argv[2]

output_archive = PlainArchive(output_path)
text_processor = TextProcessor()
metas = load_taiga_stihi_metas(input_path)

metas_dict = dict()
for meta in metas:
    metas_dict[str(meta.id)] = meta
print(f"Metas count: {len(metas_dict)}")

records = load_taiga_stihi(input_path, metas)
for record in tqdm(records):
    rid = str(record.id)
    meta = metas_dict.get(rid)
    author = None
    title = None
    if meta is not None:
        author = meta.author.name
        title = meta.title
    text = record.text
    text = text_processor(text)
    if not text:
        continue
    if text.count("...") >= 5:
        continue
    output_archive.add_data(
        text=text,
        meta={
            "source": "stihi",
            "id": rid,
            "author": author,
            "title": title
        }
    )

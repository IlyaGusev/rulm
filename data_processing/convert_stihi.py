import sys
import json

from tqdm import tqdm
from corus import load_taiga_stihi_metas, load_taiga_stihi

from data_processing.lang_detector import FasttextLanguageDetector
from data_processing.util import PlainArchive, remove_non_printable, normalize

BAD_SUBSTRINGS = (
    "• ",
    "+79",
    "@gmail",
    "var ",
    "<a ",
    "<p ",
    ".jpg",
    "http:"
)

input_path = sys.argv[1]
output_path = sys.argv[2]

lang_detector = FasttextLanguageDetector()
archive = PlainArchive(output_path)
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
    if lang_detector(text)[0] != "ru":
        continue

    text = normalize(text)
    text = " ".join(text.split())
    lines = [remove_non_printable(line.strip()) for line in text.split("\n")]
    lines = [line for line in lines if line and line not in ("***", )]
    fixed_lines = []
    for line in lines:
        if len(set(line.replace(" ", "").strip())) <= 1:
            continue
        fixed_lines.append(line)

    text = "\n".join(fixed_lines)
    if len(text) < 100:
        continue
    has_bad_ss = any(ss in text for ss in BAD_SUBSTRINGS)
    if has_bad_ss:
        continue
    if text.count("...") >= 5:
        continue
    archive.add_data(
        text=text,
        meta={
            "source": "stihi",
            "id": rid,
            "author": author,
            "title": title
        }
    )

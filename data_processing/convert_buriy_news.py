import sys
import json
from datetime import datetime

from corus import load_buriy_news
from tqdm import tqdm

from data_processing.util import normalize, remove_non_printable, PlainArchive
from data_processing.lang_detector import FasttextLanguageDetector

BAD_SUBSTRINGS = (
    "http",
    "{youtube}",
    "Показать комментарии",
    "//",
    "Видео No",
    "=",
    " ,",
    " :",
    " .",
    "О сериале",
    "О фильме",
    "Loading"
)

input_path = sys.argv[1]
output_path = sys.argv[2]

lang_detector = FasttextLanguageDetector()
archive = PlainArchive(output_path)

for record in tqdm(load_buriy_news(input_path)):
    text = record.text
    text = normalize(text)
    text = remove_non_printable(" ".join(text.split())).strip()
    if lang_detector(text)[0] != "ru":
        continue
    has_bad_ss = any(ss in text for ss in BAD_SUBSTRINGS)
    if has_bad_ss:
        continue
    archive.add_data(
        text=text,
        meta={
            "source": "buriy_news",
            "url": record.url,
            "title": record.title,
            "timestamp": int(datetime.timestamp(record.timestamp))
        }
    )


import sys
import json
from datetime import datetime

from corus import load_buriy_news
from tqdm import tqdm

from data_processing.util import TextProcessor, PlainArchive

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

text_processor = TextProcessor(join_lines=True)
archive = PlainArchive(output_path)

for record in tqdm(load_buriy_news(input_path)):
    text = text_processor(record.text)
    if not text:
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


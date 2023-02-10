import argparse
import os
import json
from datetime import datetime
import logging
import html2text

import requests
from tqdm import tqdm

from data_processing.util import PlainArchive, TextProcessor


text_processor = TextProcessor(
    min_text_part=0.0,
    fix_punct=False,
    fix_spaces=False,
    fix_short_lines=False,
    check_bad_ss=False,
    check_languages=False
)

def html2text_setup():
    instance = html2text.HTML2Text(bodywidth=0)
    instance.ignore_links = True
    instance.ignore_images = True
    instance.ignore_tables = True
    instance.ignore_emphasis = True
    instance.mark_code = True
    instance.ul_item_mark = ""
    return instance


def worker(article_id, output_archive):
    url = "https://habr.com/kek/v2/articles/{}".format(article_id)

    try:
        r = requests.get(url)
        if r.status_code == 503:
            logging.critical("503 Error")
            return
        if r.json().get("httpCode", 200) != 200:
            return
    except Exception as e:
        print(e)
        return

    data = r.json()

    lang = data["lang"]
    if lang != "ru":
        return

    content = data["textHtml"]

    html2text = html2text_setup()
    text = html2text.handle(content)
    paragraphs = [p.strip("\r").rstrip() for p in text.split("\n") if p.strip()]
    text = "\n".join(paragraphs)
    text = text_processor(text)
    if not text:
        return

    time_published = int(datetime.strptime(
        data["timePublished"], "%Y-%m-%dT%H:%M:%S%z"
    ).timestamp())

    author = data.get("author")
    if author:
        author = author.get("alias")

    output_archive.add_data(
        text=text,
        meta={
            "id": int(data["id"]),
            "time_published": time_published,
            "author": author,
            "title": data.get("titleHtml"),
            "statistics": data["statistics"],
        }
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-id", type=int, default=0)
    parser.add_argument("--max-id", type=int, default=710000)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    start_time = datetime.now()
    output_archive = PlainArchive(args.output_path)
    for article_id in tqdm(range(args.max_id, args.min_id, -1)):
        worker(article_id, output_archive)

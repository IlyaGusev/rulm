import argparse
import os
import json
import unicodedata
from datetime import datetime
import logging
import html2text

import requests
from tqdm import tqdm

from data_processing.util import TextProcessor


TEXT_PROCESSOR = TextProcessor(
    min_chars=5,
    min_text_part=0.0,
    fix_punct=False,
    fix_spaces=False,
    fix_short_lines=False,
    check_code=False,
    check_pii=False,
    check_links=False,
    check_languages=False,
    check_email=False,
    check_text_part=False
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


def process_timestamp(time_published):
    return int(datetime.strptime(
        time_published, "%Y-%m-%dT%H:%M:%S%z"
    ).timestamp())


def process_author(author):
    author_fullname = author.get("fullname", "")
    author_alias = author.get("alias", "")
    author = ""
    if author_fullname and not author_alias:
        author = author_fullname
    elif author_alias and not author_fullname:
        author = author_alias
    elif author_fullname and author_alias:
        author = "{} ({})".format(author_fullname, author_alias)
    return author


def html2markdown(html):
    html2text = html2text_setup()
    markdown = html2text.handle(html)
    paragraphs = [p.rstrip() for p in markdown.split("\n") if p.strip()]
    markdown = "\n".join(paragraphs)
    markdown = TEXT_PROCESSOR(markdown)
    return markdown


def parse_post(post_id):
    api_url = "https://habr.com/kek/v2/articles/{}".format(post_id)
    post_url = "https://habr.com/ru/post/{}/".format(post_id)

    try:
        r = requests.get(api_url)
        if r.status_code == 503:
            logging.critical("503 Error")
            return
        if r.json().get("httpCode", 200) != 200:
            return
    except Exception as e:
        print(e)
        return

    data = r.json()

    # Process text
    text_html = data["textHtml"]
    try:
        text_html.encode("utf-8")
    except UnicodeEncodeError as e:
        # Fix for two weird posts
        bad_bytes = (
            "\uded2",
            "\ude64",
            "\udd00",
            "\udc40"
        )
        for ss in bad_bytes:
            text_html = text_html.replace(ss, "")
    text_markdown = html2markdown(text_html)
    if not text_markdown:
        print("Bad text!", post_id)
        return None

    # Process lead
    lead_html = data["leadData"]["textHtml"]
    lead_markdown = html2markdown(lead_html)

    # Process title
    html2text = html2text_setup()
    title = data.get("titleHtml")
    if not title:
        print("No title!", post_id)
        return None
    title = html2text.handle(title).strip()

    # Process timestamp
    time_published = process_timestamp(data["timePublished"])

    # Process author
    author = data.get("author")
    if author:
        author = process_author(author)

    # Process other meta
    hubs = data.get("hubs", [])
    hubs = [hub["alias"] for hub in hubs if hub.get("alias")]
    flows = data.get("flows", [])
    flows = [flow["alias"] for flow in flows if flow.get("alias")]
    tags = data.get("tags", [])
    tags = [tag["titleHtml"] for tag in tags if tag.get("titleHtml")]
    labels = data.get("postLabels", [])
    original_url, original_author = None, None
    for label in labels:
        if label["type"] == "translation":
            original_author = label["data"]["originalAuthorName"]
            original_url = label["data"]["originalUrl"]
    labels = [label["type"] for label in labels]

    return {
        "id": int(data["id"]),
        "language": data["lang"],
        "url": post_url,
        "text_markdown": text_markdown,
        "text_html": text_html,
        "lead_markdown": lead_markdown,
        "lead_html": lead_html,
        "type": data.get("postType"),
        "labels": labels,
        "original_author": original_author,
        "original_url": original_url,
        "time_published": time_published,
        "author": author,
        "title": title,
        "statistics": data["statistics"],
        "hubs": hubs,
        "flows": flows,
        "tags": tags,
        "reading_time": data["readingTime"],
        "format": data["format"],
        "complexity": data["complexity"]
    }


def parse_comments(post_id):
    api_url = "https://habr.com/kek/v2/articles/{}/comments".format(post_id)
    try:
        r = requests.get(api_url)
        if r.status_code == 503:
            logging.critical("503 Error")
            return []
        if r.json().get("httpCode", 200) != 200:
            return []
    except Exception as e:
        print(e)
        return []

    data = r.json()
    if not data.get("comments"):
        return []
    comments = list(data["comments"].values())

    processed_comments = []
    for comment in comments:
        message_html = comment["message"]
        message_markdown = html2markdown(message_html)
        if not message_markdown:
            message_markdown = ""

        author = comment.get("author")
        if author:
            author = process_author(author)
        children = [int(child) for child in comment["children"]]

        processed_comments.append({
            "id": int(comment["id"]),
            "parent_id": int(comment["parentId"]) if comment.get("parentId") else None,
            "level": int(comment["level"]),
            "time_published": process_timestamp(comment["timePublished"]),
            "score": comment["score"],
            "votes": comment["votesCount"],
            "message_html": message_html,
            "message_markdown": message_markdown,
            "author": author,
            "children": children
        })

    return processed_comments


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-id", type=int, default=0)
    parser.add_argument("--max-id", type=int, default=484181)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    start_time = datetime.now()
    with open(args.output_path, "w") as w:
        for post_id in tqdm(range(args.max_id, args.min_id, -1)):
            record = parse_post(post_id)
            if not record:
                continue
            comments = parse_comments(post_id)
            record["comments"] = comments
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")

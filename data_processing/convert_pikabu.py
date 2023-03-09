import argparse
import json

import psycopg2
import html2text
from tqdm import tqdm

from data_processing.util import PlainArchive, TextProcessor

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


def html2markdown(html):
    html2text = html2text_setup()
    try:
        markdown = html2text.handle(html)
    except Exception:
        return None
    paragraphs = [p.rstrip() for p in markdown.split("\n") if p.strip()]
    markdown = "\n".join(paragraphs)
    return TEXT_PROCESSOR(markdown)


stories_mapping = {
    "pikabu_id": "id",
    "rating": "rating",
    "number_of_pluses": "pluses",
    "number_of_minuses": "minuses",
    "story_url": "url",
    "tags": "tags",
    "title": "title",
    "created_at_timestamp": "timestamp",
    "author_id": "author_id",
    "author_username": "username",
    "content_blocks": "blocks"
}

comments_mapping = {
    "pikabu_id": "id",
    "parent_id": "parent_id",
    "created_at_timestamp": "timestamp",
    "text": "text_html",
    "images": "images",
    "rating": "rating",
    "number_of_pluses": "pluses",
    "number_of_minuses": "minuses",
    "author_id": "author_id",
    "author_username": "username"
}

def fix_blocks(blocks):
    fixed_blocks = []
    for block in blocks:
        block_type = block["type"]
        assert block_type in ("i", "v", "t", "vf", "if"), block
        if block_type == "i":
            image_url = block["data"]["large"]
            if not image_url:
                image_url = block["data"]["small"]
            fixed_blocks.append({
                "data": image_url,
                "type": "image"
            })
        elif block_type == "v":
            video_url = block["data"]["url"]
            fixed_blocks.append({
                "data": video_url,
                "type": "video"
            })
        elif block_type == "t":
            text = block["data"]
            fixed_blocks.append({
                "data": text,
                "type": "text"
            })
        elif block_type == "vf":
            video_url = block["data"]["mp4"]["url"]
            fixed_blocks.append({
                "data": video_url,
                "type": "video"
            })
        else:
            continue
    return fixed_blocks


def blocks_to_markdown(blocks):
    text_blocks = []
    for block in blocks:
        if block["type"] != "text":
            continue
        markdown = html2markdown(block["data"])
        if not markdown:
            continue
        text_blocks.append(markdown)
    return "\n".join(text_blocks)


def main(output_path):

    with open(output_path, "w") as w:
        with psycopg2.connect("dbname=pikabu user=postgres password=postgres") as connection:
            with connection.cursor(name="stories") as cursor:
                cursor.itersize = 20000
                cursor.execute("SELECT * FROM pikabu_stories")
                for row in tqdm(cursor):
                    header = [desc.name for desc in cursor.description]
                    record = dict(zip(header, row))
                    record = {stories_mapping[k]: v for k, v in record.items() if k in stories_mapping}
                    record["blocks"] = fix_blocks(record["blocks"])
                    record["text_markdown"] = blocks_to_markdown(record["blocks"])
                    comments = {k: [] for k in comments_mapping.values()}
                    comments["text_markdown"] = []
                    with connection.cursor() as comments_cursor:
                        comments_cursor.execute("SELECT * FROM pikabu_comments WHERE story_id = {}".format(record["id"]))
                        for row in comments_cursor:
                            header = [desc.name for desc in comments_cursor.description]
                            comment = dict(zip(header, row))
                            comment = {comments_mapping[k]: v for k, v in comment.items() if k in comments_mapping}
                            comment["images"] = [i["large_url"] if i["large_url"] else i["small_url"] for i in comment["images"]]
                            comment["text_markdown"] = html2markdown(comment["text_html"])
                            for k, v in comment.items():
                                comments[k].append(v)
                    record["comments"] = comments
                    w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    main(**vars(args))

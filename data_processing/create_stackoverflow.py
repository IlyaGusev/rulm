# Based on https://github.com/EleutherAI/stackexchange-dataset/blob/master/pairer.py

import argparse
import os
import re
import sys
import json
import traceback
import xml.etree.ElementTree as etree
from datetime import datetime
from collections import defaultdict

import html2text
from tqdm import tqdm

from data_processing.util import PlainArchive, TextProcessor


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
        time_published, "%Y-%m-%dT%H:%M:%S.%f"
    ).timestamp())


def is_question(elem_attribs):
    post_type_id = elem_attribs["PostTypeId"]
    return post_type_id is not None and post_type_id == "1"


def is_answer(elem_attribs):
    post_type_id = elem_attribs["PostTypeId"]
    return post_type_id is not None and post_type_id == "2"


def is_accepted_answer(a_attribs, q_attribs):
    assert is_question(q_attribs), "Must be a question to have an accepted answer"
    assert is_answer(a_attribs), "Must be an answer to be an accepted answer"
    accepted_answer_id = q_attribs["AcceptedAnswerId"]
    answer_id = a_attribs["Id"]
    if accepted_answer_id is None:
        return False
    if accepted_answer_id == answer_id:
        return True
    return False


def has_answers(elem_attribs):
    assert is_question(elem_attribs), "Must be a question to have answers"
    answer_count = elem_attribs["AnswerCount"]
    return answer_count is not None and int(answer_count)


def trim_question(elem_attribs):
    assert is_question(elem_attribs)
    to_keep = {
        "Id",
        "Body",
        "Title",
        "Tags",
        "AnswerCount",
        "AcceptedAnswerId",
        "PostTypeId",
        "Score",
        "CreationDate",
        "ViewCount",
        "OwnerUserId",
        "OwnerDisplayName"
    }
    for x in list(elem_attribs.keys()):
        if x not in to_keep:
            elem_attribs.pop(x, None)
    elem_attribs["ParsedAnswers"] = 0
    elem_attribs["Answers"] = {}
    return elem_attribs


def trim_answer(elem_attribs):
    assert is_answer(elem_attribs)
    to_keep = [
        "Id",
        "CreationDate",
        "Body",
        "Score",
        "OwnerUserId",
        "OwnerDisplayName"
    ]
    return {item: elem_attribs[item] for item in to_keep}


class Converter:
    def __init__(self, posts_path, users_path, comments_path, output_path, min_score=-1000, max_responses=1000):
        self.posts_path = posts_path
        self.users_path = users_path
        self.comments_path = comments_path

        self.questions = defaultdict(lambda: None, {})
        self.records = dict()
        self.users = dict()
        self.comments = defaultdict(list)
        self.output_file = open(output_path, "w")

        self.min_score = min_score
        self.max_responses = max_responses
        self.text_processor = TextProcessor(
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

    def __call__(self):
        desc = "Parsing users XML file: {}".format(self.users_path)
        for event, elem in tqdm(etree.iterparse(self.users_path, events=('end',)), desc=desc):
            if elem.tag != "row":
                continue
            try:
                attribs = defaultdict(lambda: None, elem.attrib)
                user_id = int(attribs["Id"])
                user_name = attribs["DisplayName"]
                self.users[user_id] = user_name
                elem.clear()
            except:
                traceback.print_exc()

        desc = "Parsing comments XML file: {}".format(self.comments_path)
        for event, elem in tqdm(etree.iterparse(self.comments_path, events=('end',)), desc=desc):
            if elem.tag != "row":
                continue
            try:
                attribs = defaultdict(lambda: None, elem.attrib)
                comment_id = int(attribs["Id"])
                post_id = int(attribs["PostId"])
                text = attribs["Text"]
                if not text or not text.strip():
                    continue
                author = self.users[int(attribs["UserId"])] if attribs["UserId"] else attribs["UserDisplayName"]
                timestamp = process_timestamp(attribs["CreationDate"])
                score = int(attribs["Score"])
                self.comments[post_id].append({
                    "text": text,
                    "author": author,
                    "comment_id": comment_id,
                    "score": score,
                    "timestamp": timestamp
                })
                elem.clear()
            except:
                traceback.print_exc()

        desc = "Parsing posts XML file: {}".format(self.posts_path)
        for event, elem in tqdm(etree.iterparse(self.posts_path, events=('end',)), desc=desc):
            if elem.tag != "row":
                continue
            try:
                attribs = defaultdict(lambda: None, elem.attrib)
                if is_question(attribs):
                    self.questions[attribs["Id"]] = trim_question(attribs)
                    self.check_complete({"ParentId": attribs["Id"]})
                elif is_answer(attribs):
                    self.add_answer(attribs)
                    self.check_complete(attribs)
                elem.clear()
            except:
                traceback.print_exc()

    def to_markdown(self, html):
        html2text = html2text_setup()
        markdown = html2text.handle(html)
        paragraphs = [p.rstrip() for p in markdown.split("\n") if p.strip()]
        markdown = "\n".join(paragraphs)
        markdown = self.text_processor(markdown)
        return markdown

    def is_above_threshold(self, a_attribs):
        assert is_answer(a_attribs), "Must be an answer to be above threshold"
        score = a_attribs["Score"]
        return score is not None and int(score) >= self.min_score

    def add_answer(self, a_attribs):
        if a_attribs is None:
            return

        assert is_answer(a_attribs), "Must be an answer to add to parent"
        parent_id = a_attribs["ParentId"]
        answer_id = a_attribs["Id"]
        if self.questions[parent_id] is None:
            return
        if answer_id is None:
            return

        is_accepted = is_accepted_answer(a_attribs, self.questions[parent_id])
        is_good_score = self.is_above_threshold(a_attribs)
        if is_accepted or is_good_score:
            self.questions[parent_id]["Answers"][answer_id] = trim_answer(a_attribs)
        self.questions[parent_id]["ParsedAnswers"] += 1

    def check_complete(self, a_attribs):
        assert a_attribs is not None
        parent_id = a_attribs["ParentId"]
        parent = self.questions[parent_id]
        if parent is None:
            return
        answers_count = parent["AnswerCount"]
        parsed_answers_count = parent["ParsedAnswers"]
        if answers_count is None or parsed_answers_count is None:
            return
        answers_count = int(answers_count)
        parsed_answers_count = int(parsed_answers_count)
        if answers_count != parsed_answers_count:
            return

        question_id = int(parent["Id"])
        record = {
            "question_id": question_id,
            "answer_count": answers_count,
            "url": "https://ru.stackoverflow.com/questions/{}".format(question_id)
        }
        record["score"] = int(parent["Score"]) if parent["Score"] is not None else None
        tags = parent["Tags"] if parent["Tags"] is not None else None
        tags = tags[1:-1].split("><") if tags and len(tags) >= 2 else []
        record["tags"] = tags
        record["title"] = parent["Title"] if parent["Title"] is not None else None
        record["views"] = int(parent["ViewCount"]) if parent["ViewCount"] is not None else None
        author = self.users[int(parent["OwnerUserId"])] if parent["OwnerUserId"] else parent["OwnerDisplayName"]
        record["author"] = author
        record["comments"] = self.comments[int(parent_id)]
        if parent["CreationDate"] is not None:
            record["timestamp"] = process_timestamp(parent["CreationDate"])
        if parent["Body"] is not None:
            record["text_html"] = parent["Body"]
            record["text_markdown"] = self.to_markdown(parent["Body"])
            if not record["text_markdown"]:
                return

        accepted_answer_id = parent["AcceptedAnswerId"]
        accepted_answer_id = int(accepted_answer_id) if accepted_answer_id else None
        if parent["Answers"] is not None:
            scores = {k: int(a["Score"]) for k, a in parent["Answers"].items()}
            scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            answers = []
            for key, score in scores[:self.max_responses]:
                answer_attrs = parent["Answers"][key]
                answer_text_html = answer_attrs["Body"]
                answer_id = int(answer_attrs["Id"])
                if answer_attrs["OwnerUserId"]:
                    answer_author = self.users[int(answer_attrs["OwnerUserId"])]
                else:
                    answer_author = answer_attrs["OwnerDisplayName"]
                timestamp = process_timestamp(answer_attrs["CreationDate"])
                answer_record = {
                    "answer_id": answer_id,
                    "timestamp": timestamp,
                    "is_accepted": int(answer_id == accepted_answer_id),
                    "text_html": answer_text_html,
                    "text_markdown": self.to_markdown(answer_text_html),
                    "score": int(answer_attrs["Score"]),
                    "author": answer_author,
                    "comments": self.comments[answer_id]
                }
                if not answer_record["text_markdown"]:
                    continue
                answers.append(answer_record)
            record["answers"] = answers
        self.output_file.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
        self.questions.pop(parent_id, None)


def main(
    posts_path,
    comments_path,
    users_path,
    output_path
):
    converter = Converter(
        posts_path=posts_path,
        comments_path=comments_path,
        users_path=users_path,
        output_path=output_path
    )
    converter()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--posts-path", type=str, required=True)
    parser.add_argument("--comments-path", type=str, required=True)
    parser.add_argument("--users-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))


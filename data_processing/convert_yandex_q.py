import argparse
import json
from datetime import datetime

from tqdm import tqdm

from data_processing.util import read_jsonl

question_mapping = {
    "title": "title",
    "id": "id",
    "id2": "id2",
    "plainText": "text_plain",
    "formattedText": "text_html",
    "negativeVotes": "negative_votes",
    "positiveVotes": "positive_votes",
    "quality": "quality",
    "author": "author",
    "viewsCount": "views",
    "votes": "votes",
    "tags": "tags",
    "approvedAnswerId": "approved_answer"
}

answer_mapping = {
    "id": "id",
    "id2": "id2",
    "plainText": "text_plain",
    "formattedText": "text_html",
    "negativeVotes": "negative_votes",
    "positiveVotes": "positive_votes",
    "quality": "quality",
    "author": "author",
    "repostsCount": "reposts",
    "viewsCount": "views",
    "votes": "votes"
}

def process_timestamp_1(time_published):
    return int(datetime.strptime(
        time_published, "%Y-%m-%dT%H:%M:%S%z"
    ).timestamp())


def process_timestamp_2(time_published):
    return int(datetime.strptime(
        time_published, "%Y-%m-%dT%H:%M:%S.%f%z"
    ).timestamp())


def process_timestamp(time_published):
    try:
        return process_timestamp_1(time_published)
    except Exception:
        return process_timestamp_2(time_published)


def main(
    input_path,
    output_path
):
    with open(output_path, "w") as w:
        for record in tqdm(read_jsonl(input_path)):
            timestamp = process_timestamp(record["created"])
            question = {question_mapping[k]: v for k, v in record.items() if k in question_mapping}
            question["timestamp"] = timestamp
            answers = {k: [] for k in answer_mapping.values()}
            answers["timestamp"] = []
            for answer_record in record["answers"]:
                answer = {answer_mapping[k]: answer_record[k] for k in answer_mapping}
                answer["timestamp"] = process_timestamp(answer_record["created"])
                for k, v in answer.items():
                    answers[k].append(v)
            question["answers"] = answers
            w.write(json.dumps(question, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    main(**vars(args))

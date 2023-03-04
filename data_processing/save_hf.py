import argparse
import json

from datasets import load_dataset
from tqdm import tqdm

from data_processing.util import TextProcessor, PlainArchive


def revert_flattening(records):
    fixed_records = []
    for key, values in records.items():
        if not fixed_records:
            fixed_records = [{} for _ in range(len(values))]
        for i, value in enumerate(values):
            fixed_records[i][key] = value
    return fixed_records


def dump_habr(archive):
    text_processor = TextProcessor()
    habr = load_dataset('IlyaGusev/habr', split="train", streaming=True)
    text_processor = TextProcessor(
        min_chars=100,
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

    for row in tqdm(habr):
        if row["language"] != "ru":
            continue
        text = text_processor(row["text_markdown"])
        if not text:
            continue

        comments = revert_flattening(row["comments"])
        comments.sort(key=lambda x: x["time_published"])

        users, users_set = list(), set()
        for comment in comments:
            user = comment["author"]
            if user in users_set:
                continue
            users.append(user)
            users_set.add(user)
        user2id = {user: user_id for user_id, user in enumerate(users)}

        id2comment = {c["id"]: c for c in comments}
        saved_comments = set()

        def handle_comment(comment, current_text):
            comment_id = comment["id"]
            if comment_id in saved_comments:
                return current_text
            author = "Пользователь {}".format(user2id[comment["author"]])
            reply = ""
            parent_id = comment["parent_id"]
            if parent_id and parent_id in id2comment:
                parent_author = user2id[id2comment[parent_id]["author"]]
                reply = " (в ответ {})".format(parent_author)
            message = comment["message_markdown"]
            if not message.strip():
                message = "<картинка>"
            elif message.strip() == "UFO just landed and posted this here":
                message = "<удаленный комментарий>"
            current_text += "{}{}:\n{}\n".format(author, reply, message)
            saved_comments.add(comment_id)
            children = comment["children"]
            for child_id in children:
                child = id2comment[child_id]
                current_text = handle_comment(child, current_text)
            return current_text

        comments_text = ""
        for comment in comments:
            comments_text = handle_comment(comment, comments_text)
        final_text = text + "\n\n" + comments_text.strip()

        archive.add_data(
            text=final_text,
            meta= {
                "source": "habr",
                "title": row["title"],
                "url": row["url"],
                "timestamp": row["time_published"]
            }
        )


def dump_stackoverflow(archive):
    text_processor = TextProcessor(
        min_chars=100,
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
    stackoverflow = load_dataset("IlyaGusev/ru_stackoverflow", split="train", streaming=True)

    def process_comments(comments, is_question):
        comments = revert_flattening(comments)
        if not comments:
            return ""

        users, users_set = list(), set()
        for comment in comments:
            user = comment["author"]
            if user in users_set:
                continue
            if not user:
                continue
            users.append(user)
            users_set.add(user)
        user2id = {user: user_id for user_id, user in enumerate(users)}

        text = "Комментарии к {}:\n".format("вопросу" if is_question else "ответу")
        for comment in comments:
            author = comment["author"]
            if not author:
                continue
            comment_text = comment["text"]
            for user in users:
                comment_text = comment_text.replace(user, str(user2id[user]))
            text += "Пользователь {}: {}\n".format(user2id[author], comment_text)
        return text


    def process_answers(answers):
        if not answers:
            return ""
        text = ""
        for answer in answers:
            text = "Ответ:\n{}\n".format(answer["text_markdown"])
            text += process_comments(answer["comments"], is_question=False)
        return text


    for row in tqdm(stackoverflow):
        title = row["title"].strip()
        author = row["author"]
        question = text_processor(row["text_markdown"].strip())
        if not question:
            continue
        answers = revert_flattening(row["answers"])
        comments_text = process_comments(row["comments"], is_question=True)
        answers_text = process_answers(answers)
        final_text = f"{title}\nВопрос:\n{question}\n{comments_text}\n\n{answers_text}"
        archive.add_data(
            text=final_text,
            meta={
                "source": "stackoverflow",
                "timestamp": row["timestamp"],
                "url": row["url"]
            }
        )


def dump_gazeta(archive):
    text_processor = TextProcessor()
    gazeta = load_dataset('IlyaGusev/gazeta', revision="v2.0", split="train")
    for row in tqdm(gazeta):
        title = row["title"]
        text = text_processor(row["text"])
        if not text:
            continue
        final_text = title + "\n" + text
        archive.add_data(
            text=final_text,
            meta={
                "source": "gazeta",
                "date": row["date"],
                "url": row["url"]
            }
        )


def dump_medical_qa(archive):
    text_processor = TextProcessor(join_lines=False)
    medical_qa = load_dataset("blinoff/medical_qa_ru_data", split="train")
    for row in tqdm(medical_qa):
        text = "Вопрос: " + row["desc"]
        for i, answer in enumerate(row["ans"].split(";\n")):
            text += f"\nОтвет {i+1}: {answer}"
        text = text_processor(text)
        if not text:
            continue
        archive.add_data(
            text=text,
            meta={
                "source": "medical_qa",
                "theme": row["theme"],
                "date": row["date"],
                "categ": row["categ"],
                "spec10": row["spec10"]
            }
        )


def dump_sentiment(archive):
    text_processor = TextProcessor(join_lines=True)
    sentiment = load_dataset("Tatyana/ru_sentiment_dataset", split="train")
    labels = {
        0: "NEUTRAL",
        1: "POSITIVE",
        2: "NEGATIVE"
    }
    for row in tqdm(sentiment):
        text = text_processor(row["text"])
        if not text:
            continue
        archive.add_data(
            text=text,
            meta={
                "source": "sentiment",
                "sentiment": labels[int(row["sentiment"])]
            }
        )


def main(output_path):
    archive = PlainArchive(output_path)
    dump_medical_qa(archive)
    dump_sentiment(archive)
    dump_gazeta(archive)
    dump_stackoverflow(archive)
    dump_habr(archive)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    main(**vars(args))

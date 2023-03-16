import argparse
import json
import random
from collections import defaultdict

import razdel
from datasets import load_dataset
from tqdm import tqdm

from data_processing.util import TextProcessor, PlainArchive, gen_batch


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
        final_text = row["title"] + "\n" + text + "\n\n" + comments_text.strip()

        archive.add_data(
            text=final_text,
            meta= {
                "source": "habr",
                "url": row["url"]
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
                "url": row["url"]
            }
        )


def dump_pikabu(archive):
    post_text_processor = TextProcessor(
        min_chars=50,
        min_text_part=0.7,
        fix_punct=False,
        fix_spaces=False,
        fix_short_lines=False,
        check_code=True,
        check_pii=True,
        check_links=True,
        check_languages=True,
        check_email=True,
        check_text_part=True
    )
    comments_text_processor = TextProcessor(
        min_chars=5,
        min_text_part=0.0,
        fix_punct=False,
        fix_spaces=False,
        fix_short_lines=False,
        check_code=False,
        check_pii=True,
        check_links=False,
        check_languages=False,
        check_email=True,
        check_text_part=False
    )

    pikabu = load_dataset("IlyaGusev/pikabu", split="train", streaming=True)
    for row in tqdm(pikabu):
        final_text = ""
        title = row["title"]
        if title:
            final_text += title + "\n"
        text = row["text_markdown"]
        if text:
            final_text += text
        final_text = post_text_processor(final_text)
        if not final_text:
            continue

        comments = revert_flattening(row["comments"])
        comments.sort(key=lambda x: x["timestamp"])
        id2children = defaultdict(list)
        for comment in comments:
            parent_id = comment["parent_id"]
            if parent_id == 0:
                continue
            id2children[parent_id].append(comment["id"])
        id2comment = {c["id"]: c for c in comments}

        users, users_set = list(), set()
        for comment in comments:
            user = comment["username"]
            if user in users_set:
                continue
            if not user:
                continue
            users.append(user)
            users_set.add(user)
        user2id = {user: user_id for user_id, user in enumerate(users)}

        saved_comments = set()
        def handle_comment(comment, current_text):
            comment_id = comment["id"]
            if comment_id in saved_comments:
                return current_text

            author = "Пользователь {}".format(user2id[comment["username"]])
            reply = ""
            parent_id = comment["parent_id"]
            if parent_id and parent_id in id2comment:
                parent_author = user2id[id2comment[parent_id]["username"]]
                reply = " (в ответ {})".format(parent_author)
            message = comment["text_markdown"]
            if not message and comment["images"]:
                message = "<картинка>"
            else:
                if not message:
                    message = ""
                message = comments_text_processor(message)
                if not message:
                    return current_text
                orig_message = message
                for user in users:
                    message = message.replace("@" + user, "@" + str(user2id[user]))
            current_text += "{}{}:\n{}\n".format(author, reply, message)
            saved_comments.add(comment_id)
            children = id2children[comment["id"]]
            for child_id in children:
                child = id2comment[child_id]
                current_text = handle_comment(child, current_text)
            return current_text

        comments_text = ""
        for comment in comments:
            comments_text = handle_comment(comment, comments_text)
        final_text = "{}\n\nКомментарии:\n{}".format(final_text, comments_text)
        archive.add_data(
            text=final_text,
            meta={
                "source": "pikabu",
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
                "url": row["url"]
            }
        )


def dump_librusec(archive, sample_rate=0.15):
    max_sentences_count = 100
    text_processor = TextProcessor()
    librusec = load_dataset("IlyaGusev/librusec", split="train", streaming=True)
    for row in tqdm(librusec):
        text = row["text"]
        sentences = [s.text for s in razdel.sentenize(text)]
        for batch in gen_batch(sentences, batch_size=max_sentences_count):
            fragment = " ".join(batch)
            if fragment.count("//") > 5:
                continue
            if text_processor.count_text_part(fragment) < 0.85:
                continue
            if random.random() > sample_rate:
                continue
            archive.add_data(
                text=fragment,
                meta={
                    "source": "librusec",
                    "url": None
                }
            )


def dump_news(archive):
    text_processor = TextProcessor()
    news = load_dataset("IlyaGusev/ru_news", split="train", streaming=True)
    for row in tqdm(news):
        text = row["text"]
        url = row["url"]
        source = row["source"]
        text = text_processor(text)
        if not text:
            continue
        archive.add_data(
            text=text,
            meta={
                "source": source,
                "url": url
            }
        )


def main(output_path):
    archive = PlainArchive(output_path)

    print("==== Librusec ====")
    dump_librusec(archive)
    print("==== News ====")
    dump_news(archive)
    print("==== Pikabu ====")
    dump_pikabu(archive)
    print("==== Gazeta ====")
    dump_gazeta(archive)
    print("==== StackOverflow ====")
    dump_stackoverflow(archive)
    print("==== Habr ====")
    dump_habr(archive)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    main(**vars(args))

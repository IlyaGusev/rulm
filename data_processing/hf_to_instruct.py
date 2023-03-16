import argparse
import json
import random
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm


TASK_INSTRUCTIONS = {
    "headline": [
        "Сгенерируй хороший заголовок для статьи ниже.",
        "Придумай идеальное название для поста, который представлен далее.",
        "Сформулируй заголовок для статьи, которая находится ниже этой инструкции.",
        "Составь заголовок для статьи.",
        "Напиши название для следующего поста."
    ],
    "keywords": [
        "Протегируй текст ниже.",
        "Придумай теги для поста, который представлен далее.",
        "Напиши несколько тегов, подходящих для статьи ниже.",
        "Предложи несколько ключевых слов, которые могут быть использованы для описания статьи.",
        "Придумай несколько тегов, которые можно использовать для описания содержания статьи, которая находится ниже."
    ],
    "complexity": [
        "Как можно точнее оцени сложность текста, написанного ниже. Возможные опции: низкая, средняя, высокая.",
        "Укажи сложность поста по шкале из 3 делений: низкая, средняя, выскокая.",
        "Просмотри текст, который идет далее, и выбери уровень сложности из трех вариантов: низкая, средняя или высокая.",
        "Проанализируй сложность текста, расположенного ниже, и выбери один из трех вариантов: низкая, средняя или высокая.",
        "Определи сложность текста, представленного ниже, и выбери один из трех вариантов: низкая, средняя или высокая."
    ],
    "reply": [
        "Придумай подходящий ответ для продолжения дискуссии.",
        "Грамотно ответь на эту ветку комментариев.",
        "Ответь на следующие комментарии.",
        "Сформулируй свой ответ на комментарии, который следуют за этой инструкцией.",
        "Ответь на комментарии, которые будут представлены далее, используя свой опыт и знания."
    ],
    "comment": [
        "Напиши подходящий комментарий к статье.",
        "Сформулируй свой комментарий к статье.",
        "Создай свой комментарий к статье",
        "Прокомментируй статью",
        "Предложи свой комментарий к статье"
    ]
}

class InstructSet:
    def __init__(self, output_path):
        self.file = open(output_path, "w")

    def add(self, task, task_type, inputs, outputs, source):
        self.file.write(json.dumps({
            "task": task,
            "task_type": task_type,
            "inputs": inputs,
            "outputs": outputs,
            "source": source
        }, ensure_ascii=False).strip() + "\n")


def revert_flattening(records):
    fixed_records = []
    for key, values in records.items():
        if not fixed_records:
            fixed_records = [{} for _ in range(len(values))]
        for i, value in enumerate(values):
            fixed_records[i][key] = value
    return fixed_records


def convert_habr(archive):
    habr = load_dataset('IlyaGusev/habr', split="train", streaming=True)
    for row in tqdm(habr):
        if row["language"] != "ru":
            continue
        text = row["text_markdown"]
        if len(text) < 100:
            continue

        score = row["statistics"]["score"]

        source = "habr"
        if row["title"] and score >= 5:
            task_type = "headline"
            task = random.choice(TASK_INSTRUCTIONS[task_type])
            archive.add(
                task=task,
                task_type=task_type,
                inputs=text,
                outputs=row["title"],
                source=source
            )
        if row["tags"] and score >= 0:
            task_type = "keywords"
            task = random.choice(TASK_INSTRUCTIONS[task_type])
            archive.add(
                task=task,
                task_type=task_type,
                inputs=text,
                outputs=", ".join(row["tags"]),
                source=source
            )
        if row["complexity"]:
            task_type = "complexity"
            task = random.choice(TASK_INSTRUCTIONS[task_type])
            mapping = {
                "low": "низкая",
                "medium": "средняя",
                "high": "высокая"
            }
            archive.add(
                task=task,
                task_type=task_type,
                inputs=text,
                outputs=mapping[row["complexity"]],
                source=source
            )

        comments = revert_flattening(row["comments"])
        comments.sort(key=lambda x: x["time_published"])
        id2comment = {c["id"]: c for c in comments}

        for comment in comments:
            comment_text = comment["message_markdown"]
            score = comment["score"]
            if score < 5:
                continue
            if not comment_text:
                continue

            if not comment["parent_id"]:
                task_type = "comment"
                task = random.choice(TASK_INSTRUCTIONS[task_type])
                archive.add(
                    task=task,
                    task_type=task_type,
                    inputs=text,
                    outputs=comment_text,
                    source=source
                )
                continue

            branch = [comment_text]
            current_comment = comment
            while current_comment["parent_id"] and current_comment["parent_id"] in id2comment:
                current_comment = id2comment[current_comment["parent_id"]]
                branch.append(current_comment["message_markdown"])
            branch = branch[::-1]
            if len(branch) < 3:
                continue
            task_type = "reply"
            task = random.choice(TASK_INSTRUCTIONS[task_type])
            inputs = "Комментарий: " + "\nКомментарий: ".join(branch[:-1])
            if "UFO" in inputs:
                continue

            archive.add(
                task=task,
                task_type=task_type,
                inputs=inputs,
                outputs=branch[-1],
                source=source
            )


def main(output_path):
    archive = InstructSet(output_path)
    convert_habr(archive)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    main(**vars(args))

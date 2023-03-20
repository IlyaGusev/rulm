import argparse
import sys
import csv
import json
import gzip
import traceback
from datetime import datetime

from corus import load_buriy_news, load_lenta2, load_ods_tass, load_taiga_fontanka, load_taiga_fontanka_metas
from tqdm import tqdm

from data_processing.util import TextProcessor, read_jsonl


BAD_SUBSTRINGS = (
    "http",
    "{youtube}",
    "Показать комментарии",
    "//",
    "Видео No",
    "VIDEO",
    "!!!",
    "Пожалуйста, подождите",
    "обнаружена ошибка",
    "=",
    " ,",
    " :",
    " .",
    "О сериале",
    "О фильме",
    "Loading",
    "Читайте также:",
    "страница не найдена",
    "ПРАВИЛА КОММЕНТИРОВАНИЯ МАТЕРИАЛОВ",
    "незарегистрированный пользователь",
    "о регистрации СМИ",
    "читайте на сайте",
    "Комментарии премодерируются",
    "Читать далее",
    "используйте свой аккаунт",
    "Подробнее »",
    "Раздел:",
    "flickr",
    "AP Photo"
)

RM_SUBSTRINGS = (
    "Читать дальше >>",
    "Продолжение >>",
    "/ТАСС/",
    "Facebook Вконтакте Twitter Класс Google+",
    "Поделиться статьей в соц.сетях",
    "Внимание! Редакция может не разделять точку зрения авторов публикаций.",
    "Читайте нас в Фейсбуке, ВКонтакте, в Одноклассниках и в Твиттере",
    "Этот материал показался интересным?",
    "Поделиться:",
    "Написать комментарий",
    "Tweet Экспорт",
    "Поделись! Tweet",
    "Нашли опечатку? Выделите её мышкой и нажмите: Ctrl+Enter",
    "Если вы заметили ошибку в тексте, выделите текст с ошибкой и нажмите Ctrl+Enter",
    "Если вы заметили ошибку или опечатку в тексте, выделите ее курсором и нажмите Ctrl + Enter",
    "Мнение редакции \"Военного обозрения\" может не совпадать с точкой зрения авторов публикаций",
    "Поделиться Все по теме Комментарии",
    "MIGnews. com",
    "MIGnews.com"
)

BAN_HOSTS = (
    "sovsport.ru",
    "pravda.ru.feedsportal.com"
)


def main(
    buriy_files,
    fontanka_path,
    lenta_path,
    tass_path,
    telegram_path,
    output_path
):
    text_processor = TextProcessor(join_lines=False, min_chars=200)

    with open(output_path, "w") as w:
        for record in tqdm(read_jsonl(telegram_path)):
            text = text_processor(record["text"])
            if not text:
                continue
            has_bad_ss = any(ss in text for ss in BAD_SUBSTRINGS)
            if has_bad_ss:
                continue
            w.write(json.dumps({
                "title": record["title"],
                "text": record["text"],
                "url": record["url"],
                "timestamp": record["timestamp"],
                "source": "telegram_contest"
            }, ensure_ascii=False).strip() + "\n")

        for record in tqdm(load_ods_tass(tass_path)):
            text = record.text.replace(".n", ".\n")
            text = record.text.replace("!n", "!\n")
            text = record.text.replace("?n", "?\n")
            text = text_processor(text)
            if not text:
                continue
            has_bad_ss = any(ss in text for ss in BAD_SUBSTRINGS)
            if has_bad_ss:
                continue

            w.write(json.dumps({
                "title": record.title,
                "text": text,
                "url": record.url,
                "timestamp": int(datetime.timestamp(record.timestamp)),
                "source": "ods_tass"
            }, ensure_ascii=False).strip() + "\n")


        metas = load_taiga_fontanka_metas(fontanka_path)
        for record in tqdm(load_taiga_fontanka(fontanka_path, metas)):
            text = text_processor(record.text)
            if not text:
                continue
            has_bad_ss = any(ss in text for ss in BAD_SUBSTRINGS)
            if has_bad_ss:
                continue
            if not record.meta.timestamp:
                continue

            w.write(json.dumps({
                "title": record.meta.title,
                "text": text,
                "url": record.meta.url,
                "timestamp": int(datetime.timestamp(record.meta.timestamp)),
                "source": "taiga_fontanka"
            }, ensure_ascii=False).strip() + "\n")

        for path in buriy_files:
            for record in tqdm(load_buriy_news(path)):
                text = text_processor(record.text)
                if not text:
                    continue
                for ss in RM_SUBSTRINGS:
                    text = text.replace(ss, " ")
                text = " ".join(text.split(" ")).strip()

                is_bad_host = False
                for host in BAN_HOSTS:
                    if host in record.url:
                        is_bad_host = True
                if is_bad_host:
                    continue

                has_bad_ss = any(ss in text for ss in BAD_SUBSTRINGS)
                if has_bad_ss:
                    continue
                w.write(json.dumps({
                    "text": text,
                    "url": record.url,
                    "title": record.title,
                    "timestamp": int(datetime.timestamp(record.timestamp)),
                    "source": "buriy"
                }, ensure_ascii=False).strip() + "\n")

        for record in tqdm(load_lenta2(lenta_path)):
            if int(datetime.timestamp(record.date)) <= 0:
                continue
            text = text_processor(record.text)
            if not text:
                continue
            has_bad_ss = any(ss in text for ss in BAD_SUBSTRINGS)
            if has_bad_ss:
                continue

            w.write(json.dumps({
                "title": record.title,
                "text": text,
                "url": record.url,
                "timestamp": int(datetime.timestamp(record.date)),
                "source": "lenta"
            }, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--buriy-files', nargs='+', dest='buriy_files', type=str, required=True)
    parser.add_argument("--lenta-path", type=str, required=True)
    parser.add_argument("--fontanka-path", type=str, required=True)
    parser.add_argument("--tass-path", type=str, required=True)
    parser.add_argument("--telegram-path", type=str, required=True)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    main(**vars(args))

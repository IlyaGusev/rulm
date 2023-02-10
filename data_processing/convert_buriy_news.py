import argparse
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
    input_path,
    output_path
):
    text_processor = TextProcessor(join_lines=True, min_chars=200)
    archive = PlainArchive(output_path)

    for record in tqdm(load_buriy_news(input_path)):
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
        archive.add_data(
            text=text,
            meta={
                "source": "buriy_news",
                "url": record.url,
                "title": record.title,
                "timestamp": int(datetime.timestamp(record.timestamp))
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    main(**vars(args))

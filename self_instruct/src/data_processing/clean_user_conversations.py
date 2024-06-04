import fire

from src.data_processing.bad_substrings import has_bad_ss
from src.util.io import read_jsonl, write_jsonl

ADDITONAL_BAD_SS = (
    "извин",
    "база знаний",
    "содержание",
    "обсудим что-нибудь другое",
    "неподобающ",
    "незакон",
    "цензурн",
    "приличн",
    "такого рода",
    "этого рода",
    "менять тему",
    "другие темы",
    "таким запрос",
    "этим запрос",
    "таких разговорах",
    "о чем-то другом",
    "о чём-нибудь другом",
    "такие темы",
    "не могу обсуждать",
    "уважит",
    "не могу продолжать",
    "данный разговор",
    "соответствует правилам",
    "ограничивается",
    "противоречит политике",
    "правила платформы",
    "безопасность платформы",
    "на момент",
    "на данный момент",
    "информация ограничена",
    "актуальны на",
    "на данных до",
    "извини, но я",
    "извините, но я"
)


def set_regex_flag(records):
    new_records = []
    for record in records:
        messages = record["messages"]
        user_messages = [m for m in messages if m["role"] == "user"]
        bot_messages = [m for m in messages if m["role"] in ("assistant", "bot")]
        record["is_bad_by_regex"] = False
        if has_bad_ss(bot_messages):
            record["is_bad_by_regex"] = True
        if any([any(ss in m["content"].lower() for ss in ADDITONAL_BAD_SS) for m in bot_messages]):
            record["is_bad_by_regex"] = True
        new_records.append(record)
    return new_records


def main(input_path, output_path):
    records = list(read_jsonl(input_path))
    new_records = set_regex_flag(records)
    print(len(records))
    print(len(new_records))
    print(sum([r["is_bad_by_regex"] is False for r in new_records]))
    write_jsonl(new_records, output_path)


if __name__ == "__main__":
    fire.Fire(main)

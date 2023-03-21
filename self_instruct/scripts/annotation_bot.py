import argparse
import random
import json
from collections import defaultdict

from tinydb import TinyDB, where
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Updater, CommandHandler, Filters, CallbackContext, MessageHandler, CallbackQueryHandler


class Client:
    def __init__(self, token, db_path, input_path):
        self.db = TinyDB(db_path, ensure_ascii=False)
        self.input_path = input_path

        self.updater = Updater(token=token, use_context=True)
        self.updater.dispatcher.add_handler(CommandHandler("start", self.start, filters=Filters.command))
        self.updater.dispatcher.add_handler(CallbackQueryHandler(self.button))

        with open(input_path, "r") as r:
            self.records = json.load(r)
            random.shuffle(self.records)

        self.last_records = defaultdict(None)
        self.chat2username = dict()
        print("Bot is ready!")

    def write_result(self, result, chat_id):
        username = self.chat2username.get(chat_id)
        last_record = self.last_records[username]
        if not last_record:
            return

        last_record["label"] = result
        last_record["username"] = username
        last_record["chat_id"] = chat_id
        self.db.insert(last_record)

    def run(self):
        self.updater.start_polling()
        self.updater.idle()

    def start(self, update: Update, context: CallbackContext):
        self.show(update, context)

    def button(self, update: Update, context: CallbackContext) -> None:
        query = update.callback_query
        query.answer()

        data = query.data
        chat_id = update.effective_chat.id

        self.write_result(data, chat_id)
        self.show(update, context)

    def sample_record(self, retries=50):
        for _ in range(retries):
            record = random.choice(self.records)
            if not self.db.contains(where("instruction") == record["instruction"]):
                break
        return record

    def show(self, update: Update, context: CallbackContext):
        chat_id = update.effective_chat.id
        if update.message:
            username = update.message.chat.username
            self.chat2username[chat_id] = username
        else:
            username = self.chat2username[chat_id]

        record = self.sample_record()
        self.last_records[username] = record
        text = f"Задание: {record['instruction']}\n\n"
        if record["input"].strip() and record["input"].strip() != "<noinput>":
            text += f"Вход: {record['input']}\n\n"
        text += f"Выход: {record['output']}"

        keyboard = [
            [
                InlineKeyboardButton("Ошибка в ответе", callback_data="ok"),
                InlineKeyboardButton("Ошибка в задании/входе", callback_data="bad")
            ],
            [
                InlineKeyboardButton("Всё идеально", callback_data="all_ok")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        context.bot.send_message(
            text=text,
            reply_markup=reply_markup,
            parse_mode="Markdown",
            chat_id=chat_id
        )


def main(
    token,
    input_path,
    db_path,
    seed
):
    random.seed(seed)
    client = Client(
        token=token,
        db_path=db_path,
        input_path=input_path
    )
    client.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--db-path", type=str, default="db.json")
    parser.add_argument("--input-path", type=str, default="output.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(**vars(args))

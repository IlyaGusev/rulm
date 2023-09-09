import argparse
import random
import json
from collections import defaultdict

from tinydb import TinyDB, where
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Updater, CommandHandler, Filters, CallbackContext, CallbackQueryHandler


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
        if result == "skip":
            return True

        username = self.chat2username.get(chat_id)
        last_record = self.last_records.get(chat_id)
        if not last_record:
            return False

        last_record["label"] = result
        last_record["username"] = username
        last_record["chat_id"] = chat_id
        self.db.insert(last_record)
        return True

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

        if self.write_result(data, chat_id):
            self.show(update, context)
        else:
            context.bot.send_message(text="Нужно перезапустить бот через '/start'", chat_id=chat_id)

    def sample_record(self, username, retries=50, max_overlap=3):
        for _ in range(retries):
            record = random.choice(self.records)
            instruction = record["instruction"]
            count = self.db.count(where("instruction") == instruction)
            if count >= max_overlap:
                continue
            if not self.db.contains((where("instruction") == instruction) & (where("username") == username)):
                break
        return record

    def show(self, update: Update, context: CallbackContext):
        chat_id = update.effective_chat.id
        if update.message:
            username = update.message.chat.username
            self.chat2username[chat_id] = username
        else:
            username = self.chat2username[chat_id]

        record = self.sample_record(username)
        self.last_records[chat_id] = record
        text = f"Задание: {record['instruction']}\n\n"
        if record["input"].strip() and record["input"].strip() != "<noinput>":
            text += f"Вход: {record['input']}\n\n"
        text += f"Ответ: {record['output']}"

        keyboard = [
            [
                InlineKeyboardButton("Плохой ответ", callback_data="ok"),
                InlineKeyboardButton("Плохое задание или вход", callback_data="bad")
            ],
            [
                InlineKeyboardButton("Всё идеально", callback_data="all_ok"),
                InlineKeyboardButton("Пропустить", callback_data="skip")
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

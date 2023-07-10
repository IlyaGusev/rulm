import json

DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>\n"
DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
DEFAULT_START_TOKEN_ID = 1
DEFAULT_END_TOKEN_ID = 2
DEFAULT_BOT_TOKEN_ID = 9225


class Conversation:
    def __init__(
        self,
        message_template=DEFAULT_MESSAGE_TEMPLATE,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        role_mapping=None,
        start_token_id=DEFAULT_START_TOKEN_ID,
        end_token_id=DEFAULT_END_TOKEN_ID,
        bot_token_id=DEFAULT_BOT_TOKEN_ID
    ):
        self.message_template = message_template
        self.role_mapping = role_mapping or {}
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.bot_token_id = bot_token_id
        self.messages = [{
            "role": "system",
            "content": system_prompt
        }]

    def get_end_token_id(self):
        return self.end_token_id

    def get_start_token_id(self):
        return self.start_token_id

    def get_bot_token_id(self):
        return self.bot_token_id

    def add_user_message(self, message):
        self.messages.append({
            "role": "user",
            "content": message
        })

    def add_bot_message(self, message):
        self.messages.append({
            "role": "bot",
            "content": message
        })

    def count_tokens(self, tokenizer, messages):
        final_text = ""
        for message in messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        tokens = tokenizer([final_text])["input_ids"][0]
        return len(tokens)

    def shrink(self, tokenizer, messages, max_tokens):
        system_message = messages[0]
        other_messages = messages[1:]
        while self.count_tokens(tokenizer, [system_message] + other_messages) > max_tokens:
            other_messages = other_messages[2:]
        return [system_message] + other_messages

    def get_prompt(self, tokenizer, max_tokens: int = None, add_suffix: bool = True):
        final_text = ""
        messages = self.messages
        if max_tokens is not None:
            messages = self.shrink(tokenizer, messages, max_tokens)

        for message in messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        if add_suffix:
            final_text += tokenizer.decode([self.start_token_id, self.bot_token_id])
        return final_text.strip()

    @classmethod
    def from_template(cls, file_name):
        with open(file_name, encoding="utf-8") as r:
            template = json.load(r)
        return Conversation(
            **template
        )

    def expand(self, messages):
        if messages[0]["role"] == "system":
            self.messages = []

        for message in messages:
            self.messages.append({
                "role": self.role_mapping.get(message["role"], message["role"]),
                "content": message["content"]
            })

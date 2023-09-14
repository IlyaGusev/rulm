import fire

from src.util.chat import Conversation
from src.util.load import load_saiga
from src.util.generate import generate


def interact(model_name, template_path):
    model, tokenizer, generation_config = load_saiga(model_name)
    conversation = Conversation.from_template(template_path)
    while True:
        user_message = input("User: ")
        if user_message.strip() == "/reset":
            conversation = Conversation.from_template(template_path)
            print("History reset completed!")
            continue
        conversation.add_user_message(user_message)
        prompt = conversation.get_prompt(tokenizer)
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompts=[prompt],
            generation_config=generation_config
        )[0]
        conversation.add_bot_message(output)
        print("Saiga:", output)


if __name__ == "__main__":
    fire.Fire(interact)

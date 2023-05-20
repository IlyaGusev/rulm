import fire
from llama_cpp import Llama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from chromadb.config import Settings

SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
SYSTEM_TOKEN = 1788
USER_TOKEN = 1404
BOT_TOKEN = 9225
LINEBREAK_TOKEN = 13

ROLE_TOKENS = {
    "user": USER_TOKEN,
    "bot": BOT_TOKEN,
    "system": SYSTEM_TOKEN
}


def get_message_tokens(model, role, content):
    message_tokens = model.tokenize(content.encode("utf-8"))
    message_tokens.insert(1, ROLE_TOKENS[role])
    message_tokens.insert(2, LINEBREAK_TOKEN)
    message_tokens.append(model.token_eos())
    return message_tokens


def get_system_tokens(model):
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
    return get_message_tokens(model, **system_message)


def get_chroma_settings(output_dir):
    return Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=output_dir,
        anonymized_telemetry=False
    )


def interact(
    model_path,
    index_path,
    embeddings_model_path,
    n_ctx=2000,
    top_k=30,
    top_p=0.9,
    temperature=0.2,
    repeat_penalty=1.1
):
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_path)
    db = Chroma(persist_directory=index_path, embedding_function=embeddings, client_settings=get_chroma_settings(index_path))
    retriever = db.as_retriever(search_kwargs={"k": 2})

    model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_parts=1,
    )

    system_tokens = get_system_tokens(model)
    tokens = system_tokens
    model.eval(tokens)

    while True:
        user_message = input("User: ")

        docs = retriever.get_relevant_documents(user_message)
        context = "\n\n".join([doc.page_content for doc in docs])
        print()
        print("Документы:", context)
        print()
        user_message = f"""
        Контекст: {context}

        Вопрос: {user_message}
        """

        message_tokens = get_message_tokens(model=model, role="user", content=user_message)
        role_tokens = [model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
        tokens += message_tokens + role_tokens
        generator = model.generate(
            tokens,
            top_k=top_k,
            top_p=top_p,
            temp=temperature,
            repeat_penalty=repeat_penalty
        )
        print("Saiga: ", end="", flush=True)
        for token in generator:
            token_str = model.detokenize([token]).decode("utf-8", errors="ignore")
            tokens.append(token)
            if token == model.token_eos():
                break
            print(token_str, end="", flush=True)
        print()


if __name__ == "__main__":
    fire.Fire(interact)

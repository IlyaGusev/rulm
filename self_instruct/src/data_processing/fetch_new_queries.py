import json

import fire
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.data_processing.lang_detector import FasttextLanguageDetector
from src.data_processing.embedder import Embedder


def main(existing_path, output_path, langdetect_threshold: float = 0.8, sim_threshold: float = 0.93):
    lang_detector = FasttextLanguageDetector()
    embedder = Embedder("intfloat/multilingual-e5-base")

    existing_quries = list()
    embeddings = []
    with open(existing_path) as r:
        for line in r:
            query = json.loads(line)["prompt"].strip().lower()
            existing_quries.append(query)

    for row in load_dataset("saiga_scored", split="train"):
        query = row["messages"][0]["content"].strip().lower()
        existing_quries.append(query)

    embeddings = embedder(existing_quries).tolist()
    existing_quries_set = set(existing_quries)

    with open(output_path, "w") as w:
        for row in tqdm(load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)):
            if row["language"] != "Russian":
                continue
            orig_query = row["conversation"][0]["content"].strip()
            query = orig_query.lower()
            if query in existing_quries_set:
                continue
            if sum(row["openai_moderation"][0]["categories"].values()) >= 1:
                continue
            language, score = lang_detector(" ".join(query.split()[:6]))
            if language != "ru" or score < langdetect_threshold:
                continue
            embedding = embedder([query])[0]
            scores = cosine_similarity([embedding], embeddings)[0]
            scores = [(score, idx) for idx, score in enumerate(scores)]
            max_score, max_idx = max(scores)
            if max_score > sim_threshold:
                old_query = existing_quries[max_idx]
                print("SKIP!", max_score, query.replace("\n", " ")[:40], "####", old_query.replace("\n", " ")[:40])
                continue
            embeddings.append(embedding)
            existing_quries.append(query)
            existing_quries_set.add(query)
            w.write(json.dumps({"prompt": orig_query}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(main)

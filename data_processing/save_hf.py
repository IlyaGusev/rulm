import json
import sys

from datasets import load_dataset
from tqdm import tqdm

from data_processing.util import TextProcessor, PlainArchive

output_path = sys.argv[1]

archive = PlainArchive(output_path)
processor = TextProcessor()
gazeta = load_dataset('IlyaGusev/gazeta', revision="v2.0")["train"]
for row in tqdm(gazeta):
    text = processor(row["text"])
    if not text:
        continue
    archive.add_data(
        text=processor(row["text"]),
        meta={
            "source": "gazeta",
            "title": row["title"],
            "date": row["date"],
            "url": row["url"]
        }
    )

processor = TextProcessor(join_lines=True)
medical_qa = load_dataset("blinoff/medical_qa_ru_data")["train"]
for row in tqdm(medical_qa):
    text = "Вопрос: " + row["desc"]
    for i, answer in enumerate(row["ans"].split(";\n")):
        text += f"; Ответ {i+1}: {answer}"
    text = processor(text)
    if not text:
        continue
    archive.add_data(
        text=text,
        meta={
            "source": "medical_qa",
            "theme": row["theme"],
            "date": row["date"],
            "categ": row["categ"],
            "spec10": row["spec10"]
        }
    )

processor = TextProcessor(join_lines=True)
sentiment = load_dataset("Tatyana/ru_sentiment_dataset")["train"]
labels = {
    0: "NEUTRAL",
    1: "POSITIVE",
    2: "NEGATIVE"
}
for row in tqdm(sentiment):
    text = processor(row["text"])
    if not text:
        continue
    archive.add_data(
        text=text,
        meta={
            "source": "sentiment",
            "sentiment": labels[int(row["sentiment"])]
        }
    )

#cc100 = load_dataset("cc100", lang="ru")
#for row in cc100:
#    print(row)
#    break

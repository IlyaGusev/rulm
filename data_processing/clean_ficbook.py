import fire
import random

import pandas as pd
from tqdm import tqdm
from data_processing.util import TextProcessor, read_jsonl

def main(input_path, output_path):
    processor = TextProcessor(
        normalization="NFKC",
        min_chars=0,
        min_text_part=0,
        fix_punct=False,
        fix_spaces=True,
        fix_short_lines=True,
        check_languages=False,
        check_pii=False,
        check_code=False,
        check_links=False,
        check_email=False,
        check_text_part=False
    )

    records = []
    for record in tqdm(read_jsonl(input_path)):
        for part in record["parts"]:
            text = part["text"]
            clean_text = processor(text)
            assert clean_text
            part["clean_text"] = clean_text
        records.append(record)
    random.shuffle(records)
    pd.DataFrame(records).to_parquet(output_path)

if __name__ == "__main__":
    fire.Fire(main)


import os
import fire
import random

import dask.dataframe as da
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
    chunk_num = 0
    for record in tqdm(read_jsonl(input_path)):
        is_broken = False
        if not record.get("pairing", ""):
            record["pairing"] = ""
        for part in record["parts"]:
            text = part["text"]
            clean_text = processor(text)
            if not clean_text:
                is_broken = True
            part["clean_text"] = clean_text
        if not is_broken:
            records.append(record)
        if len(records) == 20000:
            random.shuffle(records)
            pd.DataFrame(records).to_parquet(os.path.join(output_path, f"{chunk_num:04d}.parquet"))
            records = []
            chunk_num += 1

    if records:
        pd.DataFrame(records).to_parquet(os.path.join(output_path, f"{chunk_num:04d}.parquet"))

if __name__ == "__main__":
    fire.Fire(main)


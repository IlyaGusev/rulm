import json
import os
from zipfile import ZipFile

from data_processing.parse_fb2 import FB2Parser
from data_processing.util import TextProcessor

import fire
from tqdm import tqdm



def main(input_dir, output_dir):
    parser = FB2Parser()
    processor = TextProcessor(
        min_chars=3,
        min_text_part=0.0,
        fix_punct=False,
        fix_spaces=True,
        fix_short_lines=True,
        check_code=False,
        check_pii=False,
        check_links=False,
        check_languages=False,
        check_email=False,
        check_text_part=False
    )
    existing_files = os.listdir(output_dir)
    for zip_name in os.listdir(input_dir):
        print(f"Parsing {zip_name}")
        if not zip_name.endswith(".zip"):
            continue
        start_id = int(zip_name.split(".")[0].split("-")[1])
        end_id = zip_name.split(".")[0].split("-")[2]
        is_lost = "lost" in end_id
        end_id = int(end_id.replace("_lost", ""))
        if end_id <= 500000:
            continue
        found = False
        for file_name in existing_files:
            if str(start_id) in file_name and str(end_id) in file_name:
                found = True
        if found:
            continue
        input_file = os.path.join(input_dir, zip_name)
        output_file = os.path.join(output_dir, zip_name.replace(".zip", ".jsonl"))
        with ZipFile(input_file, "r") as archive, open(output_file, "w") as w:
            names = [name for name in archive.namelist() if name.endswith(".fb2")]
            for name in tqdm(names):
                if not name.endswith(".fb2"):
                    continue
                content = archive.read(name)
                record = parser(content)
                if record is None:
                    print(f"{name} is ill-formed!")
                    continue
                record["file_name"] = name
                clean_sections = []
                for section in record["sections"]:
                    clean_section = processor(section)
                    if clean_section:
                        clean_sections.append(clean_section)
                record["sections"] = clean_sections
                w.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(main)

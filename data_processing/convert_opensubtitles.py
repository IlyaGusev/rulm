import sys
import zipfile
import os
import json
from collections import Counter

from tqdm import tqdm
from bs4 import BeautifulSoup

from data_processing.util import gen_batch


input_path = sys.argv[1]
output_path = sys.argv[2]


def get_xml_filepaths_from_zip(archive):
    xml_files = []
    for file_path in archive.namelist():
        if file_path.endswith('.xml'):
            xml_files.append(file_path)
    return xml_files


def parse_single_xml(path_to_xml, zip_file):
    xml = zip_file.open(path_to_xml)
    soup = BeautifulSoup(xml, 'html.parser')
    subtitles = soup("s")
    subs = []
    for each in subtitles:
        l = each.get_text()
        l = l.replace('\n', '')
        l = l.replace('\t', '').strip().strip('-').strip('/').strip() 
        subs.append(l)
    seperator = '\n'
    texts = seperator.join(subs)
    return "\n".join([s for s in texts.split("\n") if s])


BAD_SUBSTRINGS = (
    "ƒ",
    "ћ",
    "Ќ",
    "fad(",
    "{\\",
    "≈",
    "---",
    "ПРО СЕБЯ",
    "==",
    ")}",
    "00:",
    "Untranslated",
    "1080",
    "720",
    "~"
)

if __name__ == "__main__":
    zip_file = zipfile.ZipFile(input_path)
    xml_paths = get_xml_filepaths_from_zip(zip_file)
    with open(output_path, "w") as w:
        bad_cnt = Counter()
        for path in tqdm(xml_paths, total=len(xml_paths)):
            full_text = parse_single_xml(path, zip_file)
            lines = full_text.split("\n")
            for batch_num, batch in enumerate(gen_batch(lines, 1000)):
                batch = [line for line in batch if len(line) > 3]
                text = "\n".join(batch)
                for ss in BAD_SUBSTRINGS:
                    if ss in text:
                        bad_cnt[ss] += 1
                has_bad_ss = any(ss in text for ss in BAD_SUBSTRINGS)
                if has_bad_ss:
                    continue
                w.write(json.dumps({
                    "text": text,
                    "meta": {
                        "path": path,
                        "part_num": batch_num
                    }
                }, ensure_ascii=False) + "\n")
        print(bad_cnt.most_common())

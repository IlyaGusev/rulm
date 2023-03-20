#!/bin/bash
set -euo pipefail

OUTPUT_DIR=/data/corpora
CACHE_DIR=/data/cache

rm -f $OUTPUT_DIR/ruwiki-latest-pages-articles.xml.bz2
rm -f $OUTPUT_DIR/dm_math_ru.zip

wget https://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-pages-articles.xml.bz2 -O $OUTPUT_DIR/ruwiki-latest-pages-articles.xml.bz2
wget https://www.dropbox.com/s/h8d47dhaka3xn9i/dm_math_ru.zip -O $OUTPUT_DIR/dm_math_ru.zip

python3 -m data_processing.convert_wiki $OUTPUT_DIR/ruwiki-latest-pages-articles.xml.bz2 $OUTPUT_DIR/ruwiki.jsonl
python3 -m data_processing.convert_math $OUTPUT_DIR/dm_math_ru.zip $OUTPUT_DIR/math.jsonl
HF_DATASETS_CACHE=$CACHE_DIR python3 -m data_processing.save_hf $OUTPUT_DIR/hf.jsonl

python3 -m data_processing.merge --output-path $OUTPUT_DIR/merged.jsonl -f $OUTPUT_DIR/ruwiki.jsonl $OUTPUT_DIR/hf.jsonl
HF_DATASETS_CACHE=$CACHE_DIR python3 -m data_processing.undup $OUTPUT_DIR/merged.jsonl $OUTPUT_DIR/merged_undup.jsonl
mv $OUTPUT_DIR/merged_undup.jsonl $OUTPUT_DIR/merged.jsonl

python3 -m data_processing.merge --output-path $OUTPUT_DIR/merged_math.jsonl -f $OUTPUT_DIR/merged.jsonl $OUTPUT_DIR/math.jsonl
mv $OUTPUT_DIR/merged_math.jsonl $OUTPUT_DIR/merged.jsonl

sort /data/corpora/merged.jsonl -S 50% --random-sort > $OUTPUT_DIR/merged_shuf.jsonl
python3 -m data_processing.split --input-path $OUTPUT_DIR/merged_shuf.jsonl --train-path $OUTPUT_DIR/train.jsonl --validation-path $OUTPUT_DIR/validations.jsonl --test-path $OUTPUT_DIR/test.jsonl

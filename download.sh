OUTPUT_DIR=/data/corpora

wget https://linghub.ru/static/Taiga/retagged_taiga.tar.gz -O $OUTPUT_DIR/retagged_taiga.tar.gz
wget http://panchenko.me/data/russe/librusec_fb2.plain.gz -O $OUTPUT_DIR/librusec_fb2.plain.gz
wget https://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-pages-articles.xml.bz2 -O $OUTPUT_DIR/ruwiki-latest-pages-articles.xml.bz2

cd /data/corpora && tar -xzvf retagged_taiga.tar.gz
cd /data/corpora && gunzip librusec_fb2.plain.gz

python3 -m converters.convert_wiki /data/corpora/ruwiki-latest-pages-articles.xml.bz2 /data/corpora/ruwiki.jsonl
python3 -m converters.convert_stihi /data/corpora/stihi_ru.zip /data/corpora/stihi_ru.jsonl
python3 -m converters.convert_librusec /data/corpora/librusec_fb2.plain /data/corpora/librusec.jsonl

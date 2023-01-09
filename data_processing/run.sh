OUTPUT_DIR=/data/corpora

wget https://linghub.ru/static/Taiga/retagged_taiga.tar.gz -O $OUTPUT_DIR/retagged_taiga.tar.gz
wget http://panchenko.me/data/russe/librusec_fb2.plain.gz -O $OUTPUT_DIR/librusec_fb2.plain.gz
wget https://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-pages-articles.xml.bz2 -O $OUTPUT_DIR/ruwiki-latest-pages-articles.xml.bz2
wget https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/raw/ru.zip -O $OUTPUT_DIR/open_subtitles_ru.zip
wget https://www.dropbox.com/s/h8d47dhaka3xn9i/dm_math_ru.zip -O $OUTPUT_DIR/dm_math_ru.zip
wget https://github.com/buriy/russian-nlp-datasets/releases/download/r4/news-articles-2014.tar.bz2 -O $OUTPUT_DIR/buriy_news_2014.tar.bz2
wget https://github.com/buriy/russian-nlp-datasets/releases/download/r4/news-articles-2015-part1.tar.bz2 -O $OUTPUT_DIR/buriy_news_2015_part1.tar.bz2
wget https://github.com/buriy/russian-nlp-datasets/releases/download/r4/news-articles-2015-part2.tar.bz2 -O $OUTPUT_DIR/buriy_news_2015_part2.tar.bz2
wget https://ia600107.us.archive.org/27/items/stackexchange/ru.stackoverflow.com.7z -O $OUTPUT_DIR/ru_stackoverflow.7z

cd $OUTPUT_DIR && tar -xzvf retagged_taiga.tar.gz && rm retagged_taiga.tar.gz
cd $OUTPUT_DIR && gunzip librusec_fb2.plain.gz
cd $OUTPUT_DIR && 7z x ru_stackoverflow.7z

python3 -m data_processing.convert_wiki $OUTPUT_DIR/ruwiki-latest-pages-articles.xml.bz2 $OUTPUT_DIR/ruwiki.jsonl
python3 -m data_processing.convert_stihi $OUTPUT_DIR/stihi_ru.zip $OUTPUT_DIR/stihi_ru.jsonl
python3 -m data_processing.convert_librusec $OUTPUT_DIR/librusec_fb2.plain $OUTPUT_DIR/librusec.jsonl
python3 -m data_processing.convert_opensubtitles $OUTPUT_DIR/open_subtitles_ru.zip $OUTPUT_DIR/opensubtitles.jsonl
python3 -m data_processing.convert_math $OUTPUT_DIR/dm_math_ru.zip $OUTPUT_DIR/math.jsonl
python3 -m data_processing.convert_buriy_news $OUTPUT_DIR/buriy_news_2014.tar.bz2 $OUTPUT_DIR/buriy_news_1.jsonl
python3 -m data_processing.convert_buriy_news $OUTPUT_DIR/buriy_news_2015_part1.tar.bz2 $OUTPUT_DIR/buriy_news_2.jsonl
python3 -m data_processing.convert_buriy_news $OUTPUT_DIR/buriy_news_2015_part2.tar.bz2 $OUTPUT_DIR/buriy_news_3.jsonl
python3 -m data_processing.save_hf $OUTPUT_DIR/hf.jsonl

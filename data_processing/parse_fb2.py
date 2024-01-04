# Based on:
# https://github.com/gribuser/fb2/blob/master/FictionBook.xsd
# https://github.com/mgrankin/ru_transformers/blob/master/corpus/FB2_2_txt.xsl
# Problems:
# - no tables
# - no sequences
# - no notes
# - no doucment-info
# - no src-title-info
# - some other fields are omitted

import json
import sys
import os
import xml.etree.ElementTree as etree

import fire

NS = {"fb": "http://www.gribuser.ru/xml/fictionbook/2.0"}


class FB2Parser:
    def __init__(self):
        self.publish_keys = (
            "isbn",
            "publisher",
            "city",
            "year",
            "book-name",
        )
        self.section_skip_tags = (
             "cite",
             "table",
             "title",
             "epigraph",
             "image",
             "annotation"
        )

    def __call__(self, content):
        try:
            root = etree.fromstring(content)
        except etree.ParseError as e:
            return None
        assert root is not None
        assert "FictionBook" in root.tag

        description = root.find("./fb:description", NS)
        if description is None:
            return None

        title_info = description.find("./fb:title-info", NS)
        assert title_info is not None
        title_info_data = self.parse_title_info(title_info)
        publish_info = description.find("./fb:publish-info", NS)
        publish_info_data = self.parse_publish_info(publish_info)

        body = root.find("./fb:body", NS)
        if body is None:
            return None

        body_data = self.parse_body(body)

        return {
            **title_info_data,
            **publish_info_data,
            **body_data,
        }

    def parse_body(self, body):
        fancy_title = body.find("./fb:title", NS)
        fancy_title_str = self.parse_content(fancy_title) if fancy_title is not None else None

        epigraphs = []
        for epigraph in body.findall("./fb:epigraph", NS):
            epigraphs.append(self.parse_content(epigraph))

        sections = []
        for section in body.findall("./fb:section", NS):
            section_str = self.parse_section(section)
            if section_str:
                sections.append(section_str)

        return {
            "fancy_title": fancy_title_str,
            "epigraphs": epigraphs,
            "sections": sections
        }

    def parse_content(self, title):
        # titleType/epigraphType/annotationType
        # https://github.com/gribuser/fb2/blob/master/FictionBook.xsd#L273
        lines = []
        for p in title.findall("./fb:p", NS):
            lines.append(self.parse_p(p))
        return "\n".join(lines).strip()

    def parse_section(self, section):
        # sectionType
        # https://github.com/gribuser/fb2/blob/master/FictionBook.xsd#L396
        title = section.find("./fb:title", NS)
        title_str = self.parse_content(title) if title is not None else None
        epigraph = section.find("./fb:epigraph", NS)
        epigraph_str = self.parse_content(epigraph) if epigraph is not None else None
        annotation = section.find("./fb:annotation", NS)
        annotation_str = self.parse_content(annotation) if annotation is not None else None

        section_paragraphs = []
        for elem in section:
            if any(tag in elem.tag for tag in self.section_skip_tags):
                continue

            if "empty-line" in elem.tag:
                section_paragraphs.append("\n")
                continue

            if "poem" in elem.tag:
                section_paragraphs.append(self.parse_poem(elem))
                continue

            # Recursive subsections
            if "section" in elem.tag:
                section_paragraphs.append(self.parse_section(elem))
                continue

            section_paragraphs.append(self.parse_p(elem))

        section_str = "\n".join(section_paragraphs)
        if epigraph_str:
            section_str = epigraph_str + "\n\n" + section_str
        if title_str:
            section_str = title_str + "\n\n" + section_str
        if annotation_str:
            section_str = annotation_str + "\n\n" + section_str
        return section_str

    def parse_p(self, p):
        text = "".join(p.itertext()).strip()
        text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
        return text

    def unpack_text(self, elem):
        text = elem.text if elem is not None else ""
        text = text if text is not None else ""
        return text.strip()

    def parse_title_info(self, title_info):
        # title-infoType
        # https://github.com/gribuser/fb2/blob/master/FictionBook.xsd#L570
        # Example:
        # <title-info>
        #   <genre>detective</genre>
        #   <author>
        #     <first-name>неизвестный</first-name>
        #     <last-name>Автор</last-name>
        #   </author>
        #   <book-title>Дрейк</book-title>
        #   <lang>ru</lang>
        # </title-info>

        title = self.unpack_text(title_info.find("./fb:book-title", NS))
        genre = self.unpack_text(title_info.find("./fb:genre", NS))
        keywords = self.unpack_text(title_info.find("./fb:keywords", NS))
        lang = self.unpack_text(title_info.find("./fb:lang", NS))
        src_lang = self.unpack_text(title_info.find("./fb:src-lang", NS))
        date = self.unpack_text(title_info.find("./fb:date", NS))

        translator = title_info.find("./fb:translator", NS)
        translator = self.parse_author(translator) if translator is not None else None
        translator = "" if translator is None else self.author_to_str(translator)

        annotation = title_info.find("./fb:annotation", NS)
        annotation = self.parse_content(annotation) if annotation is not None else None
        annotation = "" if annotation is None else annotation

        authors = []
        for author in title_info.findall("./fb:author", NS):
            authors.append(self.author_to_str(self.parse_author(author)))
        return {
            "title": title,
            "annotation": annotation,
            "keywords": keywords,
            "date": date,
            "genre": genre,
            "authors": authors,
            "lang": lang,
            "src_lang": src_lang,
            "translator": translator
        }

    def parse_author(self, author):
        # authorType
        # https://github.com/gribuser/fb2/blob/master/FictionBook.xsd#L233
        last_name = self.unpack_text(author.find("./fb:last-name", NS))
        first_name = self.unpack_text(author.find("./fb:first-name", NS))
        middle_name = self.unpack_text(author.find("./fb:middle-name", NS))
        nickname = self.unpack_text(author.find("./fb:nickname", NS))
        return {
            "last_name": last_name,
            "first_name": first_name,
            "middle_name": middle_name,
            "nickname": nickname
        }

    def author_to_str(self, author):
        parts = []
        if author["last_name"]:
            parts.append(author["last_name"])
        if author["first_name"]:
            parts.append(author["first_name"])
        if author["middle_name"]:
            parts.append(author["middle_name"])
        if author["nickname"]:
            parts.append('(' + author["nickname"] + ')')
        return " ".join(parts)

    def parse_publish_info(self, publish_info):
        # https://github.com/gribuser/fb2/blob/master/FictionBook.xsd#L156
        # Example:
        # <publish-info>
        #   <publisher>Twelve</publisher>
        #   <year>2008</year>
        #   <isbn>9780446511070</isbn>
        # </publish-info>

        result = {key.replace("-", "_"): "" for key in self.publish_keys}
        if publish_info is None:
            return result

        for key in self.publish_keys:
            value = publish_info.find(f"./fb:{key}", NS)
            value = value.text if value is not None else ""
            result[key.replace("-", "_")] = value

        return result

    def parse_poem(self, poem):
        # poemType
        # https://github.com/gribuser/fb2/blob/master/FictionBook.xsd#L321
        title = poem.find("./fb:title", NS)
        title_str = self.parse_content(title) if title is not None else None

        subtitle = poem.find("./fb:subtitle", NS)
        subtitle_str = self.parse_p(subtitle) if subtitle is not None else None

        poem_lines = []
        for stanza in poem.findall("./fb:stanza", NS):
            for elem in stanza.findall("./fb:v", NS):
                poem_lines.append(self.parse_p(elem))
            poem_lines.append("\n")

        poem_str = "\n".join(poem_lines)
        if subtitle_str:
            poem_str = subtitle_str + "\n\n" + poem_str
        if title_str:
            poem_str = title_str + "\n\n" + poem_str
        return poem_str.strip()


def main(input_file):
    parser = FB2Parser()
    with open(input_file, encoding="windows-1251") as r:
        output = parser(r.read())
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    fire.Fire(main)

import os
import urllib
import urllib.request
import traceback
from urllib.parse import urlparse
from urllib.error import HTTPError
from typing import List, Dict
import time
import json
from dataclasses import dataclass, asdict, field

import fire
from bs4 import BeautifulSoup

HEADERS = {'User-Agent':'Mozilla/5.0'}


@dataclass
class Fic:
    url: str = ""
    authors: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    description: str = ""
    likes: int = 0
    parts: List[Dict[str, str]] = field(default_factory=list)
    part_count: int = 0
    title: str = ""
    rating: str = ""
    status: str = ""
    direction: str = ""
    category: str = ""
    pairing: str = ""


def clean_text(text):
    text = text.replace("\r\n", "\n")
    text = text.strip()
    return text


def parse_part(soup):
    elem = soup.find("div", {"itemprop": "articleBody"})
    if elem is None:
        return None
    part_text = clean_text(elem.text)

    part_date = ""
    elem = soup.find("div", {"class": "part-date"})
    if elem is not None:
        part_date = elem["content"]

    part_title = ""
    elem = soup.find("article")
    if elem is not None:
        elem = elem.find("h2")
        if elem is not None:
            part_title = elem.text

    return {
        "title": part_title.strip(),
        "text": part_text.strip(),
        "date": part_date.strip()
    }


def get_fic(url, sleep_time):
    parsed_uri = urlparse(url)
    host = "{uri.scheme}://{uri.netloc}".format(uri=parsed_uri)

    try:
        request = urllib.request.Request(url, None, HEADERS)
        response = urllib.request.urlopen(request)
    except HTTPError as e:
        if int(e.code) == 404:
            return None
        raise e

    soup = BeautifulSoup(response, features="lxml")
    fic = Fic(url=url)

    for elem in soup.findAll("a", {"class": "creator-nickname"}):
        fic.authors.append(elem.text.strip())
    for elem in soup.findAll("a", {"class": "creator-username"}):
        fic.authors.append(elem.text.strip())
    assert fic.authors, url

    for elem in soup.findAll("a", {"class": "tag"}):
        fic.tags.append(elem.text.strip())

    elem = soup.find("div", {"itemprop": "description"})
    assert elem is not None, url
    fic.description = elem.text.strip()

    for elem in soup.findAll("h1", {"itemprop": "name"}):
        fic.title = elem.text.strip()
    for elem in soup.findAll("h1", {"itemprop": "headline"}):
        fic.title = elem.text.strip()
    assert fic.title, url

    for elem in soup.findAll("div", {"class": "badge-with-icon"}):
        for cl in elem["class"]:
            if "direction" in cl:
                fic.direction = elem.find("span", {"class": "badge-text"}).text.strip()
                break
            if "rating" in cl:
                fic.rating = elem.find("span", {"class": "badge-text"}).text.strip()
                break
            if "status" in cl:
                fic.status = elem.find("span", {"class": "badge-text"}).text.strip()
                break
            if "like" in cl:
                fic.likes = int(elem.find("span", {"class": "badge-text"}).text.strip())
                break


    elem = soup.find("a", {"class": "pairing-link"})
    if elem is not None:
        fic.pairing = elem.text.strip()

    elem = soup.find("div", {"class": "mb-10"})
    if elem is not None:
        elem = elem.find("a")
        if elem is not None:
            fic.category = elem.text.strip()

    if len(soup.findAll("div", {"id": "content"})) >= 1:
        part = parse_part(soup)
        part["url"] = url
        fic.parts.append(part)
    else:
        links = []
        for link in soup.findAll("a", {"class": "part-link visit-link"}):
            links.append(link["href"])
        for part_link in links:
            part_url = host + part_link
            part_request = urllib.request.Request(part_url, None, HEADERS)
            part_html = urllib.request.urlopen(part_request)
            part_soup = BeautifulSoup(part_html, features="lxml")
            part = parse_part(part_soup)
            part["url"] = part_url.replace("#part_content", "")
            fic.parts.append(part)
            time.sleep(sleep_time)

    fic.part_count = len(fic.parts)
    assert fic.part_count > 0, url
    return fic


def main(output_path, visited_urls_path, sleep_time: int = 1):
    urls = set()
    if os.path.exists(visited_urls_path):
        with open(visited_urls_path) as r:
            lines = r.readlines()
            urls = {l.strip() for l in lines}

    with open(visited_urls_path, "a") as urls_file, open(output_path, "a") as out:
        for i in range(1, 15000000):
            url = f'https://ficbook.net/readfic/{i}'
            if url in urls:
                continue
            try:
                print(f"Processing {url}...")
                record = get_fic(url, sleep_time)
                if record is not None:
                    record = asdict(record)
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print("OK!")
                else:
                    print("404")
                urls.add(url)
                urls_file.write(url + "\n")
            except Exception as e:
                print(url)
                print(traceback.format_exc())
            time.sleep(sleep_time)


if __name__ == "__main__":
    fire.Fire(main)


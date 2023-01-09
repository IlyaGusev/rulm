# Based on https://github.com/EleutherAI/stackexchange-dataset/blob/master/pairer.py

import os
import re
import sys
import traceback
import xml.etree.ElementTree as etree
from collections import defaultdict

from bs4 import BeautifulSoup
from tqdm import tqdm

from data_processing.util import PlainArchive, normalize, remove_non_printable


def remove_line_breaks(text):
    return " ".join(text.split())


def header_info(xml_path):
    os.system("head {}".format(xml_path))


def handle_unicode_errors(txt):
    return txt.encode('utf-8', 'replace').decode()


def is_question(elem_attribs):
    if elem_attribs["PostTypeId"] is not None:
        if elem_attribs["PostTypeId"] == "1":
            return True
    return False


def is_answer(elem_attribs):
    if elem_attribs["PostTypeId"] is not None:
        if elem_attribs["PostTypeId"] == "2":
            return True
    return False


def is_accepted_answer(a_attribs, q_attribs):
    assert is_question(q_attribs), "Must be a question to have an accepted answer"
    assert is_answer(a_attribs), "Must be an answer to be an accepted answer"
    if q_attribs["AcceptedAnswerId"] is not None:
        if q_attribs["AcceptedAnswerId"] == a_attribs["Id"]:
            return True
    else:
        return False


def has_answers(elem_attribs):
    assert is_question(elem_attribs), "Must be a question to have answers"
    if elem_attribs["AnswerCount"] is not None:
        if int(elem_attribs["AnswerCount"]):
            return True
    return False


def trim_attribs(elem_attribs, attrib_type="question"):
    """deletes non-useful data from attribs dict for questions / answers, returns remaining"""
    if attrib_type == "question":
        to_keep = ['Id', 'Body', 'Title', 'Tags', 'AnswerCount', 'AcceptedAnswerId', 'PostTypeId']
        to_delete = [x for x in elem_attribs.keys() if x not in to_keep]
        [elem_attribs.pop(x, None) for x in to_delete]
        elem_attribs["ParsedAnswers"] = 0
        elem_attribs["Answers"] = {}
    elif attrib_type == "answer":
        to_keep = ['Id', 'Body', 'Score']
        new_dict = {}
        for item in to_keep:
            new_dict[item] = elem_attribs[item]
        return new_dict
    else:
        raise Exception('Unrecognized attribute type - please specify either question or answer')


class Converter:
    def __init__(self, xml_path, output_path, min_score=3, max_responses=3):
        self.xml_path = xml_path
        self.questions = defaultdict(lambda: None, {})
        self.min_score = min_score
        self.max_responses = max_responses
        self.archive = PlainArchive(output_path)

    def __call__(self):
        for event, elem in tqdm(etree.iterparse(self.xml_path, events=('end',)), desc="Parsing {} XML file".format(self.xml_path)):
            if elem.tag == "row":
                try:
                    attribs = defaultdict(lambda: None, elem.attrib)
                    if is_question(attribs):
                        if not has_answers(attribs):
                            # if the question has no answers, discard it
                            continue
                        trim_attribs(attribs, "question")
                        self.questions[attribs["Id"]] = attribs
                    elif is_answer(attribs):
                        # if is accepted answer, append answer Body to relevant questions "AcceptedAnswer" field
                        # if the answer's score > min_score
                        # append the answer to the relevant question's OtherAnswers dict
                        self.add_answer(attribs)
                        self.check_complete(attribs)
                    elem.clear()
                except:
                    traceback.print_exc()

    def is_above_threshold(self, a_attribs):
        assert is_answer(a_attribs), "Must be an answer to be above threshold"
        if a_attribs["Score"] is not None:
            if int(a_attribs["Score"]) >= self.min_score:
                return True
        return False

    def add_answer(self, a_attribs):
        """
        Adds answer to its parent question in self.questions if it's either an accepted answer or above self.min_score.
         If answer is an accepted answer, it gets appended to the AcceptedAnswer field, otherwise it gets appended to
         OtherAnswers.
         Also increments the question's 'ParsedAnswers' field. When ParsedAnswers = AnswerCount, the question is deleted
         from memory and saved to a text file.
        :param a_attribs: Answer's attribute dict
        """
        assert is_answer(a_attribs), "Must be an answer to add to parent"
        if a_attribs is not None and self.questions[a_attribs["ParentId"]] is not None:
            if is_accepted_answer(a_attribs, self.questions[a_attribs["ParentId"]]):
                self.questions[a_attribs["ParentId"]]["Answers"][a_attribs["Id"]] = trim_attribs(a_attribs, "answer")
                self.questions[a_attribs["ParentId"]]["ParsedAnswers"] += 1
            elif self.is_above_threshold(a_attribs):
                if a_attribs["Id"] is not None:
                    parent = self.questions[a_attribs["ParentId"]]
                    if parent is not None:
                        self.questions[a_attribs["ParentId"]]["Answers"][a_attribs["Id"]] = trim_attribs(a_attribs, "answer")
                        self.questions[a_attribs["ParentId"]]["ParsedAnswers"] += 1
                else:
                    self.questions[a_attribs["ParentId"]]["ParsedAnswers"] += 1
            else:
                self.questions[a_attribs["ParentId"]]["ParsedAnswers"] += 1

    def check_complete(self, a_attribs):
        """
        checks if the parent question of the previously added answer has no future answers, and if so,
        removes from dict and prints to file.
        """
        keys_to_del = []
        parent = self.questions[a_attribs["ParentId"]]
        if a_attribs is not None and parent is not None:
            if parent["AnswerCount"] is not None and parent["ParsedAnswers"] is not None:
                if int(parent["ParsedAnswers"]) == int(parent['AnswerCount']):
                    keys_to_del.append(a_attribs["ParentId"])
                    if parent["Answers"] is not None and len(parent["Answers"]) > 0:
                        out_str = ""
                        out_str += 'Вопрос: '
                        if parent["Title"] is not None:
                            fragment = BeautifulSoup(parent["Title"], "html.parser").get_text()
                            fragment = normalize(fragment)
                            out_str += '{} '.format(fragment)
                        if parent["Body"] is not None:
                            fragment = BeautifulSoup(parent["Body"], "html.parser").get_text()
                            fragment = normalize(fragment)
                            out_str += '{} '.format(fragment)
                        if parent["Answers"] is not None:
                            key_score_dict = {}
                            for k, a in parent["Answers"].items():
                                key_score_dict[k] = int(a["Score"])
                            key_score_dict = {k: v for k, v in sorted(key_score_dict.items(), key=lambda item: item[1], reverse=True)}
                            count = 0
                            for k in key_score_dict:
                                if count >= self.max_responses:
                                    break
                                fragment = BeautifulSoup(parent["Answers"][k]["Body"], "html.parser").get_text()
                                fragment = normalize(fragment)
                                out_str += 'Ответ: {}\n'.format(fragment)
                                count += 1
                        text = out_str
                        self.archive.add_data(text, meta={"source": "stackoverflow"})
        for key in keys_to_del:
            self.questions.pop(key, None)


input_path = sys.argv[1]
output_path = sys.argv[2]

converter = Converter(input_path, output_path)
converter()
converter.archive.commit()


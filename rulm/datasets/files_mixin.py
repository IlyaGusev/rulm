import os
from typing import List, Callable

class FilesMixin:
    def __init__(self,
                 input_files: List[str],
                 process_line: Callable,
                 encoding: str='utf-8'):
        self.input_files = input_files
        self.encoding = encoding
        assert len(self.input_files) != 0

    def _get_lines_gen(self):
        for file_name in self.input_files:
            assert os.path.exists(file_name)
            with open(file_name, "r", encoding=self.encoding) as r:
                for line in r:
                    yield line


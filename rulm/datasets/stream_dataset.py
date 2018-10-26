import os
from typing import List, Callable

from rulm.datasets.text_dataset import TextDataset


class StreamDataset(TextDataset):
    def __init__(self, input_files: List[str], process_line: Callable, encoding="utf-8"):
        TextDataset.__init__(self, input_files, process_line, encoding)

        self.overall_count = 0
        for _ in self._get_lines_gen():
            self.overall_count += 1

        self.lines_gen = self._get_lines_gen()
        self.current_line_number = -1

    def __getitem__(self, index):
        if index < self.current_line_number:
            self.lines_gen = self._get_lines_gen()
        for line in self.lines_gen:
            sample = self.process_line(line)
            self.current_line_number += 1
            if self.current_line_number == index:
                return sample

    def __len__(self):
        return self.overall_count


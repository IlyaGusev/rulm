import os
from typing import List, Callable, Generator, Any

from torch.utils.data import Dataset

from rulm.datasets.files_mixin import FilesMixin


class StreamDataset(Dataset):
    def __init__(self, process_line: Callable, lines_gen: Generator[List[str], Any, None]=tuple()):
        self.overall_count = 0
        self.current_line_number = -1
        self.process_line = process_line
        self.lines = list(lines_gen)
        self.lines_gen = (x for x in self.lines)

        # Calculate overall lines count (DataLoader requirement)
        self.restart()
        for _ in self.lines_gen:
            self.overall_count += 1
        self.restart()

    def __getitem__(self, index):
        if index < self.current_line_number:
            self.restart()
        for line in self.lines_gen:
            sample = self.process_line(line)
            self.current_line_number += 1
            if self.current_line_number == index:
                return sample

    def restart(self):
        self.lines_gen = (x for x in self.lines)
        self.current_line_number = -1

    def __len__(self):
        return self.overall_count


class StreamFilesDataset(StreamDataset, FilesMixin):
    def __init__(self, input_files: List[str], process_line: Callable, encoding="utf-8"):
        FilesMixin.__init__(self, input_files, encoding)
        StreamDataset.__init__(self, process_line)

    def restart(self):
        self.lines_gen = self._get_lines_gen()
        self.current_line_number = -1


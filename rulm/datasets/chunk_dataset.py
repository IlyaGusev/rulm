import os
from typing import List, Callable

import numpy as np
import torch

from torch.utils.data import Dataset

from rulm.datasets.files_mixin import FilesMixin


class PreprocessingState:
    def __init__(self, default_chunk_shape, dtype):
        self.chunk_count = 0
        self.sample_count = 0
        self.overall_count = 0
        self.default_chunk_shape = default_chunk_shape
        self.chunk_dtype = dtype
        self.chunk = np.zeros(default_chunk_shape, dtype=self.chunk_dtype)

    def add_sample(self, sample):
        self.chunk[self.sample_count][:len(sample)] = sample
        self.sample_count += 1
        self.overall_count += 1

    def save_chunk(self, dir_path):
        self.chunk = self.chunk[:self.sample_count, :]
        chunk_file_name = os.path.join(dir_path, "{}.dat".format(self.chunk_count))
        f = np.memmap(chunk_file_name, dtype=self.chunk_dtype, mode='w+', shape=self.chunk.shape)
        f[:, :] = self.chunk[:, :]

    def reset_chunk(self):
        self.chunk_count += 1
        self.sample_count = 0
        self.chunk = np.zeros(self.default_chunk_shape, dtype=self.chunk_dtype)


class ChunkDataset(Dataset, FilesMixin):
    dtype = "int32"

    def __init__(self,
                 input_files: List[str],
                 process_line: Callable,
                 intermediate_directory: str,
                 max_sample_length: int=50,
                 encoding: str='utf-8',
                 chunk_size: int=1000000):
        FilesMixin.__init__(self, input_files, encoding)

        self.process_line = process_line
        self.intermediate_directory = intermediate_directory
        self.max_sample_length = max_sample_length
        self.chunk_size= chunk_size
        assert self.chunk_size > 0

        self.overall_count = 0
        self._preprocess()

        self.chunks = []
        for file_name in os.listdir(intermediate_directory):
            if ".dat" in file_name:
                self.chunks.append(os.path.join(intermediate_directory, file_name))
        self.chunks.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        self.current_chunk = None
        self.current_chunk_number = None

    def __getitem__(self, index):
        chunk_number = index // self.chunk_size
        sample_number = index % self.chunk_size
        last_chunk_number = self.overall_count // self.chunk_size
        is_last_chunk = last_chunk_number == chunk_number
        if chunk_number != self.current_chunk_number:
            chunk_file_name = os.path.join(self.intermediate_directory, "{}.dat".format(chunk_number))
            current_chunk_size = self.chunk_size if not is_last_chunk else self.overall_count % self.chunk_size
            current_chunk_shape = (current_chunk_size, self.max_sample_length)
            self.current_chunk = np.memmap(chunk_file_name, dtype=ChunkDataset.dtype, mode='r', shape=current_chunk_shape)
            self.current_chunk_number = chunk_number
        return np.array(self.current_chunk[sample_number])

    def __len__(self):
        return self.overall_count

    def _preprocess(self):
        length_file_name = os.path.join(self.intermediate_directory, ".length")
        if os.path.exists(self.intermediate_directory):
            assert os.path.exists(length_file_name)
            with open(length_file_name, "r") as r:
                self.overall_count = int(next(r).strip())
            return
        os.makedirs(self.intermediate_directory, exist_ok=True)

        default_chunk_shape = (self.chunk_size, self.max_sample_length)
        state = PreprocessingState(default_chunk_shape, self.dtype)
        for line in self._get_lines_gen():
            sample = self.process_line(line)
            state.add_sample(sample)
            if state.sample_count == self.chunk_size:
                state.save_chunk(self.intermediate_directory)
                state.reset_chunk()
        if state.sample_count != 0:
            state.save_chunk(self.intermediate_directory)
        self.overall_count = state.overall_count
        with open(length_file_name, "w") as w:
            w.write(str(self.overall_count))


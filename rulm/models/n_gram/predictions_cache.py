from typing import Iterable
from queue import PriorityQueue
from datetime import datetime
import logging

from allennlp.common.registrable import Registrable

logger = logging.getLogger(__name__)


class PredictionsCache(Registrable):
    def __init__(self,
                 capacity: int,
                 timestamps_capacity: int):
        self.capacity = capacity
        self.timestamps_capacity = timestamps_capacity

        self.data = dict()
        self.timestamps = PriorityQueue()
        self.last_timestamp = dict()

        self.miss_count = 0
        self.success_count = 0

    def __getitem__(self, context: Iterable[int]):
        result = self.data.get(context, None)
        if result is not None:
            self._update_ts(context)
            self.success_count += 1
            if self.success_count % self.capacity == 0:
                ratio = int(self.ratio * 100)
                size = int(float(len(self.data)) / self.capacity * 100)
                timestamps_size = int(float(self.timestamps.qsize()) / self.timestamps_capacity * 100)
                message = "Cache ratio: {}%, size: {}%, timestamps: {}%"
                logger.info(message.format(ratio, size, timestamps_size))
            return result
        else:
            self.miss_count += 1
            return None

    def __setitem__(self, context: Iterable[int], prediction: Iterable[float]):
        assert context not in self.data
        while len(self.data) >= self.capacity or self.timestamps.qsize() >= self.timestamps_capacity:
            ts, c = self.timestamps.get()
            assert c in self.last_timestamp
            if ts == self.last_timestamp[c]:
                del self.data[c]
                del self.last_timestamp[c]
        self.data[context] = prediction
        self._update_ts(context)

    def _update_ts(self, context: Iterable[int]):
        ts_now = int(datetime.now().strftime('%s%f'))
        self.timestamps.put((ts_now, context))
        self.last_timestamp[context] = ts_now

    @property
    def ratio(self):
        return float(self.success_count)/(self.miss_count + self.success_count)

    def __len__(self):
        return len(self.data)
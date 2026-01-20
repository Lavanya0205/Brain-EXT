from collections import deque

class ShortTermMemory:
    def __init__(self, max_len=5):
        self.buffer = deque(maxlen=max_len)

    def add(self, data: dict):
        self.buffer.append(data)

    def get(self):
        return list(self.buffer)

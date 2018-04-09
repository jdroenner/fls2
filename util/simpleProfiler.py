from datetime import datetime

class SimpleProfiler:

    def __init__(self):
        self.starts = []
        self.ends = []
        self.other_params = []

    def start(self, name):
        cur = datetime.utcnow()
        self.starts.append((name, cur))

    def stop(self, name):
        cur = datetime.utcnow()
        self.ends.append((name, cur))

    def stats(self):
        start_dict = dict(self.starts)
        times = [(name, start_dict[name], end, end - start_dict[name]) for (name, end) in self.ends]
        return times

    def reset(self):
        self.starts = []
        self.ends = []
        self.other_params = []

    def add_param(self, key, value):
        self.other_params.append((key, value))

    def get_params(self):
        return self.other_params

    def stats_and_params(self):
        return self.stats().extend(self.get_params())
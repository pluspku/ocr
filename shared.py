import numpy as np
from collections import deque
class Meter:
    def __init__(self, ma = 100):
        self.data = deque(maxlen = ma)

    def update(self, x):
        self.data.append(x)

    def __str__(self):
        return '%.4f' % np.nanmean(self.data)


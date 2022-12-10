import numpy as np
from collections import deque

allow_dirty = True

class Meter:
    def __init__(self, ma = 100):
        self.data = deque(maxlen = ma)

    def update(self, x):
        self.data.append(x)

    def __str__(self):
        return '%.4f' % np.nanmean(self.data)

    def val(self):
        return np.nanmean(self.data)


def checksum():
    import git
    repo = git.Repo()
    if not allow_dirty:
        assert not repo.is_dirty()
    ret = repo.head.commit.hexsha[:7]
    if repo.is_dirty():
        ret += ".dev"
    return ret

control_key = 'train_mode'
import redis
rconn = redis.Redis()
def train_mode():
    return rconn.get(control_key).decode("utf8") or "A"
def set_mode(mode):
    rconn.set(control_key, mode)
    return rconn.get(control_key).decode("utf8")

from dataset import WordDataset, ClearDataset
from torch.utils.data import ConcatDataset

import sqlite3

db = '/mnt/tmp/ocr/sqlite.db'

with sqlite3.Connection(db) as conn:
    cur = conn.cursor()
    rows = cur.execute("SELECT id, left, top, right, bottom, label FROM labels order by id").fetchall()

cutoff = int(len(rows) * 0.9)

RATIO = 0.0
def get_training_set():
    wd = WordDataset(rows[:cutoff])
    cd = ClearDataset()
    return ConcatDataset([wd] + [cd] * int(len(wd) / len(cd) * RATIO))

def get_test_set():
    return WordDataset(rows[cutoff:])

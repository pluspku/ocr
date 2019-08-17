from aip import AipOcr

""" 你的 APPID AK SK """
APP_ID = '14965936'
API_KEY = 'GEoAONjg4MGzKYhGummfvGV8'
SECRET_KEY = 'Pu1IqTaguSz07L4FnjrhiBs3kFAGGggb'

client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

from tqdm import tqdm as log_progress
import datasets

datasets.MAX = 1000000
ds = datasets.ImageDataset("datasets/hlm2", [lambda x: x], mode='test')

print(len(ds))

import threading
import tempfile
iolock = threading.Lock()

def run(i):
    d = ds[i]
    tmp = tempfile.mktemp('.png')
    d['A'].save(tmp)
    with open(tmp, 'rb') as f:
        data = f.read()
    resp = client.basicGeneral(data, {'probability': 'true'})
    try:
        if resp['words_result']:
            with iolock:
                with open('mapping.csv', 'a') as f:
                    f.write("%d.tif,%s,%f,%d\n" % ( 
                        i,
                        resp['words_result'][0]['words'][0],
                        resp['words_result'][0]['probability']['average'],
                        i,
                        ))
    except:
        import traceback
        traceback.print_exc()
        print(resp)

from multiprocessing.pool import ThreadPool
pool = ThreadPool(10)

for _ in log_progress(pool.imap_unordered(run, range(len(ds))), total = len(ds)):
    pass


from aip import AipOcr

""" 你的 APPID AK SK """
APP_ID = '14965936'
API_KEY = 'GEoAONjg4MGzKYhGummfvGV8'
SECRET_KEY = 'Pu1IqTaguSz07L4FnjrhiBs3kFAGGggb'

client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

import os
from datasets import get_image
from tqdm import tqdm as log_progress

def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

import threading
import tempfile
iolock = threading.Lock()
from PIL import Image
chunk_size = 10
def run(subst):
    img = Image.new("L", (80 * len(subst), 80))
    for i, x in enumerate(subst):    
        im = get_image(os.path.join(root, '%s.tif' % x))
        img.paste(im, (i*80, 0))
    tmp = tempfile.mktemp('.png')
    img.save(tmp)
    with open(tmp, 'rb') as f:
        data = f.read()
    resp = None
    try:
        resp = client.basicGeneral(data, {'probability': 'true'})
        if len(resp['words_result']) == 1:
            if len(resp['words_result'][0]['words']) == len(subst):
                out = list(zip(subst, resp['words_result'][0]['words'], [resp['words_result'][0]['probability']['average']] * len(subst)))
                with iolock:
                    with open('mapping.csv', 'a') as f:
                        for row in out:
                            f.write("%s\t%s\t%f\n" % row)
        return resp
    except:
        import traceback
        traceback.print_exc()
        print(resp)
    finally:
        #os.unlink(tmp)
        pass

if __name__ == '__main__':
    import sys
    ds, mode = sys.argv[1:3]
    root = "datasets/%s/%s/A" % (ds, mode)
    files = [int(f.replace('.tif', '')) for f in os.listdir(root)]

    import pandas as pd
    if os.path.exists('mapping.csv'):
        done = set(pd.read_csv("mapping.csv", sep = '\t', names=["src", "word", "prob"]).src.tolist())
    else:
        done = set([])
    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(10)
    batches = [x for x in chunks([f for f in files if f not in done], chunk_size)]

    for _ in log_progress(pool.imap_unordered(run, batches), total = len(batches)):
        pass


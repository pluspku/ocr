from aip import AipOcr

""" 你的 APPID AK SK """
APP_ID = '14965936'
API_KEY = 'GEoAONjg4MGzKYhGummfvGV8'
SECRET_KEY = 'Pu1IqTaguSz07L4FnjrhiBs3kFAGGggb'

client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

import os
from datasets import get_image
from tqdm import tqdm as log_progress
root = "datasets/hlm2/train/A"
files = [int(f.replace('.tif', '')) for f in os.listdir(root)]

def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

import threading
import tempfile
iolock = threading.Lock()
from PIL import Image
def run(subst):
    img = Image.new("L", (80 * 3, 80 * 3 ))
    for i, x in enumerate(subst):    
        im = get_image(os.path.join(root, '%s.tif' % x))
        img.paste(im, ((i%3)*80, ((i)//3) * 80))
    tmp = tempfile.mktemp('.png')
    img.save(tmp)
    with open(tmp, 'rb') as f:
        data = f.read()
    try:
        resp = client.basicGeneral(data, {'probability': 'true'})
        if len(resp['words_result']) == 3:
            out = []
            for i in range(3):
                if len(resp['words_result'][i]['words']) == 3:
                    out.extend(list(zip(subst[i*3:i*3+3], resp['words_result'][i]['words'], [resp['words_result'][i]['probability']['average']] * 3)))            
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
        os.unlink(tmp)

if __name__ == '__main__':

    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(10)
    batches = [x for x in chunks(files, 9) if len(x) == 9]

    for _ in log_progress(pool.imap_unordered(run, batches), total = len(batches)):
        pass


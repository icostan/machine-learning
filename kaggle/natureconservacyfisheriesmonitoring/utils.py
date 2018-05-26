import os
import time
import matplotlib.pyplot as plt
from PIL import Image
import imagehash


def benchmark(func):
    def wrapper(*arg, **kw):
        t1 = time.process_time()
        res = func(*arg, **kw)
        t2 = time.process_time()
        return (t2 - t1), res, func.__name__
    return wrapper


def log(label, text, suffix=''):
    print(str(label) + ': ' + str(text) + ' ' + str(suffix))


def plot_image(img):
    plt.imshow(img)


#
# Bytes to MB, GB, ...
# https://gist.github.com/shawnbutts/3906915
#
def bytesto(bytes, to, bsize=1024):
    """convert bytes to megabytes, etc.
       sample code:
           print('mb= ' + str(bytesto(314575262000000, 'm')))
       sample output:
           mb= 300002347.946
    """
    a = {'k': 1, 'm': 2, 'g': 3, 't': 4, 'p': 5, 'e': 6}
    r = float(bytes)
    for i in range(a[to]):
        r = r / bsize

    return(r)


# perceptual hash fingerprints
def fingerprints(path):
    hs = {}
    filenames = os.listdir(path)
    log('Calculate fingerprints for', len(filenames), suffix='files')
    for filename in filenames:
        h = imagehash.average_hash(Image.open(path + filename))
        hs[filename] = h
    log('Fingerprints: ', len(hs))
    return hs

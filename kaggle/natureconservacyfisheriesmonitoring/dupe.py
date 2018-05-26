from PIL import Image
import imagehash
import os
import utils
import matplotlib.pyplot as plt


path = 'input/train/ALB/'


def show_image(filename):
    img = Image.open(path + filename)
    plt.imshow(img)


fingerprints = utils.fingerprints(path)


dupes = 0
for filename in os.listdir(path):
    print('Processing ' + filename)
    f1 = fingerprints[filename]
    for fname in os.listdir(path):
        if filename == fname:
            continue
        f2 = fingerprints[fname]
        if f1 == f2:
            dupes += 1
            os.remove(path + fname)
    utils.log('dupes: ', dupes)

utils.log('DUPES: ', dupes)

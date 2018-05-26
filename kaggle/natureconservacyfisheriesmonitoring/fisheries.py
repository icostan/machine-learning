import json
import cv2
import os
import skimage as ski
import skimage.io as io
from PIL import Image
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from vis.visualization import visualize_cam
from vis.utils import utils
import matplotlib.pyplot as plt
import matplotlib.patches as pts

import dataset
import annotations
from utils import log


def visualize():
    model = load_model('fisheries.h5')

    path = './input/train/ALB/img_00003.jpg'
    img = utils.load_img(path, target_size=(dataset.SIZE, dataset.SIZE))
    X = dataset.load_image(path)
    print(X.shape)
    # print(X[0])

    preds = model.predict(X)
    pred_class = preds.argmax()
    print('Predicted:' + str(preds))
    print('Predicted:' + dataset.TYPES[pred_class])

    plt.imshow(img)
    plt.show()

    idx = [2, 6, 8, 13]
    for i in idx:
        print(model.layers[i])
        heatmap = visualize_cam(model, i, [pred_class], img)
        plt.imshow(heatmap)
        plt.show()

def show_annotated_images():
    fish = 'ALB'
    fish_path = 'input/train/{fish}/'.format(fish=fish)

    files = os.listdir(fish_path)
    for filename in files[:5]:
        img = Image.open(fish_path + filename)
        a = annotations.for_image(filename, fish)

        x = a['x']
        y = a['y']
        size = max(a['width'], a['height'])
        h = size
        w = size

        # fish_img = img.crop((x, y, x + w, y + h))
        # plt.imshow(fish_img)

        # print(str(x) + ':' + str(y) + ' ' + str(w) + ':' + str(h))
        log('Size', size)

        plt.gca().add_patch(
            Rectangle((a['x'], a['y']), w, h, fill=None))
        plt.imshow(img)

        plt.show()

def fish_path(fish):
    return 'input/train/{fish}/'.format(fish=fish)

def bounding_boxes():
    test_path = 'input/test_stg2/'
    model_x = load_model('fisheries-localization-x.h5')
    model_y = load_model('fisheries-localization-y.h5')

    files = os.listdir(test_path)
    for filename in files:
        file_path = test_path + filename
        img = io.imread(file_path)
        X = dataset.load_image(file_path)

        pred_x = model_x.predict(X)
        pred_y = model_y.predict(X)
        print(str(pred_x) + ',' + str(pred_y))

        plt.imshow(img)
        plt.gca().add_patch(pts.Circle((pred_x[0,0], pred_y[0,0]),50))
        plt.gca().add_patch(pts.Rectangle((pred_x[0,0], pred_y[0,0]), 300, 300, fill=None))
        plt.show()

# show_annotated_images()
# visualize()
bounding_boxes()

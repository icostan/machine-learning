import cv2
import numpy as np

from keras.preprocessing.image import img_to_array
from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_saliency


# Build the VGG16 network with ImageNet weights
model = VGG16(weights='imagenet', include_top=True)
print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'predictions'
layer_idx = [idx for idx, layer in enumerate(
    model.layers) if layer.name == layer_name][0]

# Images corresponding to tiger, penguin, dumbbell, speedboat, spider
image_paths = [
    "http://www.tigerfdn.com/wp-content/uploads/2016/05/How-Much-Does-A-Tiger-Weigh.jpg",
    "http://www.slate.com/content/dam/slate/articles/health_and_science/wild_things/2013/10/131025_WILD_AdeliePenguin.jpg.CROP.promo-mediumlarge.jpg",
    "https://www.kshs.org/cool2/graphics/dumbbell1lg.jpg",
    "http://tampaspeedboatadventures.com/wp-content/uploads/2010/10/DSC07011.jpg",
    "http://ichef-1.bbci.co.uk/news/660/cpsprodpb/1C24/production/_85540270_85540265.jpg"
]

heatmaps = []
for path in image_paths:
    # Predict the corresponding class for use in `visualize_saliency`.
    seed_img = utils.load_img(path, target_size=(224, 224))
    pred_class = np.argmax(model.predict(np.array([img_to_array(seed_img)])))

    # Here we are asking it to show attention such that prob of `pred_class`
    # is maximized.
    heatmap = visualize_saliency(model, layer_idx, [
                                 pred_class], seed_img, text=utils.get_imagenet_label(pred_class))
    heatmaps.append(heatmap)

cv2.imshow("Saliency map", utils.stitch_images(heatmaps))
cv2.waitKey(0)

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

# img_path = '/Users/icostan/Github/darknet/data/dog.jpg'
# img_path = '/Users/icostan/Work/kaggle/natureconservacyfisheriesmonitoring/input/train/LAG/img_04741.jpg'
img_path = '/Users/icostan/Work/kaggle/natureconservacyfisheriesmonitoring/input/train/ALB/img_00085.jpg'
# img_path = '/Users/icostan/Work/kaggle/natureconservacyfisheriesmonitoring/input/train/YFT/img_00217.jpg'

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('')
print('==> Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225),
# (u'n01871265', u'tusker', 0.1122357), (u'n02504458',
# u'African_elephant', 0.061040461)]

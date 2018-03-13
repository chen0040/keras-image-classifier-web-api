from keras.preprocessing.image import load_img, img_to_array
import os

current_dir = os.path.join(os.path.dirname(__file__))
img = load_img(current_dir + '/bi_classifier_data/training/cat/cat.0.jpg')
x = img_to_array(img)
print(x.shape)
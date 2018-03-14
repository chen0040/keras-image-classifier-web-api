from keras.preprocessing.image import load_img, img_to_array
import os

current_dir = os.path.dirname(__file__)
current_dir = current_dir if current_dir is not '' else '.'

img = load_img(current_dir + '/bi_classifier_data/training/cat/cat.0.jpg')
x = img_to_array(img)
print(x.shape)
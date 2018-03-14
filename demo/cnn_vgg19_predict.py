from keras.models import Model
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.optimizers import SGD
from PIL import Image
from keras.preprocessing.image import img_to_array
import numpy as np
import os

model = VGG19(include_top=True, weights='imagenet')
model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])


def predict(filename):
    img = Image.open(filename)
    img = img.resize((224, 224), Image.ANTIALIAS)
    input = img_to_array(img)
    input = np.expand_dims(input, axis=0)
    input = preprocess_input(input)
    output = decode_predictions(model.predict(input), top=3)
    print(output)


for i in range(100):
    current_dir = os.path.dirname(__file__)
    current_dir = current_dir if current_dir is not '' else '.'

    predict(current_dir + '/bi_classifier_data/training/cat/cat.' + str(i) + '.jpg')
    predict(current_dir + '/bi_classifier_data/training/dog/dog.' + str(i) + '.jpg')

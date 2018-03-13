from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np
import os

current_dir = os.path.join(os.path.dirname(__file__))

json = open(current_dir + '/models/cnn_bi_classifier_architecture.json', 'r').read()

model = model_from_json(json)
model.load_weights(current_dir + '/models/cnn_bi_classifier_weights.h5')

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


def predict(filename, label):
    img = Image.open(filename)
    img = img.resize((150, 150), Image.ANTIALIAS)

    input = np.asarray(img)
    input = input.astype('float32') / 255
    input = np.expand_dims(input, axis=0)

    print(input.shape)

    output = model.predict(input)

    print('This is a ' + label)
    print('Probability that it is a dog ' + str(output[0][0]))


for i in range(100):
    predict(current_dir + '/bi_classifier_data/training/cat/cat.' + str(i) + '.jpg', 'cat')
    predict(current_dir + '/bi_classifier_data/training/dog/dog.' + str(i) + '.jpg', 'dog')

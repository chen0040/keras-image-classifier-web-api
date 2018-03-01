from keras.models import model_from_json
import os
from PIL import Image
import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils

# load and configure the cifar19 classifier model
cifar10_model = model_from_json(
    open(os.path.join('./models', 'cnn_cifar10_architecture.json')).read())
cifar10_model.load_weights(os.path.join('./models', 'cnn_cifar10_weights.h5'))

cifar10_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


def predict_cifar10(filename):
    img = Image.open(filename)
    img = img.resize((32, 32), Image.ANTIALIAS)

    input = np.asarray(img)
    input = input.astype('float32') / 255
    input = np.expand_dims(input, axis=0)

    print(input.shape)

    output = cifar10_model.predict_classes(input)[0]
    return output


(Xtrain, Ytrain), (Xtest, Ytest) = cifar10.load_data()

Xtest = Xtest.astype('float32') / 255
Ytest = np_utils.to_categorical(Ytest, 10)

for i in range(Xtest.shape[0]):
    x = Xtest[i]
    x = np.expand_dims(x, axis=0)
    y = np.argmax(Ytest[i])
    predicted_y = cifar10_model.predict_classes(x)
    print('actual: ', y, '\tpredicted: ', predicted_y)

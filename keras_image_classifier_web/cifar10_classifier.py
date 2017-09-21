from keras.models import model_from_json
from PIL import Image
import numpy as np
import os

class Cifar10Classifier:
    cifar10_model = None

    def __init__(self):
        # load and configure the cifar19 classifier model
        self.cifar10_model = model_from_json(
            open(os.path.join('../keras_image_classifier/models', 'cnn_cifar10_architecture.json')).read())
        self.cifar10_model.load_weights(os.path.join('../keras_image_classifier/models', 'cnn_cifar10_weights.h5'))
        self.cifar10_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def predict(self, filename):
        img = Image.open(filename)
        img = img.resize((32, 32), Image.ANTIALIAS)

        input = np.asarray(img)
        input = input.astype('float32') / 255
        input = np.expand_dims(input, axis=0)

        print(input.shape)

        predicted_class = self.cifar10_model.predict_classes(input)[0]

        labels = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"
        ]
        return predicted_class, labels[predicted_class]

    def run_test(self):
        print(self.predict('../keras_image_classifier/bi_classifier_data/training/cat/cat.2.jpg'))

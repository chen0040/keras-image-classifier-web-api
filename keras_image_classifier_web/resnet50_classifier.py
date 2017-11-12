from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.optimizers import SGD
from PIL import Image
from keras.preprocessing.image import img_to_array
import numpy as np


class ResNet50Classifier:
    model = None

    def __init__(self):
        self.model = ResNet50(include_top=True, weights='imagenet')
        self.model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

    def predict(self, filename):
        img = Image.open(filename)
        img = img.resize((224, 224), Image.ANTIALIAS)
        input = img_to_array(img)
        input = np.expand_dims(input, axis=0)
        input = preprocess_input(input)
        output = decode_predictions(self.model.predict(input), top=3)
        return output[0]

    def run_test(self):
        print(self.predict('../keras_image_classifier/bi_classifier_data/training/cat/cat.3.jpg'))


if __name__ == '__main__':
    classifier = ResNet50Classifier()
    classifier.run_test()

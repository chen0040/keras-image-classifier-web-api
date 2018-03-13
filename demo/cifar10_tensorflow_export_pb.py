import sys
import os


def main():
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from keras_image_classifier.library.cifar10_classifier import Cifar10Classifier

    current_dir = os.path.join(os.path.dirname(__file__))

    classifier = Cifar10Classifier()
    classifier.load_model(os.path.join(current_dir, 'models'))
    classifier.export_tensorflow_model(output_fld=os.path.join(current_dir, 'models', 'tf'))


if __name__ == '__main__':
    main()


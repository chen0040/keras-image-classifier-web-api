from keras_image_classifier.library.cifar10_classifier import Cifar10Classifier


def main():
    classifier = Cifar10Classifier()
    classifier.load_model('./models')
    classifier.export_tensorflow_model(output_fld='./models/tf')

if __name__ == '__main__':
    main()


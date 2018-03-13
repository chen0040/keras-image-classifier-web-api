from keras.datasets import cifar10
import keras.backend as K
import os
import sys


def main():
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from keras_image_classifier.library.cifar10_classifier import Cifar10Classifier

    img_width, img_height = 32, 32
    batch_size = 128
    epochs = 20
    nb_classes = 10
    output_dir_path = os.path.join(os.path.dirname(__file__), 'models')

    (Xtrain, Ytrain), (Xtest, Ytest) = cifar10.load_data()

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    classifier = Cifar10Classifier()

    classifier.fit(Xtrain, Ytrain, model_dir_path=output_dir_path,
                   batch_size=batch_size,
                   epochs=epochs,
                   input_shape=input_shape, nb_classes=nb_classes)

    score = classifier.evaluate(Xtest, Ytest, batch_size=batch_size)

    print('score: ', score[0])
    print('accurarcy: ', score[1])




if __name__ == '__main__':
    main()


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten, Dropout
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
import keras.backend as K

train_data_dir = 'multi_classifier_data/training'
validation_data_dir = 'multi_classifier_data/validation'
img_width, img_height = 32, 32
batch_size = 128
epochs= 20
nb_classes = 10

(Xtrain, Ytrain), (Xtest, Ytest) = cifar10.load_data()

Xtrain = Xtrain.astype('float32') / 255
Xtest = Xtest.astype('float32') / 255

Ytrain = np_utils.to_categorical(Ytrain, nb_classes)
Ytest = np_utils.to_categorical(Ytest, nb_classes)

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(filters=32, input_shape=input_shape, padding='same', kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=32, padding='same', kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, padding='same', kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))

model.add(Flatten())
model.add(Dense(units=512))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=nb_classes))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=Xtrain, y=Ytrain, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2)

score = model.evaluate(x=Xtest, y=Ytest, batch_size=batch_size, verbose=1)

print('score: ', score[0])
print('accurarcy: ', score[1])

json = model.to_json()
open('models/cnn_cifar10_architecture.json', 'w').write(json)
model.save_weights('models/cnn_cifar10_weights.h5', overwrite=True)


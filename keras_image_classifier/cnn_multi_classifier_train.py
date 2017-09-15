from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

train_data_dir = 'multi_classifier_data/training'
validation_data_dir = 'multi_classifier_data/validation'
img_width, img_height = 150, 150
batch_size = 16
epochs= 50
nb_classes = 10
nb_train_samples = 2000
nb_validation_samples = 800

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

validation_datagen = ImageDataGenerator(
    rescale=1. / 255,
)

train_generator = train_datagen.flow_from_directory(
    directory=train_data_dir,
    class_mode='categorical',
    classes=nb_classes,
    batch_size=batch_size
)

validation_generator = validation_datagen.flow_from_directory(
    directory=validation_data_dir,
    class_mode='categorical',
    classes=nb_classes,
    batch_size=batch_size
)

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(filters=32, input_shape=input_shape, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=32, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))

model.add(Flatten())
model.add(Dense(units=64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=nb_classes))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(generator=train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    validation_generator=validation_generator,
                    validation_steps=nb_validation_samples // batch_size)

json = model.to_json()
open('models/cnn_multi_classifier_architecture.json', 'w').write(json)
model.save_weights('models/cnn_multi_classifier_weights.h5', overwrite=True)


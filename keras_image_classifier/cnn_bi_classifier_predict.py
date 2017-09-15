from keras.models import model_from_json

model = model_from_json('models/cnn_bi_classifier_architecture.json')
model.load_weights('models/cnn_bi_classifier_weights.h5')

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.predict_classes()
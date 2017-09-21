import os
from keras.models import model_from_json
from PIL import Image
import numpy as np

from flask import Flask, request, session, g, redirect, url_for, abort, \
    render_template, flash
from flask import send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)  # create the application instance :)
app.config.from_object(__name__)  # load config from this file , flaskr.py

# Load default config and override config from an environment variable
app.config.from_envvar('FLASKR_SETTINGS', silent=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

model = model_from_json(open(os.path.join('../keras_image_classifier/models', 'cnn_bi_classifier_architecture.json')).read())
model.load_weights(os.path.join('../keras_image_classifier/models', 'cnn_bi_classifier_weights.h5'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


def predict(filename):
    img = Image.open(filename)
    img = img.resize((150, 150), Image.ANTIALIAS)

    input = np.asarray(img)
    input = np.expand_dims(input, axis=0)

    print(input.shape)

    output = model.predict(input)

    probability_of_a_dog = output[0][0]
    predicted_label = ("Dog" if probability_of_a_dog > 0.5 else "Cat")
    return probability_of_a_dog, predicted_label

prob, label = predict('../keras_image_classifier/bi_classifier_data/training/cat/cat.2.jpg')
print(prob)
print(label)


@app.route('/')
def classifiers():
    return render_template('classifiers.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/about', methods=['GET'])
def about():
    return 'about us'


@app.route('/cats_vs_dogs', methods=['GET', 'POST'])
def cats_vs_dogs():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('cats_vs_dogs_result',
                                    filename=filename))
    return render_template('cats_vs_dogs.html')


@app.route('/cats_vs_dogs_result/<filename>')
def cats_vs_dogs_result(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    probability_of_dog, predicted_label = predict(filepath)
    return render_template('cats_vs_dogs_result.html', filename=filename,
                           probability_of_dog=probability_of_dog, predicted_label=predicted_label)


@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == '__main__':
    app.run(debug=True)

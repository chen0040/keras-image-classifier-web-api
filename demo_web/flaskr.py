import os
import sys
from flask import Flask, request, redirect, url_for, render_template, flash
from flask import send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def main():
    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir is not '' else '.'

    app = Flask(__name__)  # create the application instance :)
    app.config.from_object(__name__)  # load config from this file , flaskr.py

    # Load default config and override config from an environment variable
    app.config.from_envvar('FLASKR_SETTINGS', silent=True)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

    from keras_image_classifier.library.cnn_bi_classifier import BiClassifier
    from keras_image_classifier.library.cifar10_classifier import Cifar10Classifier
    from keras_image_classifier.library.vgg16_classifier import VGG16Classifier
    from keras_image_classifier.library.vgg19_classifier import VGG19Classifier
    from keras_image_classifier.library.resnet50_classifier import ResNet50Classifier

    bi_classifier = BiClassifier()
    cifar10_classifier = Cifar10Classifier()
    vgg16_classifier = VGG16Classifier()
    vgg19_classifier = VGG19Classifier()
    resnet50_classifier = ResNet50Classifier()


    @app.route('/')
    def classifiers():
        return render_template('classifiers.html')


    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


    def store_uploaded_image(action):
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
            return redirect(url_for(action,
                                    filename=filename))


    @app.route('/about', methods=['GET'])
    def about():
        return 'about us'


    @app.route('/cats_vs_dogs', methods=['GET', 'POST'])
    def cats_vs_dogs():
        if request.method == 'POST':
            return store_uploaded_image('cats_vs_dogs_result')
        return render_template('cats_vs_dogs.html')


    @app.route('/cifar10', methods=['GET', 'POST'])
    def cifar10():
        if request.method == 'POST':
            return store_uploaded_image('cifar10_result')
        return render_template('cifar10.html')


    @app.route('/vgg16', methods=['GET', 'POST'])
    def vgg16():
        if request.method == 'POST':
            return store_uploaded_image('vgg16_result')
        return render_template('vgg16.html')


    @app.route('/vgg19', methods=['GET', 'POST'])
    def vgg19():
        if request.method == 'POST':
            return store_uploaded_image('vgg19_result')
        return render_template('vgg19.html')


    @app.route('/resnet50', methods=['GET', 'POST'])
    def resnet50():
        if request.method == 'POST':
            return store_uploaded_image('resnet50_result')
        return render_template('resnet50.html')


    @app.route('/cats_vs_dogs_result/<filename>')
    def cats_vs_dogs_result(filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        probability_of_dog, predicted_label = bi_classifier.predict(filepath)
        return render_template('cats_vs_dogs_result.html', filename=filename,
                               probability_of_dog=probability_of_dog, predicted_label=predicted_label)


    @app.route('/cifar10_result/<filename>')
    def cifar10_result(filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        predicted_class, predicted_label = cifar10_classifier.predict_label(filepath)
        return render_template('cifar10_result.html', filename=filename,
                               predicted_class=predicted_class, predicted_label=predicted_label)


    @app.route('/vgg16_result/<filename>')
    def vgg16_result(filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        top3 = vgg16_classifier.predict(filepath)
        return render_template('vgg16_result.html', filename=filename,
                               top3=top3)


    @app.route('/vgg19_result/<filename>')
    def vgg19_result(filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        top3 = vgg19_classifier.predict(filepath)
        return render_template('vgg19_result.html', filename=filename,
                               top3=top3)


    @app.route('/resnet50_result/<filename>')
    def resnet50_result(filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        top3 = resnet50_classifier.predict(filepath)
        return render_template('resnet50_result.html', filename=filename,
                               top3=top3)


    @app.route('/images/<filename>')
    def get_image(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'],
                                   filename)

    bi_classifier.run_test(current_dir)
    cifar10_classifier.run_test(current_dir)
    vgg16_classifier.run_test(current_dir)
    vgg19_classifier.run_test(current_dir)
    resnet50_classifier.run_test(current_dir)
    app.run(debug=True, use_reloader=False)


if __name__ == '__main__':
    main()

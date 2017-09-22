# keras-image-classifier-web-api

This provides keras DCNN-based image classifiers codes with flask as the web api server for the image classifiers.

# Usage

Run the following command to install the keras, flask and other dependency modules:

```bash
sudo pip install -r requirements.txt
```

Goto keras_image_classifier_web directory and run the following command:

```bash
python flaskr.py
```

Now navigate your browser to http://localhost:5000 and you can try out various predictors built with the following
trained classifiers:

* binary classifier on "cats vs dogs" data
* multi-class DCNN classifier trained with CIFAR-10 data
* multi-class VGG16 classifier trained with ImageNet data
* multi-class VGG19 classifier trained with ImageNet data




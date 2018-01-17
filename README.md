# keras-image-classifier-web-api

This provides keras DCNN-based image classifiers codes with flask as the web api server for the image classifiers.

# Usage

Run the following command to install the keras, flask and other dependency modules:

```bash
sudo pip install -r requirements.txt
```

## Training (Optional)

As the trained models are already included in the "keras_image_classifier/models" folder in the project, the training is
not required. However, if you like to tune the parameters and retrain the models, you can use the 
following command to run the training:

```bash
cd keras_image_classifier/training
python cnn_cifar10_train.py
```

The above commands will train dcnn model on the cifar10 dataset 
dataset and store the trained model in "keras_image_classifier/models/cnn_cifar10_**"

If you like to train other models, you can use the same command above on another train python scripts:

* cnn_bi_classifier.py: a simple cnn bi-class classifier trained using data in the "keras_image_classifier/bi-classifier_data" folder

## Running Web Api Server

Goto keras_image_classifier/web directory and run the following command:

```bash
python flaskr.py
```

Now navigate your browser to http://localhost:5000 and you can try out various predictors built with the following
trained classifiers:

* binary classifier on "cats vs dogs" data
* multi-class DCNN classifier trained with CIFAR-10 data
* multi-class VGG16 classifier trained with ImageNet data
* multi-class VGG19 classifier trained with ImageNet data
* multi-class Residual Network classifier trained with ImageNet data




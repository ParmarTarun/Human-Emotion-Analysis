
# Human Emotion Analysis

Using a Convolutional Neural Network (CNN) to recognize facial expressions from images or video/camera stream.

## Steps Involved In the Project:

- Pre-Processing :heavy_check_mark:
- Training :heavy_check_mark:
- Optimizing :heavy_check_mark:
- Predicting

## <a>1. Installed dependencies</a>

- Numpy
- Pandas
- Tensorflow
- Scipy
- Dlib
- cv2
- skimage
- hyperopt

## <a>2. Dataset Used</a>
- [ferc2013.csv](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

Fer2013 is a challenging dataset. The images are not aligned and some of them are uncorrectly labeled. Moreover, some samples do not contain faces.


### <a>Dataset includes:</a>
- Emotion column containg labels as 0,1,2,3,...,6 corresponding to 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise and Neutral)
- Pixels column with actual pixel values.
- Usage column with label as Training, PublicTest or PrivateTest.

## <a>3. Tasks Performed</a>
- [Convert From fer2013 to Images and numpy arrays](convert.py)
- The above file extracts data from the dataset fer2013 and stores the image file (.jpg) and numpy array files (.npy) for images, labels, landmarks and hog features to a folder named "fer2013_features" into different subfolders as TestData, TrainingData and PrivateData.
- [Train the model](train.py)
- The above file imports the [model.py](model.py) using [data_loader.py](data_loader.py) and train the model using Tensorflow. You can also adjust the parameters according to your interest in [parameters.py](paramters.py)
- [Optimize with hyperopt](optimize_hyperparameters.py)
- The above file uses hyperopt to provide us with best hyper parameters.

# How To Run:
- Clone the project into a folder
- Download [ferc2013.csv](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- Download [Dlib Shape Preictor model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) 
- Move the csv file and shape predictor file in root folder of your project.
- Run command <code>pip install -r requirements.txt</code> 
    <b>Note:</b> Requirements contain requirements for whole project.
- Run <code>python convert.py</code> <br/>
- Run <code>python train.py --train=yes</code><br/>
    <b>Note:</b> You can also provide argument like --evaluate=yes to just evaluate.
- Run <code>python optimize_hyperparameters.py <code/><br/>This file takes argument -m for total number of evaluations.
<i>Recommended: create a virtual environment with python3.6</i>

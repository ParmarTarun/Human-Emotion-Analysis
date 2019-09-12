
# Human Emotion Analysis

Using a Convolutional Neural Network (CNN) to recognize facial expressions from images or video/camera stream.

## Steps Involved In the Project:

- Pre-Processing :heavy_check_mark:
- Training :heavy_check_mark:
- Otimizing
- Predicting

## <a>1. Installed dependencies</a>

- Numpy
- Pandas
- Tensorflow
- Scipy
- Dlib
- cv2
- skimage

## <a>2. Dataset Used</a>
- [ferc2013.csv](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

Fer2013 is a challenging dataset. The images are not aligned and some of them are uncorrectly labeled as we can see from the following images. Moreover, some samples do not contain faces. 


### <a>Dataset includes:</a>
- Emotion column containg labels as 0,1,2,3,...,6 corresponding to 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise and Neutral)
- Pixels column with actual pixel values.
- Usage column with label as Training, PublicTest or PrivateTest.

## <a>3. Tasks Performed</a>
- [Convert From fer2013 to Images](convert.py)
- The above file extracts data from the dataset fer2013 and stores the image file (.jpg) and numpy array files (.npy) for images, labels, landmarks and hog features to a folder named "fer2013_features" into different subfolders as TestData, TrainingData and PrivateData.

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

<i>Recommended: create a virtual environment with python3.6</i>
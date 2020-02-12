# kws_ml_capstone_project
A keyword spotting system to classify simple words

In 2017 Google’s teams TensorFlow and AIY released to the public the dataset “Speech Commands Dataset”, containing 65,000 one-second long utterances of 30 short words [1]. This dataset was aimed to provide a simple but robust base for the development and testing of speech recognition systems. Alongside this, they also released a set o example models [2] and sponsor the Kaggle competition “TensorFlow Speech Recognition Challenge” [3] so developers or enthusiast with low knowledge could start right away working and learning.
Following the bases of the Kaggle competition, the aim of the this project is to build a functional speech recognition system able to spot, from an one second audio file, the following set of words: yes, no, up, down, left, right, on, off, stop, and go, plus two labels for unknown words and silence, so there are 12 possible labels.   


# Problem Statement
The approach to find a solution for the given problem statement is to build, train and test a TensorFlow model for a classification task, having as input a one-second clip and as output one of the 12 possible labels. The task involved will be:
1. Download the dataset and explore the data provided in it. Notebook “1_Data_Exploration.ipynb”.
2. Build an input-pipeline to feed efficiently the data into the model and define some data augmentations techniques. Notebook “2_Preprocessing.ipynb”.
3. Evaluate different audio features that could be used as input for the models. Notebook “3_Feature_Engineering.ipynb”.
4. Build and test different custom models, evaluating its performance on the proposed features. Refinement is applied,  improving model architecture and tuning training hyperparameters. Notebook “4_Model_Training.ipynb”.


# Requirements
    Python >= 3.6
    tensorflow 2.x
    numpy >= 1.17
    pandas >= 0.25
    matplotlib >= 3.1
    seaborn >= 0.10


# References
[1]. Warden P. (2017, August 24). Launching the Speech Commands Dataset. Retrieved from: https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html

[2] TensorFlow tutorials, Simple Audio Recognition. 2017. https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/sequences/audio_recognition.md#simple-audio-recognition

[3] TensorFlow Speech Recognition Challenge. 2017. https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/overview

[4] TensorFlow Speech Recognition Challenge, Evaluation. 2017. https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/overview/evaluation

[5] Sainath T, Parada C. Convolutional Neural Networks for Small-footprint Keyword Spotting. INTERSPEECH 2015.

[6] Nair Pratheeksha, The dummy’s guide to MFCC, 2018. https://medium.com/prathena/the-dummys-guide-to-mfcc-aceab2450fd

[7] Simonyan K, Zisserman A, Very Deep Convolutional Networks for Large-Scale Image Recognition, ICLR 2015. 


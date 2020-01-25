# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 17:39:31 2020

@author: Min Thaw
"""

# -*- coding: utf-8 -*-

# Artifical Neural Network
# Min Thaw Zin (Udemy Course)

# Installing Theano (open-sourced numerical library, numpy syntax on both
# CPU and GPU) GPU is a stronger computational power compared to CPU as well


# Installing TensorFlow (open-sourced numerical library, developed by Google
# used mainly for deep learning)


# Installing Keras (wraps Theano and Tensorflow, for deep-learning model)
# pip install keras 
# conda upgrade --all

# Part 1 - Data Reprocessing

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
 # Import the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# Encoding categorical data (Country and Gender in this case)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Create encoder change France, Germany and Spain to 0,1,2
labelencoder_National = LabelEncoder()
# [:, 1] -> index of the column in this case the 2nd column
X[:, 1] = labelencoder_National.fit_transform(X[:, 1])

# Encoder for Gender
labelencoder_Gender = LabelEncoder()
X[:, 2] = labelencoder_Gender.fit_transform(X[:, 2])

# Use of ColumnTransformer to transfer the data type to float
# Create dummy variables for nationals
transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [1]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
X = transformer.fit_transform(X.tolist())
# change type to float64
X = X.astype('float64')

# remove one column to avoid dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Feature Scaling to ease out calculations
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Creating Artificial Neural Network

# Import Keras Libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialise the ANN
classifier = Sequential()

# Adding input layer and first hidden layer
# output dim -> (inputs + 1)/2
# activation -> rectivation -> relu
classifier.add(Dense(6, kernel_initializer='uniform', activation ='relu', input_dim = 11))
classifier.add(Dropout(p = 0.1))

# Adding second hidden layer
classifier.add(Dense(6, kernel_initializer='uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

# Adding output layer use of sigmoid function
classifier.add(Dense(1, kernel_initializer='uniform', activation = 'sigmoid'))

# Compile the ANN 
# compile(optimiser, loss, metrics=[], weight)
# optimiser -> algorithm for calculating weights
classifier.compile(optimizer= 'adam', loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

# Fit the ANN to the training set data
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch=100)

# Predicting the test results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Adding a prediction for a specific criteria
# France, 600, Male, 40, 3, 60000, 2, Yes, Yes, 50000
# Must apply the same scale

new_prediction = classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction = (new_prediction > 0.5)

# Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Evalute the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# Function to build classifier ( ANN )
def build_classifier():
    # Initialise the ANN
    classifier = Sequential()
    
    # Adding input layer and first hidden layer
    # output dim -> (inputs + 1)/2
    # activation -> rectivation -> relu
    classifier.add(Dense(6, kernel_initializer='uniform', activation ='relu', input_dim = 11))
    
    # Adding second hidden layer
    classifier.add(Dense(6, kernel_initializer='uniform', activation = 'relu'))
    
    # Adding output layer use of sigmoid function
    classifier.add(Dense(1, kernel_initializer='uniform', activation = 'sigmoid'))
    
    # Compile the ANN 
    # compile(optimiser, loss, metrics=[], weight)
    # optimiser -> algorithm for calculating weights
    classifier.compile(optimizer= 'adam', loss = 'binary_crossentropy',
                       metrics = ['accuracy'])
    
    return classifier

# running n_jobs will run all cpus at the same time at -1 value
# running the classifier on 10 folds in small batches   
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch=100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10,
                             n_jobs = -1)

# getting mean and variance for visualising in the bias_variance trade off
mean = accuracies.mean()
variance = accuracies.std()

# Dropout regulation 
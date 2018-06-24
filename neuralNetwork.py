#this file was used to train the neural network and extract the data from the trained network so it could be used in the application
#Framework for setting up a neural network adapted from https://www.kdnuggets.com/2016/10/beginners-guide-neural-networks-python-scikit-learn.html
import numpy as np
from skimage import io

import csv
import os

X , y = [], []

def readCSVFile(path):
    with open(path, newline = '') as f:
        reader = csv.reader(f)
        return list(reader)

def add(classification, unprocessedFolder, processedFolder):
    imgsAndDigits = readCSVFile(classification)

    def parseName(imgName):
        return str(int(imgName[4:-4])) + "-"
        #the str(int()) is a check to make sure it's an int

    #imgNos = list(map(parseName, imgsAndDigits))
    for i in range(len(imgsAndDigits)):
        imgName, digits = imgsAndDigits[i][0], imgsAndDigits[i][1:]
        prefix, suffix = parseName(imgName), ".jpg"
        print(digits)
        for (j, digit) in enumerate(digits):
            if digit == "?": continue
            if digit == "": break
            num = digit
            imgName = prefix + str(j) + suffix
        
            filePath = os.path.join(unprocessedFolder, imgName)
            img = io.imread(filePath)
            size = img.shape
            
            filePath = os.path.join(processedFolder, imgName)
            img = io.imread(filePath)
            x = img.flatten()
            x = np.append(x, size)
            X.append(x)
            try:
                num = int(num)
            except:
                if num == "+": num = 11
                elif num == "-": num = 12
                elif num == "*": num = 13
                else: raise Exception
            y0 = num
            y.append(y0)

add('totalDigit-classification.csv', 'unprocessedDigits', 'processedDigits')

X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)

#split data and classification into training sets and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

#normalize the training data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
    # 1. Fit only to the training data
scaler.fit(X_train)

    # 2. Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#create an instance of the model (neural network)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30), activation = 'logistic')

#now train the model
print("Training")
mlp.fit(X_train,y_train)
print("Trained")

#now use it to make predictions and see how accurate it is
predictions = mlp.predict(X_test)
probabilities = mlp.predict_proba(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

print(y_test)
print(predictions)

from sklearn.externals import joblib
joblib.dump(mlp, 'NewMLPClassifier.pkl')
joblib.dump(scaler, 'NewScaler.pkl')

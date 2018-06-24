#tester to find misclassified images

# idea: currently using as input 1) size ratio and 
# 2) output of neural network
# can add 3) position of top few maxes in histogram data and pass into a 
# logistic regressor to see if the digits should be further segmented or not

from sklearn.externals import joblib

mlp = joblib.load('NewMLPClassifier.pkl')
scaler = joblib.load('NewScaler.pkl')

def classify(img, size):
    x = img.flatten()
    x = np.append(x, size)
    x = x.reshape(1, -1)
    prediction = mlp.predict(scaler.transform(x))
    if prediction == 11: prediction = "+"
    elif prediction == 12: prediction = "-"
    elif prediction == 13: prediction = "*"
    else: prediction = str(prediction[0])
    #print(prediction)
    return mlp.predict_proba(scaler.transform(x)), prediction

from skimage import io
from skimage.morphology import *
from skimage import measure, feature
import matplotlib.pyplot as plt
import numpy as np

'''
img = io.imread("digits/8-1.jpg")
size = (io.imread("unprocessedDigits/8-1.jpg")).shape
print(size)
print(classify(img , size))
'''
#print(np.amax(img.flatten()))
#plt.imshow(img)
#plt.show()


import csv
import os
#Conc: if < 0.65, definitely segment further it seems...
def readCSVFile(path):
    with open(path, newline = '') as f:
        reader = csv.reader(f)
        return list(reader)

digits = readCSVFile("evenMoreDigit-classification.csv")

imgs = os.listdir('evenMoreDigits')
imgs = list(filter(lambda x: ".jpg" in x, imgs))

def parseName(fileName):
    fileName = fileName[:-4]
    return list(map(int, fileName.split("-")))

imgNos = list(map(parseName, imgs))

concerningImgs = []
normalImgs = []
excludeRows = []
for i in range(len(imgs)):
    imgName, [rowNo, colNo] = imgs[i], imgNos[i]
    if rowNo in excludeRows or digits[rowNo][colNo] in "-+": continue
    if digits[rowNo][colNo] == "?":
        concerningImgs.append(imgName)
    else:
        normalImgs.append(imgName)
print(concerningImgs)
for imgName in concerningImgs:
    img = io.imread(os.path.join("evenMoreUnprocessedDigits", imgName))
    img = img > 0.5 * 255
    print(np.amax(img.flatten()))

    for i in range(1):
        img = opening(img)#, selem = np.ones((10, 10)))
    img = 1 - img
    img = skeletonize(img)
    img = dilation(img)
    plt.imshow(img)
    plt.show()
    contours = measure.find_contours(img, 0)
    '''
    img = feature.canny(img, sigma = 5)
    dilatedImg = dilation(img)#, np.ones((3, 1)))
    #for i in range(2):
    #    dilatedImg = dilation(dilatedImg)#, selem = np.ones((3, 1)))
    contours = measure.find_contours(dilatedImg, 0.8)
    '''
    print(len(contours))
    #print(len(contours))
    #plt.imshow(dilatedImg)
    #plt.show()
    #continue 
    #plt.imshow(dilatedImg)
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    plt.show()



def getProbs(imgs):
    resultProbs = []
    resultPreds = []
    for imgName in imgs:
        img = io.imread(os.path.join("moreDigits", imgName))
        size = (io.imread(os.path.join("moreUnprocessedDigits", imgName))).shape
        probs, prediction = classify(img, size)
        resultProbs.append(np.amax(probs))
        resultPreds.append(prediction)
    return resultProbs, resultPreds

concerningProbs, concerningPreds = getProbs(concerningImgs)
normalProbs, normalPreds = getProbs(normalImgs)

concerningProbs, normalProbs = np.array(concerningProbs), np.array(normalProbs)

print("Concerning Probs: ")
print("Max, min, average: ", np.amax(concerningProbs), np.amin(concerningProbs), np.average(concerningProbs))

print("Normal Probs: ")
print("Max, min, average: ", np.amax(normalProbs), np.amin(normalProbs), np.average(normalProbs))
'''
sortedNormalImgs = list(zip(normalImgs, normalProbs.tolist(), normalPreds))
sortedNormalImgs.sort(key = lambda x: x[1])

sortedConcerningImgs = list(zip(concerningImgs, concerningProbs.tolist(), concerningPreds))
sortedConcerningImgs.sort(key = lambda x: -x[1])

print(sortedNormalImgs[:10], sortedNormalImgs[10][1])
print(sortedConcerningImgs)
'''
#print(concerningImgs)

concerningSizes = []
normalSizes = []

import numpy as np
from skimage import io
import matplotlib.pyplot as plt

for imgName in concerningImgs:
    filePath = os.path.join('unprocessedDigits', imgName)
    img = io.imread(filePath)
    #plt.imshow(img)
    #plt.show()
    concerningSizes.append(img.shape[0]/img.shape[1])
    if img.shape[0]/img.shape[1] > 0.65: print(imgName, img.shape[0]/img.shape[1])
#print(normalImgs)
for imgName in normalImgs:
    filePath = os.path.join('unprocessedDigits', imgName)
    img = io.imread(filePath)
    if imgName == "0-9.jpg": print(img.shape)
    normalSizes.append(img.shape[0]/img.shape[1])

concerningSizes = np.array(concerningSizes)
normalSizes = np.array(normalSizes)

print(list(filter(lambda x: x > 0.65, concerningSizes)))

print("Concerning height-to-width: ")
print(np.amax(concerningSizes), np.average(concerningSizes), np.amin(concerningSizes))
print("Normal height-to-width: ")
print(np.amax(normalSizes), np.average(normalSizes), np.amin(normalSizes)) 
print(normalImgs[np.argmin(normalSizes)])  
normalSizes[np.argmin(normalSizes)] = np.average(normalSizes)
print(np.amin(normalSizes)) 
print(normalImgs[np.argmin(normalSizes)]) 

for i in range(len(normalImgs)):
    size, prob = normalSizes[i], normalProbs[i]
    if (size < 0.7 and prob < 0.8) or (size < 0.8 and prob < 0.5):
        print("Uh-oh: misclassified", normalImgs[i], size, prob, normalPreds[i])

for i in range(len(concerningImgs)):
    size, prob = concerningSizes[i], concerningProbs[i]
    if (size >= 0.7 or prob >= 0.8) and (prob >= 0.5 or size >= 0.8):
        print("Uh-oh: unclassified", concerningImgs[i], size, prob, concerningPreds[i])

#IDEA: Classify all one and two stroke images first?

#1, -
#7, +, *

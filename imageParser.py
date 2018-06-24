#file to combine data from neural network with position data from image
#to generate the mathematical expression to be solved

from sklearn.externals import joblib
import numpy as np
from integratedImageProcessing import *

mlp = joblib.load('NewMLPClassifier.pkl')
scaler = joblib.load('NewScaler.pkl')

def classify(img):
    size = img.shape
    box = resizeToSquare(img)
    x = box.flatten()
    x = np.append(x, size)
    x = x.reshape(1, -1)
    prediction = mlp.predict(scaler.transform(x))
    prob = np.amax(mlp.predict_proba(scaler.transform(x)))
    #print(mlp.predict_proba(scaler.transform(x)))
    if prediction == 11: prediction = "+"
    elif prediction == 12: prediction = "-"
    elif prediction == 13: prediction = "*"
    else: prediction = str(prediction[0])
    return prob, prediction

def splitImg(img):
    img = img/255
    img = opening(img)#, selem = np.ones((10, 10)))
    img = 1 - img
    img = skeletonize(img)
    img = dilation(img)
    contours = measure.find_contours(img, 0)
 
    kMin = lambda x, y: [min(x[0], y[0]), min(x[1], y[1])]
    kMax = lambda x, y: [max(x[0], y[0]), max(x[1], y[1])]
    boxes = [reduce(kMin, contour)+reduce(kMax, contour) 
            for contour in contours]
    
    def overlapsInside(box, boxes, i):
        [ya, xa, yb, xb] = box
        for j in range(len(boxes)):
            if i == j: continue
            [y0, x0, y1, x1] = boxes[j]
            if ((x0 <= xa and xb <= x1) and (y0 <= ya and yb <= y1)):
                return True
        return False
    
    boundingBoxes = []
    i = 0
    while i < len(boxes):
        if not overlapsInside(boxes[i], boxes, i):
            boundingBoxes += [boxes[i]]
        i += 1
    #print(boundingBoxes)
    boundingBoxes.sort(key = lambda x: x[1])
    #print("Max = ", np.amax(img.flatten()))
    if len(boundingBoxes) == 2:
        return (int(boundingBoxes[0][3]), int(boundingBoxes[1][1]))
    else:
        #print("Length of bounding boxes is %d"%len(boundingBoxes))
        '''
        plt.imshow(img[:, :len(img)//2])
        plt.show()
        plt.imshow(img[:, len(img)//2:])
        plt.show()
        '''
        return (len(img)//2, len(img)//2)

def parseText(threshImg, boxCoords):
    assert(max(threshImg.flatten()) > 250)
    boxCoords.sort(key = lambda x: x[1])
    predictions = []
    for coords in boxCoords:
        [r0, c0, r1, c1] = list(map(int, coords))
        prob, prediction = classify(threshImg[r0:r1+1, c0:c1+1])
        size = (r1+1-r0)/(c1+1-c0)
        '''
        if (size < 0.7 and prob < 0.8) or (size < 0.8 and prob < 0.5):
            #keep both alternatives like a dictionary?
            end1, start2 = splitImg(threshImg[r0:r1+1, c0:c1+1])
            mainImg = threshImg[r0:r1+1, c0:c1+1]
            
            plt.figure(2)
            plt.imshow(img1)
            plt.figure(3)
            plt.imshow(img2)
            plt.show()
            
            #print("Shapes")
            #print((r1+1-r0), (c1+1-c0))
            #print(img1.shape, img2.shape)
            prob1, prediction1 = classify(mainImg[:, :end1])
            prob2, prediction2 = classify(mainImg[:, start2:])
            #if prob1 > 0.5 and prob2 > 0.5:
            print(prob, prediction, prob1, prediction1, prob2, prediction2)
            print("Predictions", prediction1, prediction2)
            predictions.append(prediction1)
            predictions.append(prediction2)
        else:
        '''
        predictions.append(prediction)
    #print(predictions, boxCoords)
    predictions, boxCoords = dealWithDivides(predictions, boxCoords)
    #print("Expression")
    print("".join(predictions))
    return "".join(predictions)

#deals with the case of fractions, including nested fractions
def dealWithDivides(predictions, boxCoords, startIndex = 0):
    try:
        pos = predictions.index("-", startIndex)
    except:
        return predictions, boxCoords #base case
    
    #recursive case
    start = pos
    def overlaps(target, focus):
        #print("Target", "Focus")
        #print(target, focus)
        x0, x1, xa, xb = target[1], target[3], focus[1], focus[3]
        xmid = (x0+x1)/2
        return xmid > xa and xmid < xb
    while start > 0 and overlaps(boxCoords[start-1], boxCoords[pos]):
        start -= 1
    end = pos+1
    while end < len(predictions) and overlaps(boxCoords[end], boxCoords[pos]):
        end += 1
    if start == pos and end == pos+1:
        return dealWithDivides(predictions, boxCoords, pos+1)
    
    #sort out what is above and below
    above, aboveBoxes, below, belowBoxes = [], [], [], []
    line = (boxCoords[pos][0] + boxCoords[pos][2]) / 2
    for i in range(start, end):
        if i == pos: continue
        mid = (boxCoords[i][0] + boxCoords[i][2]) / 2
        if mid < line:
            above.append(predictions[i])
            aboveBoxes.append(boxCoords[i])
        else:
            below.append(predictions[i])
            belowBoxes.append(boxCoords[i])
    if len(above) == 0 or len(below) == 0:
        #probably a minus then
        return dealWithDivides(predictions, boxCoords, pos+1)
    
    aboveList, aboveBoxList = dealWithDivides(above, aboveBoxes)
    belowList, belowBoxList = dealWithDivides(below, belowBoxes)
    newEndPos = end + len(aboveList) + len(belowList)
    #print(aboveList, belowList)
    newPredictionList = (predictions[:start] + 
                         ["("] + aboveList + [")"] + ["/"] +
                         ["("] + belowList + [")"] + predictions[end:])
    newBoxCoords = (boxCoords[:start] + 
                         [None] + aboveBoxList + [None] + [None] +
                         [None] + belowBoxList + [None] + boxCoords[end:])
    return dealWithDivides(newPredictionList, newBoxCoords, newEndPos)
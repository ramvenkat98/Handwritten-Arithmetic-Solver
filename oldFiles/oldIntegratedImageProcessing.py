#Still does not separate all joint-together numbers

import matplotlib.pyplot as plt
import numpy as np
from skimage import color, io
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage import feature
import skimage
from skimage.filters import threshold_local
from skimage.morphology import *
from skimage import transform
from functools import *
from skimage import measure

# sometimes there are multiple numbers in a single box

#only use outer bounding box if both dimensions correspond 
#with inner bounding box correctly

#another idea for acceptable - find region to focus on and just get contours
#in that region

#better acceptable sign
#also, problems with recognizing 1, fraction symbol, division symbol

#fraction sign not being recognized in contours(watch for it)

#use techniques like noise removal etc. to make whole thing more robust
#find a way to take out really noisy random contours

#img = color.rgb2gray(io.imread("IMG_1320.jpg"))[500:, :]
#img *= 255
#img = ndimage.median_filter(img, (15, 15))
#filteredImg = medianImg
#print(img)
#print(np.max(img), np.min(img))
#img = ndimage.gaussian_filter(img, 10)
#print(filteredImg)
#from skimage.morphology import square
#from skimage.filters import threshold_otsu, rank
#radius = 200
#selem = square(radius)
#threshImg = rank.otsu(img, selem)
#thresh = threshold_otsu(img)
#threshImg = img > thresh
#threshImg = threshold_local(img, 501)

def processImg(img):
    threshImg = (img > 0.5)#.astype(float)
    filteredImg = threshImg
#filteredImg = img
#print(np.max(filteredImg), np.min(filteredImg))
    v = np.median(filteredImg)
#print(v)
#---- apply automatic Canny edge detection using the computed median----
    sigma = 0.1
    lower = max(0, (1.0 - sigma) * v)
    upper = min(1, (1.0 + sigma) * v)
    edgeImg = feature.canny(filteredImg)
    dilatedImg = dilation(edgeImg)
    for i in range(2):
        dilatedImg = dilation(dilatedImg)
    return dilatedImg
'''
img = color.rgb2gray(io.imread("attachments/IMG_1300.jpg"))
#img *= 255
img2 = processImg(img) * 255
img *= 255
plt.figure(1)
plt.imshow(img)
plt.figure(2)
plt.imshow(img2)
plt.show()
'''
def showContours(img, show = True):
    plt.clf()
    threshImg = img > 0.5
    filteredImg = threshImg
#filteredImg = img
#print(np.max(filteredImg), np.min(filteredImg))
    v = np.median(filteredImg)
#print(v)
#---- apply automatic Canny edge detection using the computed median----
    sigma = 0.1
    lower = max(0, (1.0 - sigma) * v)
    upper = min(1, (1.0 + sigma) * v)
    edgeImg = feature.canny(filteredImg)#, low_threshold = lower, high_threshold = upper)
    #print(lower, upper)
    ''''
    edgeImg1 = edgeImg
    edgeImg = skeletonize(edgeImg)
    plt.clf()
    plt.figure(1)
    plt.imshow(edgeImg1)
    plt.figure(2)
    plt.imshow(edgeImg)
    plt.show()
    plt.clf()
    '''
    #3 dilations being done

    dilatedImg = dilation(edgeImg)
    for i in range(2):
        dilatedImg = dilation(dilatedImg)
    #dilatedImg = opening(dilatedImg)
    
    contours = measure.find_contours(dilatedImg, 0.8)

    kMin = lambda x, y: [min(x[0], y[0]), min(x[1], y[1])]
    kMax = lambda x, y: [max(x[0], y[0]), max(x[1], y[1])]
    diff = lambda x, y: [abs(x[0]-y[0]), abs(x[1]-y[1])]
    width, height = dilatedImg.shape
    def acceptable(x):
        [x0, y0], [x1, y1] = reduce(kMin, x), reduce(kMax, x)
        #if x0 - 5 < 0 or y0 - 5 < 0 or x1+5 >= width or y1 + 5 >= height:
        #    return False
        diffHeight = diff([x0, y0], [x1, y1])
        return ((diffHeight[0] > 50 or diffHeight[1] > 50)
                and not (diffHeight[0] > width * 4/5 and diffHeight[1] > height * 4/5))
        return diffHeight[0] < 1000 and diffHeight[1] < 1000
        return ((diffHeight[0] > 100 and diffHeight[1] > 100 
                and 1/2 < (diffHeight[0]/diffHeight[1]) < 2
                and diffHeight[0] < 1000 and diffHeight[1] < 1000))
    dims = [diff(reduce(kMin, x), reduce(kMax, x)) for x in contours]
    #print(dims)
    #correctContours = contours
    correctContours = list(filter(acceptable, contours))
    if not show: return kMin, kMax, correctContours, edgeImg, dilatedImg
    plt.figure(1)
    for n, contour in enumerate(correctContours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    plt.show()
    return kMin, kMax, correctContours, edgeImg, dilatedImg
def getBoxes(img):
    kMin, kMax, correctContours, edgeImg, dilatedImg = showContours(img, False)
    boxes = [reduce(kMin, contour)+reduce(kMax, contour) 
            for contour in correctContours]
    
    #THIS
    #maybe make simpler, and restart with original idea of just taking big
    #contours first
    def overlapsInside(box, boxes, i):
        [ya, xa, yb, xb] = box
        for j in range(len(boxes)):
            if i == j: continue
            [y0, x0, y1, x1] = boxes[j]
            if ((x0 <= xa and xb <= x1) and (y0 <= ya and yb <= y1)):
                return True
        '''
                and ((xb-xa) < 50 or (xb-xa) >= 0.7 * (x1-x0) or (yb-ya) <= 0.7 * (y1-y0))):
                return True
            if ((x0 >= xa and xb >= x1) and (y0 >= ya and yb >= y1)
                and ((xb-xa) >= (x1-x0)/0.5 and not (yb-ya) >= (y1-y0)/0.7)):
                return True
        '''
        return False
    
    boundingBoxes = []
    i = 0
    while i < len(boxes):
        if not overlapsInside(boxes[i], boxes, i):
            boundingBoxes += [boxes[i]]
        i += 1
    #print("Bounding boxes", boundingBoxes)
    
    realBoundingBoxes = []
    
    for boundingBox in boundingBoxes:
        [y0, x0, y1, x1] = boundingBox
        inside = lambda l: (l[0] >= y0 and l[1] >= x0
                            and l[2] <= y1 and l[3] <= x1
                            and (not l == [y0, x0, y1, x1]))
        innerBoxes = list(filter(inside, boxes))
        #try this
        innerBoundingBoxes = []
        i = 0
        while i < len(innerBoxes):
            if not overlapsInside(innerBoxes[i], innerBoxes, i):
                innerBoundingBoxes += [innerBoxes[i]]
            i += 1
        if len(innerBoundingBoxes) > 1:
            print("len", len(innerBoundingBoxes))
            #check if contiguous in x direction
            check = True
            for i1 in range(len(innerBoundingBoxes)):
                [yl1, xl1, yr1, xr1] = innerBoundingBoxes[i1]
                print(abs((yl1+yr1)/2 - (y0+y1)/2)/((y0+y1)/2))
                if abs((yl1+yr1)/2 - (y0+y1)/2)/((y0+y1)/2) > 0.2:
                    print("Oh")
                    check = False
                for i2 in range(i1+1, len(innerBoundingBoxes)):
                    [yl2, xl2, yr2, xr2] = innerBoundingBoxes[i2]
                    #print(xl1, xr1, xl2, xr2)
                    if not (xr1 <= xl2 or xl1 >= xr2):
                        #print("nope")
                        check = False
            if check:
                realBoundingBoxes += innerBoundingBoxes
            else:
                realBoundingBoxes += [boundingBox]
        else: realBoundingBoxes += [boundingBox]
            
        #print("Bounding boxes", boundingBoxes)
    print("Ratios")
    print(list(map(lambda x: abs(x[0]-x[2])/abs(x[1]-x[3]), realBoundingBoxes)))
    '''
    realBoundingBoxes = []
    for box in boundingBoxes:
        [ya, xa, yb, xb] = list(map(int, box))
        section = dilatedImg[ya:yb, xa:xb]
        
        plt.clf()
        plt.figure(1)
        plt.imshow(convex_hull_image(section))
        plt.show()
        plt.clf()
        

        plt.figure(1)
        for n, contour in enumerate(contours):
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
        plt.show()
        plt.clf()
    '''
    print("Bounding boxes", realBoundingBoxes)
    return realBoundingBoxes

#getBoxes(color.rgb2gray(io.imread("attachments/IMG_1321.jpg")))

#print(len(contours), len(correctContours))
#print(contours[0])
#print(correctContours[0])
#print(dims)
#contours.sort(key = lambda x: -len(x))

#1. filter out based on dimensions
#2. draw bounding boxes around each contour, merge bounding boxes that
#   overlap a lot

#for i in range(10):
#    print(len(contours[i]))


#print(npContours[0], type(npContours[0]))
#maxes = np.max(npContours, axis = 1)
#mins = np.min(npContours, axis = 1)
#diffs = maxes - mins
#print(npContours)
'''
plt.figure(1)
plt.imshow(img, cmap = 'gray')
plt.figure(2)
plt.imshow(edgeImg, cmap = 'gray')
plt.figure(3)
plt.imshow(dilatedImg, cmap = 'gray')
plt.figure(4)
for n, contour in enumerate(contours):
    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
fig5 = plt.figure(5)
for n, contour in enumerate(correctContours):
    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
import matplotlib.patches as patches
#ax1 = fig5.add_subplot(111, aspect='equal')
#for box in boundingBoxes:
#    rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1])
#    ax.add_patch(rect)

plt.show()

boxList = []
for i, box in enumerate(boundingBoxes):
'''
def resizeToSquare(croppedImg):
    #[r0, c0, r1, c1] = list(map(int, box))
    #croppedImg = threshImg[r0:r1+1, c0:c1+1]
    rows, cols = croppedImg.shape
    extras = abs((rows-cols)//2)
    if rows < cols:
        extras = np.full((extras, cols), 255)
        sqrImg = np.vstack([extras, croppedImg, extras])
    elif cols < rows:
        extras = np.full((rows, extras), 255)
        sqrImg = np.hstack([extras, croppedImg, extras])
    else:
        sqrImg = croppedImg
    #print("OK again")
    sizedImg = transform.resize(sqrImg, (28, 28))
    #print("OK still")
    return sizedImg
'''
    boxList.append(sizedImg)
    plt.figure(i+1)
    plt.imshow(sizedImg)
plt.show()

#create a GUI where you can run through each image, press the number it corresponds to, which will save it, and record the number in a CSV file

'''
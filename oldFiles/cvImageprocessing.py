import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('sampleIntegral.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
img, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img, contours, -1, (0,255,0), 3)

#kernel = np.ones((25,25),np.uint8)
#img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

#blur = cv2.GaussianBlur(gray,(3,3), 0)
#img = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#            cv2.THRESH_BINARY_INV, 7,10)

"""
img = cv2.imread('sampleIntegral.jpg')
#img = cv2.medianBlur(img,11)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.bitwise_not(img)
ret, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#img = cv2.Canny(img,20,50)
#img = cv2.adaptiveThreshold(img, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            #thresholdType=cv2.THRESH_BINARY, blockSize = 13, C = 2)
_, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
cv2.drawContours(img, contours, -1, (0,255,0), 3)
#imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#ret,thresh = cv2.threshold(imgray,127,255,0)
#image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#img = cv2.drawContours(img, contours, -1, (0,255,0), 3)
#img = cv2.medianBlur(img,71)
#cv2.imshow('My window',img)


#cv2.waitKey(0)
#cv2.destroyAllWindows()

#ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

#th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#            cv2.THRESH_BINARY, 11,2)
#th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#            cv2.THRESH_BINARY,15,2)
#titles = ['Original Image', 'Global Thresholding (v = 127)',
#            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
#images = [img, th1, th2, th3]

#for i in xrange(4):
#print(ret)
"""
plt.subplot(2,2,1),plt.imshow(img,'gray')
plt.title('Blurred')
plt.xticks([]),plt.yticks([])

plt.show()

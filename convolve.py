"""
Program Name: Convolution
Purpose: Program that takes an image and a filter that is both square and odd in size and returns a convoluted image
Author: George Roros
References:
1. George's Brain
"""
 
import numpy as np
import cv2

#Read in image file as array
img = np.array(cv2.imread('bird.jpg', 1), dtype='f')
#Get height and width of image array
height, width = img.shape[:2]

#Create 3x3 filter to convolute image
filter = np.array([[1., 1., 1.],
                   [1., 1., 1.],
                   [1., 1., 1.]], dtype='f')

filter = np.multiply(filter, 0.1111111)

fHeight, fWidth = filter.shape[:2]
hfHeight = fHeight/2
hfWidth = fWidth/2
img2 = []

for i in range(hfHeight, height-hfHeight):
    column = []
    for j in range(hfWidth, width-hfWidth):
        iMin = i-hfHeight
        iMax = i+hfHeight+1
        jMin = j-hfHeight
        jMax = j+hfHeight+1
        arrayFrag = img[iMin:iMax, jMin:jMax]
        arrayFrag = np.divide(arrayFrag, 255.0)
        #After the matrix multiplication the tmpArray Values zero out
        tmpArray = filter.dot(arrayFrag)
        tmpPixel = [0,0,0]
        for k in range(0, fHeight):
            for l in range(0, fWidth):
                tmpPixel = np.add(tmpPixel, tmpArray[k][l])
        column.append(tmpPixel)
    img2.append(column)
                    

MIN = np.amin(img2)
MAX = np.amax(img2)

img2 = np.array(img2, dtype='f')
print img2.shape[:2]
print img2

"""for i in range(hfHeight, height-hfHeight-1):
    for j in range(hfWidth, width-hfWidth-1):
        for k in range(0, 3):
            print("i:%d j:%d k:%d" %(i, j, k))
            img2[i][j][k] = (img2[i][j][k] - MIN)/(MAX - MIN)
            print img[i][j][k]
"""

"""
for each pixel apply the filter based on its size
if the filter is 1x1 then do not worry about the 
conditionals below and apply the filter directly to the current pixel
if not add the correct amount of padding for each filter size
"""
""" if i == 0 and not j == 0 and not j == width-1 or j == 0 and not i ==0 and not i == height - 1:
print ("%d %d" %(i, j))
else:
print "0"
"""

#Dispaly new image and wait for user input to close end program
cv2.imshow('bird', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

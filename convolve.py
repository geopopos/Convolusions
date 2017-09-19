"""
Program Name: Convolution
Purpose: Program that takes an image and a filter that is both square and odd in size and returns a convoluted image
Author: George Roros
References:
1. George's Brain
2. https://en.wikipedia.org/wiki/Kernel_(image_processing) - used for filters(edge detection gaussian blur)
"""
 
import numpy as np
import cv2

"""
FUNCTION: apply filter to image
PARAMETERS: img, filter
    img - the image upon which the filter will be applied
    filter - the filter to apply to the image
RETURN: numpy array with convoluted image
"""
def applyFilter(img, filter):
    height, width = img.shape[:2]
    try:
        fHeight, fWidth = filter.shape[:2]
    except:
        fHeight, fWidth = (1, 1)
    
    hfHeight = fHeight/2
    hfWidth = fWidth/2
    
    #create new list to hold image
    img2 = []

    for i in range(0, height):
        column = []
        for j in range(0, width):
            iMin = i-hfHeight
            iMax = i+hfHeight+1
            jMin = j-hfHeight
            jMax = j+hfHeight+1
            arrayFrag = img[iMin:iMax, jMin:jMax]
            # arrayFrag = np.divide(arrayFrag, 255.0)
            #After the matrix multiplication the tmpArray Values zero out
            tmpPixel = [0,0,0]
            for k in range(0, fHeight):
                for l in range(0, fWidth):
                    for m in range(0, 3):
                        try:
                            #calculate new pixel value
                            tmpPixel[m] += filter[k][l] * arrayFrag[k][l][m]
                        except IndexError:
                            try:
                                #if filter size is 1-Dimensional only check for 1-Dimensional index
                                tmpPixel[m] += filter[k] * arrayFrag[k][l][m]
                            except IndexError:
                                #if part of arrayFrag is out of index skip addition jump to next index
                                pass
            column.append(tmpPixel)
        img2.append(column)

    img2 = np.array(img2, dtype='f')
    return img2

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
#MAIN

#Read in image file as array
img = np.array(cv2.imread('bald-eagle-1.jpg', 1), dtype='f')
#Get height and width of image array
img = np.divide(img, 255.0)
#Create nXn filter to convolute image
# **ref 2
# filter = np.array([[1, 4, 6, 4, 1],
#                     [4, 16, 24, 16, 4],
#                     [6, 24, 36, 24, 6],
#                     [4, 16, 24, 16, 4],
#                     [1, 4, 6, 4, 1]], dtype='f')
# filter = np.multiply(filter, 0.00390625)

filter = np.array([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]], dtype='f')

# filter = np.array([1], dtype='f')

img2 = applyFilter(img, filter)

#Dispaly new image and wait for user input to close end program
cv2.imshow('Original Image', img)
cv2.imshow('Convoluted Image', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
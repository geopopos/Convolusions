#program that convolutes a 1d array 'image'
import numpy as np

array = np.array([21, 8, 12, 13, 5, 19, 38])
cArray = array
filter = np.array([-1, 0, 1])

width = array.shape[:2][0]

avg = 0

for i in range(0, width):
    avg += array[i]

avg = avg/width

array = np.insert(array, 0, avg)
array = np.append(array, avg)

for i in range(0, width):
    arrayFrag = array[i:i+3]
    tmpArray = np.multiply(arrayFrag, filter)
    num = 0
    for j in range(0, 3):
        num += tmpArray[j]
    cArray[i] = num

print cArray

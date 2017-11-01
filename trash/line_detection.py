import cv2
import numpy as np
from numpy import ones, vstack
from numpy.linalg import lstsq
from scipy.spatial import distance
from numpy.linalg import norm

img = cv2.imread('cu.jpg')
edges = edges = cv2.Canny(img, 100, 200)

minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)

for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite('cube2.jpg', img)




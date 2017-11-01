import cv2
import numpy as np

filename = 'cu.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imblue = np.float32(gray)
imblue = cv2.medianBlur(imblue, 3)

dst = cv2.cornerHarris(imblue,4,5,0.01)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imwrite('dst.png',img)

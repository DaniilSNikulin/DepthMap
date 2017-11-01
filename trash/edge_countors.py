import cv2
import numpy as np
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.nan)

im = cv2.imread('cube.png', 0)
# imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
imblue = cv2.medianBlur(im, 5)

# edged = cv2.Canny(imgray, 4, 6)

th3 = cv2.adaptiveThreshold(imblue, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                            cv2.THRESH_BINARY, 11, 2)
edged = cv2.Canny(th3, 70, 150)
plt.subplot(131), plt.imshow(im, cmap='gray')
plt.subplot(132), plt.imshow(th3, cmap='gray')
plt.subplot(133), plt.imshow(edged, cmap='gray')
plt.show()
edged = th3

# Detect contours using both methods on the same image
img1, contours1, hierarchy = cv2.findContours(edged, cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)
print(len(contours1), len(hierarchy[0]))

true_contours = []

for i, hier in enumerate(hierarchy[0]):
    # print(hier)
    if hier[2] == -1:
        true_contours.append(contours1[i])

print(len(true_contours))

# cv2.cornerHarris(img1,)

img1 = im.copy()
img2 = im.copy()

# cv2.imwrite("edged", edged)
cv2.drawContours(img1, contours1, -1, (255, 0, 0), 2)
cv2.imwrite("./cube1_all.png", img1)

cv2.drawContours(img2, true_contours, -1, (255, 0, 0), 2)
cv2.imwrite("./cube1_true.png", img1)

import cv2
import numpy as np

img = cv2.imread('cube.png')
edges = edges = cv2.Canny(img, 100, 200)
#~ print(edges)

minLineLength = 100
maxLineGap = 5
rho = 10
theta = 2
lines = cv2.HoughLinesP(edges,rho,theta * np.pi/180,100,minLineLength,maxLineGap)
print(lines)

new_lines = []
for line in lines:
    for x1,y1,x2,y2 in line:
        new_lines.append((x1, y1, x2, y2))
print(new_lines)



cv2.imwrite('houghlines5.jpg',img)

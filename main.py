from itertools import combinations
from pyquaternion import Quaternion

import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from numpy.linalg import norm


def dist_to_line(l, p):
    """
    >>> p = np.array((1, 2))
    >>> l1 = np.array((-7.51, 8))
    >>> l2 = np.array((21.3, 15.6))
    >>> print(dist_to_line(l1, l2, p))
    """
    l1 = np.array([l[0], l[1]])
    l2 = np.array([l[2], l[3]])
    d = norm(np.cross(l2 - l1, l1 - p)) / norm(l2 - l1)
    return d


def get_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imblue = cv2.medianBlur(gray, 5)
    #~ imblue = cv2.blur(imblue,(8,8))
    #~ edges = cv2.Canny(gray, 100, 200)

    # ==============
    edges = cv2.adaptiveThreshold(imblue, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                  cv2.THRESH_BINARY, 11, 2)
    edges = 255 - edges

    #~ kernel = np.ones((3,3),np.uint8)
    #~ edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    #~ edges = cv2.dilate(edges,kernel,iterations = 1)
    plt.imshow(edges, cmap='gray')
    plt.show()
    # ================

    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength,
                            maxLineGap)

    return list(map(lambda line: line[0], lines))

def intersect_union(l1, l2):
    v1 = [l1[2] - l1[0], l1[3] - l1[1]]
    v2 = [l2[2] - l2[0], l2[3] - l2[1]]
    degree = ( math.atan2(v1[1],v1[0]) + math.atan2(v2[1],v2[0]) ) / 2
    q = Quaternion(axis=(0.0, 0.0, 1.0), radians=degree)
    v12 = np.array([norm([l1[0], l1[1], l2[2], l2[3]]) / 2, 0, 0])
    offset = q.rotate(v12)
    base = np.array([(l1[0]+l1[2] + l2[0]+l2[2]) / 4, (l1[1]+l1[3] + l2[1]+l2[3]) / 4, 0])
    #~ print(offset, base)
    start = base - offset
    last = base + offset
    return list(map(lambda x : int(round(x)), [start[0], start[1], last[0], last[1]]))

def union_x(l1, l2):
    if l1[2] < l1[0]:
        return union_x([l1[2], l1[3], l1[0], l1[1]], l2)
    if l2[2] < l2[0]:
        return union_x(l1, [l2[2], l2[3], l2[0], l2[1]])
    if l2[0] < l1[0]:
        return union_x(l2, l1)
    if l1[0] < l2[2]:
        return intersect_union(l1, l2)
    return l1

def union_y(l1, l2):
    if l1[3] < l1[1]:
        return union_y([l1[2], l1[3], l1[0], l1[1]], l2)
    if l2[3] < l2[1]:
        return union_y(l1, [l2[2], l2[3], l2[0], l2[1]])
    if l2[1] < l1[1]:
        return union_y(l2, l1)
    if l1[1] < l2[3]:
        return intersect_union(l1, l2)
    return l1

def union(l1, l2):
    lx = union_x(l1, l2)
    ly = union_y(l1, l2)
    vx = np.array([lx[2] - lx[0], lx[3] - lx[1]])
    vy = np.array([ly[2] - ly[0], ly[3] - ly[1]])
    if norm(vx) < norm(vy):
        return ly
    return lx


def verify_and_delete(i1, i2, d_lines):
    if i1 in d_lines and i2 in d_lines:
        l1 = d_lines[i1]
        l2 = d_lines[i2]

        v1 = [l1[2] - l1[0], l1[3] - l1[1]]
        v2 = [l2[2] - l2[0], l2[3] - l2[1]]
        cos_a = np.dot(v1, v2) / (norm(v1) * norm(v2))

        if abs(cos_a) > 0.9:
            if dist_to_line(l1, [l2[0], l2[1]]) < 10 and \
                            dist_to_line(l1, [l2[2], l2[3]]) < 10:
                d_lines[i1] = union(l1, l2)
                del d_lines[i2]


def pretty_show(img, lines):
    img_with_lines = img.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(img_with_lines, (x1, y1), (x2, y2), (2, 0, 0), 3)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_with_lines, cmap='gray')
    plt.title('Image with Lines'), plt.xticks([]), plt.yticks([])

    plt.show()


if __name__ == "__main__":
    img = cv2.imread("cubicrgb.png")
    # print(len(img[0][0]))
    lines = get_lines(img)
    d_lines = {i: line for i, line in enumerate(lines)}
    combs = combinations(range(len(lines)), 2)
    for comb in combs:
        verify_and_delete(comb[0], comb[1], d_lines)
        # print(lines[comb[0]])
        pass
    print(len(d_lines))
    lines_orig = lines
    lines = [val for val in d_lines.values()]
    pretty_show(img, lines)

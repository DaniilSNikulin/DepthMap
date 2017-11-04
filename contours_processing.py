from matplotlib import pyplot as plt
from itertools import combinations

import cv2

from image_processing import img_edged
from utils import dist
import numpy as np

close_points_dist = 30


def contours_are_close(cnt1, cnt2):
    for p1 in cnt1:
        for p2 in cnt2:
            # print(p1, p2)
            if dist(p1[0], p2[0]) < close_points_dist:
                return True

    return False


def ordered_merged_contours(img):
    edged = img_edged(img.copy())

    img_gray, contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                                             cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = contours[:10]
    for i, cnt in enumerate(contours):
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        contours[i] = approx

    # print(len(contours))
    for i in range(len(contours) - 1, -1, -1):
        # print("\ni", i)
        for j in range(i + 1, len(contours)):
            # print(j, end="")
            if contours[j] is None:
                continue
            if contours_are_close(contours[i], contours[j]):
                contours[i] = np.concatenate((contours[i], contours[j]))
                contours[j] = None
    # print()

    # print(len(contours), len(contours_new))

    # print(len(contours))


    contours = list(filter(lambda x: x is not None, contours))
    # print(len(contours))
    contours = [cv2.convexHull(cnt) for cnt in contours]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for i in range(len(contours)):
        if contours[i] is None:
            continue
        # print("cnt", cv2.arcLength(contours[i], True),
        #       cv2.contourArea(contours[i]))
        if cv2.arcLength(contours[i], True) < 200 or cv2.contourArea(
                contours[i]) < 10000:
            contours[i] = None

    contours = list(filter(lambda x: x is not None, contours))
    # for i in range(len(contours)):
    #     img_cpy = img.copy()
    #     cv2.drawContours(img_cpy, contours, i, (0, 255, 0), 3)
    #     plt.subplot(111), plt.imshow(img_cpy)
    #     plt.show()

    return contours


def max_contour(img):
    edged = img_edged(img.copy())

    img_gray, contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                                             cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = contours[:10]
    for i, cnt in enumerate(contours):
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        contours[i] = approx

    combs = combinations(range(len(contours)), 2)
    contours_new = []
    for comb in combs:
        cnt1 = contours[comb[0]]
        cnt2 = contours[comb[1]]
        if contours_are_close(cnt1, cnt2):
            contours_new.append(np.concatenate((cnt1, cnt2)))
        else:
            contours_new.append(cnt1)
            contours_new.append(cnt2)

    # print(len(contours), len(contours_new))
    contours = contours_new
    # print(len(contours))

    c = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    return hull

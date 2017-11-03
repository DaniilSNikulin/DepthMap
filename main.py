import math
from copy import deepcopy
from itertools import combinations

import cv2

from matplotlib import pyplot as plt

from mid_point import *

close_points_dist = 30


def dist(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))


def contours_are_close(cnt1, cnt2):
    for p1 in cnt1:
        for p2 in cnt2:
            # print(p1, p2)
            if dist(p1[0], p2[0]) < close_points_dist:
                return True

    return False


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


def union_x(l1, l2):
    if l1[2] < l1[0]:
        return union_x([l1[2], l1[3], l1[0], l1[1]], l2)
    if l2[2] < l2[0]:
        return union_x(l1, [l2[2], l2[3], l2[0], l2[1]])
    if l2[0] < l1[0]:
        return union_x(l2, l1)
    if l1[2] < l2[2]:
        return [l1[0], l1[1], l2[2], l2[3]]
    return l1


def union_y(l1, l2):
    if l1[3] < l1[1]:
        return union_y([l1[2], l1[3], l1[0], l1[1]], l2)
    if l2[3] < l2[1]:
        return union_y(l1, [l2[2], l2[3], l2[0], l2[1]])
    if l2[1] < l1[1]:
        return union_y(l2, l1)
    if l1[3] < l2[3]:
        return [l1[0], l1[1], l2[2], l2[3]]
    return l1


def union(l1, l2):
    lx = union_x(l1, l2)
    ly = union_y(l1, l2)
    vx = np.array([lx[2] - lx[0], lx[3] - lx[1]])
    vy = np.array([ly[2] - ly[0], ly[3] - ly[1]])
    if norm(vx) < norm(vy):
        return ly
    return lx


def cut_cube(img):
    m_counter = max_contour(img)
    x, y, w, h = cv2.boundingRect(m_counter)

    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # plt.imshow(img)
    # plt.show()
    img_contours = img.copy()

    return y - 10, y + h + 10, x - 10, x + w + 10, img_contours[
                                                   y - 10:(y + h + 10),
                                                   x - 10:(x + w + 10)]


def cut_contour(img, counter):
    cnted_image = img.copy()

    mask = np.ones(cnted_image.shape[:2], dtype="uint8") * 255

    # Draw the contours on the mask
    cv2.drawContours(mask, [counter], -1, 0, -1)
    mask = 255 - mask

    # remove the contours from the image and show the resulting images
    cnted_image = cv2.bitwise_and(cnted_image, cnted_image, mask=mask)
    return cnted_image


def cube_contour(img):
    edged = img_edged(img.copy())
    img_gray, contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE,
                                             cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    return hull


def img_edged(img):
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    # edges = cv2.Canny(gray, 100, 200)

    # ==============
    imblue = cv2.medianBlur(gray, 9)
    # imblue = cv2.blur(imblue, (8, 8))

    edges = cv2.adaptiveThreshold(imblue, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                  cv2.THRESH_BINARY, 11, 2)
    edges = 255 - edges
    return edges


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
    # plt.imshow(edges, cmap='gray')
    # plt.show()
    # ================
    edged = img_edged(img)

    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edged.copy(), 1, np.pi / 180, 100,
                            minLineLength,
                            maxLineGap)

    return list(map(lambda line: line[0], lines))


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


def vanish_points(lines):
    d_lines = {i: line for i, line in enumerate(lines)}
    combs = combinations(range(len(lines)), 2)
    d_interseptions = dict()
    for comb in combs:
        l1 = lines[comb[0]]
        l2 = lines[comb[1]]

        p_intersept = intersection(l1, l2)
        if p_intersept:
            d_interseptions[p_intersept] = (l1, l2)

    max_x, max_y = -float("inf"), -float("inf")
    min_x, min_y = float("inf"), float("inf")
    vp_bottom = []
    vp_top = []
    vp_left = []
    vp_right = []
    for point in d_interseptions.keys():

        # print("point intersept", point)

        x, y = point
        if x < min_x:
            vp_left = point
            min_x = x
        if x > max_x:
            vp_right = point
            max_x = x
        if y < min_y:
            vp_bottom = point
            min_y = y
        if y > max_y:
            vp_top = point
            max_y = y
    # print("durty vps", vp_bottom, vp_left, vp_top, vp_right)

    vps = [vp_bottom, vp_left, vp_top, vp_right]
    combs = combinations(range(4), 2)
    min_norm = float("inf")
    vp_looser_i = 0
    for comb in combs:
        vp1 = np.array(vps[comb[0]])
        vp2 = np.array(vps[comb[1]])

        if norm(vp1 - vp2) < min_norm:
            vp_looser_i = comb[1]
            min_norm = norm(vp1 - vp2)

    del vps[vp_looser_i]
    return vps


def mean_point(lines, vps):
    acc_x = 0
    acc_y = 0
    counter = 0
    for x1, y1, x2, y2 in lines:
        acc_x += x1 + x2
        acc_y += y1 + y2
        counter += 2
    return [int(round(acc_x / counter)), int(round(acc_y / counter))]


def img_with_lines(img, lines):
    img_lines = img.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(img_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return img_lines


def createGradient(img, vps, st):
    def depth_pixel(pix):
        np_pix = np.array(pix)

        dist1 = dist_to_line(l12, pix)
        dist2 = dist_to_line(l23, pix)
        dist3 = dist_to_line(l31, pix)

        dist = dist1 / dist_1st
        dist = min(dist, dist2 / dist_2st)
        dist = min(dist, dist3 / dist_3st)
        return int(max(min(round(dist * 255), 255), 0))

    vp1 = vps[0]
    vp2 = vps[1]
    vp3 = vps[2]
    np_vp1 = np.array(vp1)
    np_vp2 = np.array(vp2)
    np_vp3 = np.array(vp3)
    np_st = np.array(st)

    l12 = [vp1[0], vp1[1], vp2[0], vp2[1]]
    l23 = [vp2[0], vp2[1], vp3[0], vp3[1]]
    l31 = [vp3[0], vp3[1], vp1[0], vp1[1]]

    dist_1st = dist_to_line(l12, st)
    dist_2st = dist_to_line(l23, st)
    dist_3st = dist_to_line(l31, st)

    grad = [[depth_pixel([j + 1, i + 1]) for j in range(len(img[0]))] for i in
            range(len(img))]
    grad = np.array(grad)
    return grad


def img_gradient(img, vps, mean_p):
    grad_img = img.copy()
    grad_img = createGradient(grad_img, vps, mean_p)
    return grad_img


def pretty_show(img, img_edged, img_durty_lines, img_lines, img_grad):
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(232), plt.imshow(img_edged, cmap='gray')
    # plt.title('Edged'), plt.xticks([]), plt.yticks([])
    # plt.subplot(233), plt.imshow(img_durty_lines, cmap='gray')
    # plt.title('Durty lines'), plt.xticks([]), plt.yticks([])
    # plt.subplot(234), plt.imshow(img_lines, cmap='gray')
    # plt.title('Clear lines'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_grad, cmap='gray')
    plt.title('Gradient'), plt.xticks([]), plt.yticks([])

    plt.show()


if __name__ == "__main__":

    origin_img = cv2.imread("./images/c.jpg")

    x_base, x_last, y_base, y_last, img = cut_cube(origin_img)
    lines = get_lines(img)
    lines_orig = deepcopy(lines)
    d_lines = {i: line for i, line in enumerate(lines)}
    combs = combinations(range(len(lines)), 2)
    for comb in combs:
        verify_and_delete(comb[0], comb[1], d_lines)

    lines = [val for val in d_lines.values()]
    print(len(d_lines))

    # ================================

    vps = vanish_points(lines)
    # mean_p = mean_point(lines, vps)
    mean_p = midp(lines)
    img_grad = img_gradient(img, vps, mean_p)
    # img_grad = img
    cntr = cube_contour(img)
    cntr = max_contour(img)
    cntred_img = cut_contour(img_grad.copy(), cntr)
    back_ground = origin_img.copy()
    len_a = len(back_ground)
    for i in range(x_base, x_last, 1):
        for j in range(y_base, y_last, 1):
            if cntred_img[i - x_base][j - y_base] > 5:
                tmp = cntred_img[i-x_base][j-y_last]
                back_ground[i][j] = tmp, tmp, tmp
    # cntred_img = img
    pretty_show(origin_img, img_edged(img), img_with_lines(img, lines_orig),
                img_with_lines(img, lines), back_ground)

    # ~ print("clear vps", vps)
    print("mean_p", mean_p)

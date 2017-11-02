from itertools import combinations

import cv2
import numpy as np
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


def intersection(l1, l2):
    def line(p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0] * p2[1] - p2[0] * p1[1])
        return A, B, -C

    l1 = line([l1[0], l1[1]], [l1[2], l1[3]])
    l2 = line([l2[0], l2[1]], [l2[2], l2[3]])
    D = l1[0] * l2[1] - l1[1] * l2[0]
    Dx = l1[2] * l2[1] - l1[1] * l2[2]
    Dy = l1[0] * l2[2] - l1[2] * l2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


def get_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 100, 200)

    # ==============
    imblue = cv2.medianBlur(gray, 9)
    # imblue = cv2.blur(imblue, (8, 8))

    edges = cv2.adaptiveThreshold(imblue, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                  cv2.THRESH_BINARY, 11, 2)
    edges = 255 - edges
    print(len(edges), len(edges[0]))
    # plt.imshow(edges, cmap='gray')
    # plt.show()
    # ================

    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength,
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
                del d_lines[i2]


def vanish_points(lines):
    d_lines = {i: line for i, line in enumerate(lines)}
    combs = combinations(range(len(lines)), 2)
    d_interseptions = dict()
    for comb in combs:
        l1 = lines[comb[0]]
        l2 = lines[comb[1]]

        p_intersept = intersection(l1, l2)
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
    # print("vps", vp_bottom, vp_left, vp_top, vp_right)

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


def mean_point(lines):
    acc_x = 0
    acc_y = 0
    counter = 0
    for x1, y1, x2, y2 in lines:
        acc_x += x1 + x2
        acc_y += y1 + y2
        counter += 2
    return [int(round(acc_x / counter)), int(round(acc_y / counter))]


def pretty_show(img, lines):
    img_with_lines = img.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(img_with_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_with_lines, cmap='gray')
    plt.title('Image with Lines'), plt.xticks([]), plt.yticks([])

    plt.show()


def createGradient(img, vps, st):
    def depth_pixel(vp1, vp2, vp3, st, pix):
        np_vp1 = np.array(vp1)
        np_vp2 = np.array(vp2)
        np_vp3 = np.array(vp3)
        np_st = np.array(st)
        np_pix = np.array(pix)

        l12 = [vp1[0], vp1[1], vp2[0], vp2[1]]
        l23 = [vp2[0], vp2[1], vp3[0], vp3[1]]
        l31 = [vp3[0], vp3[1], vp1[0], vp1[1]]
        dist1 = dist_to_line(l12, pix)
        dist2 = dist_to_line(l23, pix)
        dist3 = dist_to_line(l31, pix)
        if norm(np_pix - np_vp3) < dist1 or norm(
                        np_pix - np_vp2) < dist3 or norm(
                    np_pix - np_vp1) < dist2:
            return 0

        dist = dist1 / dist_to_line(l12, st)
        dist = min(dist, dist2 / dist_to_line(l23, st))
        dist = min(dist, dist3 / dist_to_line(l31, st))
        dost = dist - 1
        return int(max(min(round(dist * 255), 255), 0))

    grad = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(250, 300):
        for j in range(300):
            pix = [i + 1, j + 1]
            grad[i][j] = depth_pixel(vps[0], vps[1], vps[2], st, pix)
    return grad


def gradient_show(img, vps, mean_p):
    from scipy.interpolate import interp2d

    grad_img = img.copy()
    grad_img = createGradient(grad_img, vps, mean_p)
    # xs = [vp[0] for vp in vps]
    # ys = [vp[1] for vp in vps]
    # zs = [0 for vp in vps]
    # xs.append(mean_p[0])
    # ys.append(mean_p[1])
    # zs.append(255)
    # f = interp2d(xs, ys, zs, kind='linear')
    # for i in range(len(grad_img)):
    #     for j in range(len(grad_img[0])):
    #         grad_img[i, j] = f(i, j)
    plt.imshow(grad_img, cmap='gray')
    plt.show()


if __name__ == "__main__":
    img = cv2.imread("cube.png")
    # print(len(img[0][0]))
    lines = get_lines(img)
    d_lines = {i: line for i, line in enumerate(lines)}
    combs = combinations(range(len(lines)), 2)
    for comb in combs:
        verify_and_delete(comb[0], comb[1], d_lines)
        # print(lines[comb[0]])
        pass

    lines = [val for val in d_lines.values()]
    # print(len(lines))
    # pretty_show(img, lines)

    # ================================

    vps = vanish_points(lines)
    mean_p = mean_point(lines)
    gradient_show(img, vps, mean_p)

    print("clear vps", vps)

import getopt
from copy import deepcopy

import cv2
import sys
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

    # for i in range(len(contours)):
    #     img_cpy = img.copy()
    #     cv2.drawContours(img_cpy, contours, i, (0, 255, 0), 3)
    #     plt.subplot(111), plt.imshow(img_cpy)
    #     plt.show()

    return contours


def get_cubes(img):
    contours = ordered_merged_contours(img)
    # for cnt in contours:
    # m_counter = max_contour(img)
    # x, y, w, h = cv2.boundingRect(cnt)
    x, y, w, h = cv2.boundingRect(contours[0])
    y_base = max(y-10, 0)
    y_last = min(y + h + 10, len(img))
    x_base = max(x - 10, 0)
    x_last = min(x + w + 10, len(img[0]))
    img_contours = img.copy()

    return y_base, y_last, x_base, x_last, img_contours[y_base:y_last,x_base:x_last]


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
    # edges = cv2.dilate(edges,None)
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

    minLineLength = 60
    maxLineGap = 10
    lines = cv2.HoughLinesP(edged.copy(), 1, np.pi / 180, threshold=50,
                            minLineLength=minLineLength,
                            maxLineGap=maxLineGap)
    lines = list(map(lambda line: line[0], lines))
    lines_orig = deepcopy(lines)

    d_lines = {i: line for i, line in enumerate(lines)}
    combs = combinations(range(len(lines)), 2)
    for comb in combs:
        verify_and_delete(comb[0], comb[1], d_lines)

    lines = [val for val in d_lines.values()]

    lines = sorted(lines, key=(lambda line: dist([line[0], line[1]],
                                                 [line[2], line[3]])),
                   reverse=True)

    return lines_orig, lines[:9]


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


def create_floor_grad(cubes, length_y, length_x, vps):
    def choice_vp(vps):
        new_vps = deepcopy(vps)
        max_y = -float("inf")
        max_y_ind = 0
        for i in range(3):
            if max_y < new_vps[i][1]:
                max_y = new_vps[i][1]
                max_y_ind = i
        del new_vps[max_y_ind]
        return new_vps

    def floor_base_value(grad):
        amount_acc = 5
        count_acc = 1
        acc = 0
        y_max = 1
        x_max = 1
        for y_i in range(len(grad) - 1, 0, -1):
            if 0 < grad[y_i].max():
                x_max = grad[y_i].argmax()
                y_max = y_i
                acc += grad[y_i].max()
                count_acc += 1
            if count_acc > amount_acc:
                break
        acc = (acc / amount_acc)
        y_max = y_max + amount_acc
        return x_max, y_max, acc

    def create_start_floor(img, vps):
        floor = img
        vps = choice_vp(vps)
        y_base = (vps[0][1] + vps[1][1]) / 2

        length_y = len(floor)
        length_x = len(floor[0])
        x_max, y_max, base_value = floor_base_value(floor)
        max_value = math.sqrt(float((length_y-y_base)*(length_y-y_base)) / float((y_max - y_base)*(y_max - y_base)))*base_value
        diff = max_value - 255
        for y_i in range(length_y):
            for x_i in range(length_x):
                if y_i < y_base:
                    floor[y_i][x_i] = 0
                else:
                    y = y_i - y_base
                    length = y_max - y_base
                    floor[y_i][x_i] = float(max(math.sqrt(float(y*y) / float(length*length))*base_value - diff, 0))
        return floor, diff

    first_cube = cubes[0]
    mask = np.ones((length_y, length_x), dtype="float") * 0
    for i in range(first_cube.x_base, first_cube.x_last, 1):
        for j in range(first_cube.y_base, first_cube.y_last, 1):
            if first_cube.img[i - first_cube.x_base][j - first_cube.y_base] > 0:
                tmp = first_cube.img[i - first_cube.x_base][j - first_cube.y_last]
                mask[i][j] = tmp

    img_backgrnd, diff = create_start_floor(mask, vps)

    for cube in cubes:
        x_max, y_max, base_value = floor_base_value(cube.img)
        diff = base_value - img_backgrnd[cube.x_base + y_max][0]
        print(diff)
        for i in range(cube.x_base, cube.x_last, 1):
            for j in range(cube.y_base, cube.y_last, 1):
                if cube.img[i - cube.x_base][j - cube.y_base] > 0:
                    tmp = float(cube.img[i - cube.x_base][j - cube.y_last]) - diff
                    img_backgrnd[i][j] = float(max(tmp, 0))
    img_backgrnd = img_backgrnd / img_backgrnd.max() * 255
    img_backgrnd = img_backgrnd.astype(int)
    return img_backgrnd



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


def mean_point(lines):
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


def pretty_show(img, img_edged, img_durty_lines, img_lines, img_grad,
                img_background):
    plt.subplot(231), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(232), plt.imshow(img_edged, cmap='gray')
    plt.title('Edged'), plt.xticks([]), plt.yticks([])
    plt.subplot(233), plt.imshow(img_durty_lines, cmap='gray')
    plt.title('Dirty lines'), plt.xticks([]), plt.yticks([])
    plt.subplot(234), plt.imshow(img_lines, cmap='gray')
    plt.title('Clear lines'), plt.xticks([]), plt.yticks([])
    plt.subplot(235), plt.imshow(img_grad, cmap='gray')
    plt.title('Gradient'), plt.xticks([]), plt.yticks([])
    plt.subplot(236), plt.imshow(img_background, cmap='gray')
    plt.title('Backgroung'), plt.xticks([]), plt.yticks([])

    plt.show()

class Cube:
    def __init__(self, offset, x_base, x_last, y_base, y_last, img):
        self.offset = offset
        self.x_base = x_base
        self.x_last = x_last
        self.y_base = y_base
        self.y_last = y_last
        self.img = img


if __name__ == "__main__":

    # **************************************888
    argv = sys.argv[1:]
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('main.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('main.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    if inputfile == '':
        print('Error: you need usage "main.py -i <inputfile> -o <outputfile>"')
        exit(0)
    # ********************************************8

    origin_img = cv2.imread(inputfile)
    if origin_img is None:
        print("Error: the file name is not valid")
        exit(0)
    n_cubes = 5

    x_base, x_last, y_base, y_last, img = get_cubes(origin_img)
    lines_orig, lines = get_lines(img)
    print(len(lines))

    # ================================

    vps = vanish_points(lines)
    # mean_p = mean_point(lines, vps)
    mean_p = midp(lines)
    # cntr = cv2.convexHull(cntr)
    # print(len(cntr))
    #~ img_grad = img
    img_grad = img_gradient(img, vps, mean_p)
    #~ img_grad = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cntr = max_contour(img)
    cntred_img = cut_contour(img_grad.copy(), cntr)
    # img_background = origin_img.copy()
    # len_a = len(img_background)

    cubes = [Cube(10, x_base, x_last, y_base, y_last, cntred_img)]

    length_y, length_x = origin_img.shape[:2]
    print(length_y, length_x)
    img_backgrnd = create_floor_grad(cubes, length_y, length_x, vps)

    #~ mask = np.ones(origin_img.shape[:2], dtype="uint8") * 0
    #~ for i in range(x_base, x_last, 1):
        #~ for j in range(y_base, y_last, 1):
            #~ if cntred_img[i - x_base][j - y_base] > 0:
                #~ tmp = cntred_img[i - x_base][j - y_last]
                #~ mask[i][j] = tmp

    #~ img_backgrnd, diff = create_floor_grad(mask, vps)

    #~ for i in range(x_base, x_last, 1):
        #~ for j in range(y_base, y_last, 1):
            #~ if cntred_img[i - x_base][j - y_base] > 0:
                #~ tmp = cntred_img[i - x_base][j - y_last] - diff
                #~ img_backgrnd[i][j] = min(max(tmp, 0), 255)

    # cntred_img = img
    # img_backgrnd = 255 - img_backgrnd
    pretty_show(origin_img, img_edged(img), img_with_lines(img, lines_orig),
                img_with_lines(img, lines), cntred_img, img_backgrnd)

    # ~ print("clear vps", vps)
    print("mean_p", mean_p)

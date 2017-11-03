import cv2
from copy import deepcopy
import numpy as np

import math


def img_edged(img):
    assert len(img[0][0]) == 3, img[0][0]
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    # edges = cv2.Canny(gray, 70, 150)

    # ==============
    imblue = cv2.medianBlur(gray, 9)

    edges = cv2.adaptiveThreshold(imblue, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                  cv2.THRESH_BINARY, 11, 2)
    edges = 255 - edges

    return edges


def img_with_lines(img, lines):
    img_lines = img.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(img_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return img_lines


def img_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def img_contoured(img, counter):
    cnted_image = img.copy()

    mask = np.ones(cnted_image.shape[:2], dtype="uint8") * 255

    # Draw the contours on the mask
    cv2.drawContours(mask, [counter], -1, 0, -1)
    mask = 255 - mask

    # remove the contours from the image and show the resulting images
    cnted_image = cv2.bitwise_and(cnted_image, cnted_image, mask=mask)
    return cnted_image


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
        max_value = math.sqrt(
            float((length_y - y_base) * (length_y - y_base)) / float(
                (y_max - y_base) * (y_max - y_base))) * base_value
        diff = max_value - 255
        for y_i in range(length_y):
            for x_i in range(length_x):
                if y_i < y_base:
                    floor[y_i][x_i] = 0
                else:
                    y = y_i - y_base
                    length = y_max - y_base
                    floor[y_i][x_i] = float(max(math.sqrt(float(y * y) / float(
                        length * length)) * base_value - diff, 0))
        return floor, diff

    first_cube = cubes[0]
    mask = np.ones((length_y, length_x), dtype="float") * 0
    for i in range(first_cube.x_base, first_cube.x_last, 1):
        for j in range(first_cube.y_base, first_cube.y_last, 1):
            if first_cube.img_grad_cntred[i - first_cube.x_base][
                        j - first_cube.y_base] > 0:
                tmp = first_cube.img_grad_cntred[i - first_cube.x_base][
                    j - first_cube.y_last]
                mask[i][j] = tmp

    img_backgrnd, diff = create_start_floor(mask, vps)

    for cube in cubes:
        x_max, y_max, base_value = floor_base_value(cube.img_grad_cntred)
        diff = base_value - img_backgrnd[cube.x_base + y_max][0]
        # print(diff)
        for i in range(cube.x_base, cube.x_last, 1):
            for j in range(cube.y_base, cube.y_last, 1):
                if cube.img_grad_cntred[i - cube.x_base][j - cube.y_base] > 0:
                    tmp = float(
                        cube.img_grad_cntred[i - cube.x_base][j - cube.y_last]) - diff
                    img_backgrnd[i][j] = float(max(tmp, 0))
    img_backgrnd = img_backgrnd / img_backgrnd.max() * 255
    img_backgrnd = img_backgrnd.astype(int)
    return img_backgrnd

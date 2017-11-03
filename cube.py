from itertools import combinations

import cv2
import numpy as np
from numpy.linalg import norm
from copy import deepcopy

from contours_processing import max_contour
from image_processing import img_edged, img_gray, img_contoured, img_with_lines
from utils import dist_to_line, dist, intersection
from mid_point import midp


class Cube:
    def __init__(self, img, x_base, x_last, y_base, y_last, offset=10):
        self.img = img
        self.x_base = x_base
        self.x_last = x_last
        self.y_base = y_base
        self.y_last = y_last
        self.offset = offset

        self.width = len(self.img)
        self.height = len(self.img[0])

        self.cntr = max_contour(self.img)
        self.img_edged = img_edged(self.img)
        self.img_gray = img_gray(self.img)
        self.dirty_lines, self.clear_lines = self.__get_lines()
        print("n_lines", len(self.clear_lines))
        self.mid_point = None
        try:
            self.mid_point = midp(self.clear_lines)
        except Exception:
            print(">>>>> mid_p failed")
            self.mid_point = self.__mean_point()

        self.vps = self.__vanish_points()
        self.img_grad = self.__createGradient()
        # self.img_grad = self.img
        self.img_cntred = img_contoured(self.img_gray, self.cntr)
        self.img_grad_cntred = img_contoured(self.img_grad, self.cntr)
        self.img_lines = img_with_lines(self.img, self.clear_lines)

    def __get_lines(self):
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

        def verify_and_delete(i1, i2, d_lines):
            if i1 in d_lines and i2 in d_lines:
                l1 = d_lines[i1]
                l2 = d_lines[i2]

                v1 = [l1[2] - l1[0], l1[3] - l1[1]]
                v2 = [l2[2] - l2[0], l2[3] - l2[1]]
                cos_a = np.dot(v1, v2) / (norm(v1) * norm(v2))

                if abs(cos_a) > 0.9:
                    if dist_to_line(l1, [l2[0], l2[1]]) < 15 and \
                                    dist_to_line(l1, [l2[2], l2[3]]) < 15:
                        d_lines[i1] = union(l1, l2)
                        del d_lines[i2]

        edged = self.img_edged

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

    def __vanish_points(self):
        lines = self.clear_lines
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

    def __mean_point(self):
        lines = self.clear_lines
        acc_x = 0
        acc_y = 0
        counter = 0
        for x1, y1, x2, y2 in lines:
            acc_x += x1 + x2
            acc_y += y1 + y2
            counter += 2
        return [int(round(acc_x / counter)), int(round(acc_y / counter))]

    def __createGradient(self):
        img = self.img
        vps = self.vps
        st = self.mid_point

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

        grad = [[depth_pixel([j + 1, i + 1]) for j in range(len(img[0]))] for i
                in
                range(len(img))]
        grad = np.array(grad)
        return grad

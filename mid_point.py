from numpy.linalg import norm
import numpy as np
from itertools import combinations


def cross_pp(p1, p2, p3, p4):
    if p1[0] == p2[0]:
        x = p1[0]
        y = ((x - p4[0])/(p3[0] - p4[0]))*(p3[1]-p4[1]) + p4[1]
    elif p3[0] == p4[0]:
        x = p3[0]
        y = ((x - p2[0]) / (p1[0] - p2[0])) * (p1[1] - p2[1]) + p2[1]
    else:
        c1 = (p1[1] - p2[1])/(p1[0] - p2[0])
        c2 = (p3[1] - p4[1])/(p3[0] - p4[0])
        x = (p3[1] - p1[1] + p1[0]*c1 - p3[0]*c2)/(c1-c2)
        y = p1[1] + c1*(x-p1[0])
    return [x, y]


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


def vanish_points1(lines):
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
    #return vps
    return [[vps[0], d_interseptions[vps[0]][0], d_interseptions[vps[0]][1]],
            [vps[1],d_interseptions[vps[1]][0], d_interseptions[vps[1]][1]],
            [vps[2], d_interseptions[vps[2]][0], d_interseptions[vps[2]][1]]]


def distance(v, p, t):
    if p[1] == v[1]:
        return t[1] - v[1]
    elif p[0] == v[0]:
        return t[0]-v[0]
    else:
        return (t[1]-v[1])/(p[1]-v[1]) - (t[0]-v[0])/(p[0]-v[0])


def eq(l1, l2):
    if l1[0] == l2[0] and l1[1] == l2[1] and l1[2] == l2[2] and l1[3] == l2[3]:
        return True
    else:
        return False


def check_cont(vp, line, lines):
    part1 = 0
    part2 = 0
    for l in lines:
        if not eq(l, line):
            for i in range(2):
                if distance(vp, [line[0], line[1]], [l[i*2], l[i*2 + 1]]) > 0:
                    part1 += 1
                else:
                    part2 += 1
    if part1 == 0:
        return 1
    if part2 == 0:
        return 2
    return 0


def reroute(v1, v2, v3):
    if v1[1] > v2[1] and v1[1] > v3[1]:
        if v2[0] < v3[0]:
            return [1, 2, 0]
        else:
            return [2, 1, 0]
    if v2[1] > v1[1] and v2[1] > v3[1]:
        if v1[0] < v3[0]:
            return [0, 2, 1]
        else:
            return [2, 0, 1]
    else:
        if v1[0] < v2[0]:
            return [0, 1, 2]
        else:
            return [1, 0, 2]

#
# def find_contour(vps, lines):
#     pass


def zip_lines(lines):
    for i in range(len(lines)):
        lines[i] = [(3 * lines[i][0] + lines[i][2]) / 4,
                    (3 * lines[i][1] + lines[i][3]) / 4,
                    (lines[i][0] + 3 * lines[i][2]) / 4,
                    (lines[i][1] + 3 * lines[i][3]) / 4]
    return lines


def midp(lines):
    vps_n_lines = vanish_points1(lines)
    vps = [vps_n_lines[0][0], vps_n_lines[1][0], vps_n_lines[2][0]]
    ind = reroute(vps[0], vps[1], vps[2])
    vps = [vps[ind[0]], vps[ind[1]], vps[ind[2]]]
    lines_for_vps = [vps_n_lines[ind[0]][1], vps_n_lines[ind[0]][2],
                     vps_n_lines[ind[1]][1], vps_n_lines[ind[1]][2],
                     vps_n_lines[ind[2]][1], vps_n_lines[ind[2]][2]]
    liness = []
    for i in range(len(lines)):
        liness.append(lines[i])
    z_lines = zip_lines(liness)
    lines_for_vps = zip_lines(lines_for_vps)
    mid_lines = []
    cont_lines = []
    cont_signs = []
    con_check = []
    sig_mid = []
    for i in range(3):
        tr1 = check_cont(vps[i], lines_for_vps[2*i], z_lines)
        tr2 = check_cont(vps[i], lines_for_vps[2*i + 1], z_lines)
        if tr1 == 0:
            mid_lines.append(lines_for_vps[2*i])
            sig_mid.append(i)
        else:
            cont_lines.append(lines_for_vps[2*i])
            cont_signs.append(tr1)
            con_check.append(2 * i)
        if tr2 == 0:
            mid_lines.append(lines_for_vps[2*i + 1])
            sig_mid.append(i)
        else:
            cont_lines.append(lines_for_vps[2 * i + 1])
            cont_signs.append(tr2)
            con_check.append(2 * i + 1)
    if len(mid_lines) >= 2:
        mid_p = cross_pp([mid_lines[0][0], mid_lines[0][1]],
                         [mid_lines[0][2], mid_lines[0][3]],
                         [mid_lines[1][0], mid_lines[1][1]],
                         [mid_lines[1][2], mid_lines[1][3]])
        return [int(mid_p[0]), int(mid_p[1])]
    elif len(mid_lines) == 1:
        if sig_mid[0] == 0:
            if cont_signs[0] == 2:
                if cont_signs[1] == 2:
                    l = 0
                else:
                    l = 1
                k = con_check[0]
                p = cross_pp([lines_for_vps[2 + l][0], lines_for_vps[2 + l][1]],
                             [lines_for_vps[2 + l][2], lines_for_vps[2 + l][3]],
                             [lines_for_vps[k][0], lines_for_vps[k][1]],
                             [lines_for_vps[k][2], lines_for_vps[k][3]])
                mid_p = cross_pp([mid_lines[0][0], mid_lines[0][1]],
                                 [mid_lines[0][2], mid_lines[0][3]],
                                 vps[2],
                                 p)
                return [int(mid_p[0]), int(mid_p[1])]
            if cont_signs[0] == 1:
                k = con_check[0]
                if lines_for_vps[4][0] > lines_for_vps[5][0]:
                    p = cross_pp([lines_for_vps[4][0], lines_for_vps[4][1]],
                                 [lines_for_vps[4][2], lines_for_vps[4][3]],
                                 [lines_for_vps[k][0], lines_for_vps[k][1]],
                                 [lines_for_vps[k][2], lines_for_vps[k][3]])
                else:
                    p = cross_pp([lines_for_vps[5][0], lines_for_vps[5][1]],
                                 [lines_for_vps[5][2], lines_for_vps[5][3]],
                                 [lines_for_vps[k][0], lines_for_vps[k][1]],
                                 [lines_for_vps[k][2], lines_for_vps[k][3]])

                mid_p = cross_pp([mid_lines[0][0], mid_lines[0][1]],
                                 [mid_lines[0][2], mid_lines[0][3]],
                                 vps[1],
                                 p)
                return [int(mid_p[0]), int(mid_p[1])]
        if sig_mid[0] == 1:
            if cont_signs[2] == 2:
                if cont_signs[0] == 2:
                    l = 0
                else:
                    l = 1
                k = con_check[2]
                p = cross_pp([lines_for_vps[l][0], lines_for_vps[l][1]],
                             [lines_for_vps[l][2], lines_for_vps[l][3]],
                             [lines_for_vps[k][0], lines_for_vps[k][1]],
                             [lines_for_vps[k][2], lines_for_vps[k][3]])
                mid_p = cross_pp([mid_lines[0][0], mid_lines[0][1]],
                                 [mid_lines[0][2], mid_lines[0][3]],
                                 vps[2],
                                 p)
                return [int(mid_p[0]), int(mid_p[1])]
            if cont_signs[2] == 1:
                k = con_check[2]
                if lines_for_vps[4][0] < lines_for_vps[5][0]:
                    p = cross_pp([lines_for_vps[4][0], lines_for_vps[4][1]],
                                 [lines_for_vps[4][2], lines_for_vps[4][3]],
                                 [lines_for_vps[k][0], lines_for_vps[k][1]],
                                 [lines_for_vps[k][2], lines_for_vps[k][3]])
                else:
                    p = cross_pp([lines_for_vps[5][0], lines_for_vps[5][1]],
                                 [lines_for_vps[5][2], lines_for_vps[5][3]],
                                 [lines_for_vps[k][0], lines_for_vps[k][1]],
                                 [lines_for_vps[k][2], lines_for_vps[k][3]])
                mid_p = cross_pp([mid_lines[0][0], mid_lines[0][1]],
                                 [mid_lines[0][2], mid_lines[0][3]],
                                 vps[0],
                                 p)
                return [int(mid_p[0]), int(mid_p[1])]
        else:
            c_line = cont_lines[4]
            mid_l = mid_lines[0]
            if c_line[0] < mid_l[0]:
                if cont_signs[2] == 1:
                    l = 2
                else:
                    l = 3
                p = cross_pp([lines_for_vps[l][0], lines_for_vps[l][1]],
                             [lines_for_vps[l][2], lines_for_vps[l][3]],
                             [c_line[0], c_line[1]],
                             [c_line[2], c_line[3]])
                mid_p = cross_pp([mid_lines[0][0], mid_lines[0][1]],
                                 [mid_lines[0][2], mid_lines[0][3]],
                                 vps[0],
                                p)
                return [int(mid_p[0]), int(mid_p[1])]
            else:
                if cont_signs[0] == 1:
                    l = 0
                else:
                    l = 1
                p = cross_pp([lines_for_vps[l][0], lines_for_vps[l][1]],
                             [lines_for_vps[l][2], lines_for_vps[l][3]],
                             [c_line[0], c_line[1]],
                             [c_line[2], c_line[3]])
                mid_p = cross_pp([mid_lines[0][0], mid_lines[0][1]],
                                 [mid_lines[0][2], mid_lines[0][3]],
                                 vps[1],
                                 p)
                return [int(mid_p[0]), int(mid_p[1])]
    else:
        if lines_for_vps[4][0] < lines_for_vps[5][0]:
            l = 4
            r = 5
        else:
            l = 5
            r = 4
        if cont_signs[0] == 1:
            r2 = 0
        else:
            r2 = 1
        if cont_signs[2] == 1:
            l2 = 2
        else:
            l2 = 3
        p_l = cross_pp([lines_for_vps[l][0], lines_for_vps[l][1]],
                       [lines_for_vps[l][2], lines_for_vps[l][3]],
                       [lines_for_vps[l2][0], lines_for_vps[l2][1]],
                       [lines_for_vps[l2][2], lines_for_vps[l2][3]])
        p_r = cross_pp([lines_for_vps[r][0], lines_for_vps[r][1]],
                       [lines_for_vps[r][2], lines_for_vps[r][3]],
                       [lines_for_vps[r2][0], lines_for_vps[r2][1]],
                       [lines_for_vps[r2][2], lines_for_vps[r2][3]])
        mid_p = cross_pp(vps[0], p_l,
                         vps[1], p_r)
        return [int(mid_p[0]), int(mid_p[1])]

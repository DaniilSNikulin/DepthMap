import math
from numpy.linalg import norm
import numpy as np
from itertools import combinations
# midp(lines) возвращает midpoint


def dist(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))
#################################################


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


# пересечение
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


def bothcheck(l1, l2, l3):  # проверка что есть два ребра внутри угла от vp
    a2 = check_edgeinside(l1[0], [l1[1][0], l1[1][1]],
                          [l1[2][0], l1[2][1]], l2[1])
    b2 = check_edgeinside(l1[0], [l1[1][0], l1[1][1]],
                          [l1[2][0], l1[2][1]], l2[2])
    a3 = check_edgeinside(l1[0], [l1[1][0], l1[1][1]],
                          [l1[2][0], l1[2][1]], l3[1])
    b3 = check_edgeinside(l1[0], [l1[1][0], l1[1][1]],
                          [l1[2][0], l1[2][1]], l3[2])
    if (a2 and b2) or (a3 and b3):
        return True
    else:
        return False


# Вернуть 4 точки куба [p1, p2, p3, p4]
def cub(l1, l2):
    a1 = cross_pp([l1[2][0], l1[2][1]], [l1[2][2], l1[2][3]],
                  [l2[2][0], l2[2][1]], [l2[2][2], l2[2][3]])
    a2 = cross_pp([l1[2][0], l1[2][1]], [l1[2][2], l1[2][3]],
                  [l2[1][0], l2[1][1]], [l2[1][2], l2[1][3]])
    a3 = cross_pp([l1[1][0], l1[1][1]], [l1[1][2], l1[1][3]],
                  [l2[2][0], l2[2][1]], [l2[2][2], l2[2][3]])
    a4 = cross_pp([l1[1][0], l1[1][1]], [l1[1][2], l1[1][3]],
                  [l2[1][0], l2[1][1]], [l2[1][2], l2[1][3]])
    return [a1, a2, a3, a4]


def magic_func(l1, l2, l3, lines):
    cvadr = cub(l1, l2)
    min1 = dist([l3[1][2], l3[1][3]], cvadr[0])
    k = 0
    for i in range(1, 4):
        m = dist([l3[1][2], l3[1][3]], cvadr[i])
        if m < min1:
            min1 = m
            k = i
    q = 0
    min2 = dist([l3[2][2], l3[2][3]], cvadr[0])
    for i in range(1, 4):
        m = dist([l3[2][2], l3[2][3]], cvadr[i])
        if m < min2:
            min2 = m
            q = i
    pts = []
    for i in range(4):
        if i != k and i != q:
            pts.append(i)
    min1 = dist(cvadr[pts[0]], [lines[0][0], lines[0][1]])
    min2 = dist(cvadr[pts[1]], [lines[0][0], lines[0][1]])
    for i in range(3):
        for j in range(2):
            if dist(cvadr[pts[0]], [lines[i][2*j], lines[i][2*j+1]]) < min1:
                min1 = dist(cvadr[pts[0]], [lines[i][2*j], lines[i][2*j+1]])
            if dist(cvadr[pts[1]], [lines[i][2*j], lines[i][2*j+1]]) < min2:
                min2 = dist(cvadr[pts[1]], [lines[i][2*j], lines[i][2*j+1]])
    if min1 < min2:
        l = pts[0]
        z = pts[1]
    else:
        l = pts[1]
        z = pts[0]
    if in_dif_part(cvadr[l], cvadr[q], cvadr[k], cvadr[z]):
        start = k
        return cvadr[start]
    else:
        start = q
        return cvadr[start]


def magic_func_for_pp6(l1, l2, l3, lines):
    cvadr = cub(l1, l2)
    min1 = dist([l3[1][2], l3[1][3]], cvadr[0])
    k = 0
    for i in range(1, 4):
        m = dist([l3[1][2], l3[1][3]], cvadr[i])
        if m < min1:
            min1 = m
            k = i
    q = 0
    min2 = dist([l3[2][2], l3[2][3]], cvadr[0])
    for i in range(1, 4):
        m = dist([l3[2][2], l3[2][3]], cvadr[i])
        if m < min2:
            min2 = m
            q = i
    pts = []
    for i in range(4):
        if i != k and i != q:
            pts.append(i)
    min1 = dist(cvadr[pts[0]], [lines[0][0], lines[0][1]])
    min2 = dist(cvadr[pts[1]], [lines[0][0], lines[0][1]])
    for i in range(3):
        for j in range(2):
            if dist(cvadr[pts[0]], [lines[i][2*j], lines[i][2*j+1]]) < min1:
                min1 = dist(cvadr[pts[0]], [lines[i][2*j], lines[i][2*j+1]])
                p1 = i
            if dist(cvadr[pts[1]], [lines[i][2*j], lines[i][2*j+1]]) < min2:
                min2 = dist(cvadr[pts[1]], [lines[i][2*j], lines[i][2*j+1]])
                p2 = i
    if min1 < min2:
        l = pts[0]
        z = pts[1]
        rp = p1
    else:
        l = pts[1]
        z = pts[0]
        rp = p2
    if in_dif_part(cvadr[l], cvadr[q], cvadr[k], cvadr[z]):
        #start = k
        ind = edges_from_st(k)
        pp3 = [l1[ind[0]], l2[ind[1]], l3[1]]
        #pp3 = pts.append(q)
        return [cvadr[k], pp3]

    else:  # if in_dif_part(cvadr[l], cvadr[k], cvadr[q], cvadr[z]):
        # start = q
        # pp3 = pts.append(k)
        ind = edges_from_st(q)
        pp3 = [l1[ind[0]], l2[ind[1]], l3[2]]
        return [cvadr[q], pp3]


def edges_from_st(i):
    if i == 0:
        return [2, 2]
    if i == 1:
        return [2, 1]
    if i == 2:
        return [1, 2]
    if i == 3:
        return [1, 1]


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
    return [[vps[0], d_interseptions[vps[0]][0], d_interseptions[vps[0]][1]], [vps[1],d_interseptions[vps[1]][0], d_interseptions[vps[1]][1]], [vps[2], d_interseptions[vps[2]][0], d_interseptions[vps[2]][1]]] ## вернуть vps


def krestik(l1, l2):
    min = dist([l1[1][2], l1[1][3]], [l2[1][2], l2[1][3]])
    pr = [1, 1]
    min2 = dist([l1[1][2], l1[1][3]], [l2[2][2], l2[2][3]])
    if min2 < min:
        min = min2
        pr = [1, 2]
    min3 = dist([l1[2][2], l1[2][3]], [l2[1][2], l2[1][3]])
    if min3 < min:
        min = min3
        pr = [2, 1]
    min4 = dist([l1[2][2], l1[2][3]], [l2[2][2], l2[2][3]])
    if min4 < min:
        pr = [2, 2]
    return cross_pp([l1[pr[0]][0], l1[pr[0]][1]], [l1[pr[0]][2], l1[pr[0]][3]], [l2[pr[1]][0], l2[pr[1]][1]], [l2[pr[1]][2], l2[pr[1]][3]])


def signdef(l1, l2, l3):
    a = [0, 0]
    if distor([l1[1][0], l1[1][1]], [l1[1][2], l1[1][3]],l2[0]) < 0:
        a[0] = 1
    if distor([l1[2][0], l1[2][1]], [l1[2][2], l1[2][3]], l3[0]) < 0:
        a[1] = 1
    return a


def check_bot_edge_inside(s, p, q, l, a):
    a1 = distor(s, p, [l[0], l[1]])
    a2 = distor(s, p, [l[2], l[3]])
    b1 = distor(s, q, [l[0], l[1]])
    b2 = distor(s, q, [l[2], l[3]])
    if a1 < 0 and a2 < 0 and a[0] == 0:
        k1 = True
    if a1 > 0 and a2 > 0 and a[0] == 1:
        k1 = True
    else:
        k1 = False
    if b1 < 0 and b2 < 0 and a[1] == 0:
        k2 = True
    if b1 > 0 and b2 > 0 and a[1] == 1:
        k2 = True
    else:
        k2 = False
    return k1 and k2


def botombothcheck(l1, l2, l3):
    a = signdef(l1, l2, l3)
    a2 = check_bot_edge_inside(l1[0], [l1[1][0], l1[1][1]],
                          [l1[2][0], l1[2][1]], l2[1], a)
    b2 = check_bot_edge_inside(l1[0], [l1[1][0], l1[1][1]],
                          [l1[2][0], l1[2][1]], l2[2], a)
    a3 = check_bot_edge_inside(l1[0], [l1[1][0], l1[1][1]],
                          [l1[2][0], l1[2][1]], l3[1], a)
    b3 = check_bot_edge_inside(l1[0], [l1[1][0], l1[1][1]],
                          [l1[2][0], l1[2][1]], l3[2], a)
    if (a2 and b2) or (a3 and b3):
        return True
    else:
        return False


# нахождение центральной точки
def midpoint(l1, l2, l3, lines):
    c1 = bothcheck(l1, l2, l3)
    c2 = bothcheck(l2, l1, l3)
    c3 = botombothcheck(l3, l1, l2)
    if c1 and c2:
        start = magic_func(l1, l2, l3, lines)
        # ps = magic_func_for_pp6(l1, l2, l3, lines)
        return start #[start, ps]
    elif c1 and c3:
        start = magic_func(l1, l3, l2, lines)
        # ps = magic_func_for_pp6(l1, l3, l2, lines)
        return start #[start, ps]
    elif c2 and c3:
        start = magic_func(l2, l3, l1, lines)
        # ps = magic_func_for_pp6(l2, l3, l1, lines)
        return start #[start, ps]
    else:
        return krestik(l1, l2)


# Определения положения  p3, p4 относительно лайн p1, p2
def in_dif_part(p1, p2, p3, p4):
    if p2[1] - p1[1] == 0:
        return (p3[1] - p2[1])*(p2[1] - p4[1]) > 0
    if p2[0] - p1[0] == 0:
        return (p3[0] - p2[0])*(p2[0] - p4[0]) > 0
    else:
        a = ((p3[1] - p1[1]) / (p2[1] - p1[1]) - \
            (p3[0] - p1[0]) / (p2[0] - p1[0]))
        b = ((p4[1] - p1[1]) / (p2[1] - p1[1]) - \
             (p4[0] - p1[0]) / (p2[0] - p1[0]))
        return a*b < 0


# cos
def cos(v1, v2):  #
    return (v1[0]*v2[0] + v1[1]*v2[1])/(dist(v1,[0,0])*dist(v2,[0,0])) #np.dot(v1, v2) / (norm(v1) * norm(v2))


def distor(v, p, t):
    if p[1] == v[1]:
        return t[1] - v[1]
    elif p[0] == v[0]:
        return t[0]-v[0]
    else:
        return (t[1]-v[1])/(p[1]-v[1]) - (t[0]-v[0])/(p[0]-v[0])


# True если обе точки ребра внутри ребра l
def check_edgeinside(vp, p1, p2, l):
    count = 0
    for i in range(2):
        if distor(vp, p1, [l[2*i], l[2*i+1]])*distor(vp, p2, [l[2*i], l[2*i+1]]) < 0:
            count += 1
    #print(count)
    if count == 2:
        return True
    else:
        return False


def check_edgeinside_res(vp, p1, p2, l):
    v1 = [p1[0] - vp[0], p1[1] - vp[1]]
    v2 = [p2[0] - vp[0], p2[1] - vp[1]]
    v3 = [l[0] - vp[0], l[1] - vp[1]]
    v4 = [l[2] - vp[0], l[3] - vp[1]]
    cos1 = cos(v1, v2)
    #print(cos1)
    cos2 = cos(v1, v3)
    cos3 = cos(v2, v3)
    #print(cos2, cos3)
    count = 0
    if cos1 <= cos2 and cos1 <= cos3:
        count += 1
    cos2 = cos(v1, v4)
    cos3 = cos(v2, v4)
    #print(cos2, cos3)
    if cos1 <= cos2 and cos1 <= cos3:
        count += 1
    if count == 2:
        return True
    else:
        return False


# проверка, что p внутри vp1, vp2, vp3
def check_inside(p, vp1, vp2, vp3):
    if vp1[1] == vp2[1]:
        a = p[1] < vp1[1]
    else:
        a = (p[1] - vp1[1])/(vp2[1] - vp1[1]) - \
                    (p[0] - vp1[0])/(vp2[0] - vp1[0]) < 0
    b = (p[1] - vp1[1])/(vp3[1] - vp1[1]) - \
                    (p[0] - vp1[0])/(vp3[0] - vp1[0]) > 0
    c = (p[1] - vp3[1])/(vp2[1] - vp3[1]) - \
                    (p[0] - vp3[0])/(vp2[0] - vp3[0]) > 0
    return a and b and c


def eq(l1, l2):
    if l1[0] == l2[0] and l1[1] == l2[1] and l1[2] == l2[2] and l1[3] == l2[3]:
        return True
    else:
        return False


def lines_zip(r):
    l = r[1]
    if dist(r[0], [l[0],l[1]]) > dist(r[0], [l[2],l[3]]):
        l[0], l[1], l[2], l[3] = l[2], l[3], l[0], l[1]
    z1 = [(3 * l[0] + l[2]) / 4, (3 * l[1] + l[3]) / 4, (l[0] + 3 * l[2]) / 4,
          (l[1] + 3 * l[3]) / 4]
    l = r[2]
    if dist(r[0], [l[0],l[1]]) > dist(r[0], [l[2],l[3]]):
        l[0], l[1], l[2], l[3] = l[2], l[3], l[0], l[1]
    z2 = [(3 * l[0] + l[2]) / 4, (3 * l[1] + l[3]) / 4, (l[0] + 3 * l[2]) / 4,
              (l[1] + 3 * l[3]) / 4]

    return [r[0], z1, z2]


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


def midp(lines):
    vpssh = vanish_points1(lines)
    vps = [vpssh[0][0], vpssh[1][0], vpssh[2][0]]
    # print(vps)
    linesss = []
    for k in range(len(lines)):
        vvv = False
        for i in range(3):
            for j in range(1, 3):
                if eq(lines[k], vpssh[i][j]):
                    vvv = True
        if not vvv:
            linesss.append(lines[k])

    ind = reroute(vps[0], vps[1], vps[2])
    # print(vps[ind[0]], vps[ind[1]], vps[ind[2]])
    mean_p = midpoint(lines_zip(vpssh[ind[0]]), lines_zip(vpssh[ind[1]]),
                      lines_zip(vpssh[ind[2]]), linesss)
    # if len(mean_p[0]) > 1:
    #     pp3 = mean_p[1]
    #     # lin = []
    #     # for i in range(len(lines)):
    #     #     vv = False
    #     #     for j in range(len(pp3)):
    #     #         if eq(pp3[1][j], lines[i]):
    #     #             vv = True
    #     #     if vv == False:
    #     #         lin.append(lines[i])
    #     # print(len(lin))
    #     mean_p = mean_p[0]
    return [int(mean_p[0]), int(mean_p[1])]

############################################################
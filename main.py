from itertools import combinations

import cv2
import numpy as np
from numpy.linalg import norm


def dist_to_line(l, p):
    """
    >>> p = np.array((1, 2))
    >>> l1 = np.array((-7.51, 8))
    >>> l2 = np.array((21.3, 15.6))
    >>> print(dist_to_line(l1, l2, p))
    """
    l1 = np.array(l[0], l[1])
    l2 = np.array(l[2], l[3])
    d = norm(np.cross(l2 - l1, l1 - p)) / norm(l2 - l1)
    return d


def get_lines(file_name):
    img = cv2.imread(file_name)
    edges = cv2.Canny(img, 100, 200)

    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength,
                            maxLineGap)
    return list(map(lambda line: line[0], lines))


def verify_and_delete(i1, i2, d_lines):
    # TODO:
    pass


if __name__ == "__main__":
    lines = get_lines("cube.png")
    d_lines = {i: line for i, line in enumerate(lines)}
    combs = combinations(range(len(lines)), 2)
    for comb in combs:
        # verify_and_delete(comb[0], comb[1], d_lines)
        print(lines[comb[0]])


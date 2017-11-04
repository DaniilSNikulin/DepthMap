import math
import numpy as np
from numpy.linalg import norm


def dist(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))


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

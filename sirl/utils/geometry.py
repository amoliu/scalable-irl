
from __future__ import division


import numpy as np


__all__ = [
           "normalize",
           "addangles",
           "angle_between",
           "normalize_vector",
           "distance_to_segment",
           "edist"
           ]


def edist(v1, v2):
    """ Euclidean distance between two 2D vectors """
    return np.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)


def distance_to_segment(x, xs, xe):
    xa = xs[0]
    ya = xs[1]
    xb = xe[0]
    yb = xe[1]
    xp = x[0]
    yp = x[1]

    # x-coordinates
    A = xb-xa
    B = yb-ya
    C = yp*B+xp*A
    a = 2*((B*B)+(A*A))
    b = -4*A*C+(2*yp+ya+yb)*A*B-(2*xp+xa+xb)*(B*B)
    c = 2*(C*C)-(2*yp+ya+yb)*C*B+(yp*(ya+yb)+xp*(xa+xb))*(B*B)
    if b*b < 4*a*c:
        return None, False
    x1 = (-b + np.sqrt((b*b)-4*a*c))/(2*a)
    x2 = (-b - np.sqrt((b*b)-4*a*c))/(2*a)

    # y-coordinates
    A = yb-ya
    B = xb-xa
    C = xp*B+yp*A
    a = 2*((B*B)+(A*A))
    b = -4*A*C+(2*xp+xa+xb)*A*B-(2*yp+ya+yb)*(B*B)
    c = 2*(C*C)-(2*xp+xa+xb)*C*B+(xp*(xa+xb)+yp*(ya+yb))*(B*B)
    if b*b < 4*a*c:
        return None, False
    y1 = (-b + np.sqrt((b*b)-4*a*c))/(2*a)
    y2 = (-b - np.sqrt((b*b)-4*a*c))/(2*a)

    # Put point candidates together
    xfm1 = [x1, y1]
    xfm2 = [x2, y2]
    xfm3 = [x1, y2]
    xfm4 = [x2, y1]

    dvec = list()
    dvec.append(edist(xfm1, x))
    dvec.append(edist(xfm2, x))
    dvec.append(edist(xfm3, x))
    dvec.append(edist(xfm4, x))

    dmax = -1.0
    imax = -1
    for i in range(4):
        if dvec[i] > dmax:
            dmax = dvec[i]
            imax = i

    xf = xfm1
    if imax == 0:
        xf = xfm1
    elif imax == 1:
        xf = xfm2
    elif imax == 2:
        xf = xfm3
    elif imax == 3:
        xf = xfm4

    xs_xf = [xs[0]-xf[0], xs[1]-xf[1]]
    xe_xf = [xe[0]-xf[0], xe[1]-xf[1]]
    dotp = (xs_xf[0] * xe_xf[0]) + (xs_xf[1] * xe_xf[1])

    inside = False
    if dotp <= 0.0:
        inside = True

    return dmax, inside


def normalize(theta, start=0):
    """
    Normalize an angle to be in the range :math:`[0, 2\pi]`

    Parameters
    -----------
    theta : float
        input angle to normalize

    start: float
        input start angle (optional, default: 0.0)

    Returns
    --------
    res : float
        normalized angle or :math:`\infty`

    """
    if theta < np.inf:
        while theta >= start + 2 * np.pi:
            theta -= 2 * np.pi
        while theta < start:
            theta += 2 * np.pi
        return theta
    else:
        return np.inf


def addangles(alpha, beta):
    """
    Add two angles

    Parameters
    ----------
    alpha : float
        Augend (in radians)
    beta : float
        Addend (in radians)

    Returns
    -------
    sum : float
        Sum (in radians, normalized to [0, 2pi])
    """
    return normalize(alpha + beta, start=0)


def normalize_vector(vector):
    """ Returns the unit vector of the vector.  """
    norm = np.linalg.norm(vector)
    if abs(norm-0.0) < 0.5 * 10**(-32):
        return vector
    return vector/norm


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'

    Example
    ------------
    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = normalize_vector(v1)
    v2_u = normalize_vector(v2)
    dp = np.dot(v1_u, v2_u)
    if np.isinf(dp) or np.isnan(dp):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return np.pi
    if dp < -1 or dp > 1:  # to ensure real values arccos
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return np.pi
    return np.arccos(dp)

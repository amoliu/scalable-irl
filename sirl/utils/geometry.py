
from __future__ import division

import itertools
import numpy as np


__all__ = [
           'normangle',
           'addangles',
           'subangles',
           'angle_between',
           'normalize_vector',
           'distance_to_segment',
           'edist',
           'line_crossing',
           'anisotropic_distance',
           'trajectory_length',
           'goal_bearing',
           'relative_heading',
           ]


def trajectory_length(traj):
    """
    Compute the length of a path travelled by an agent
    Parameters
    -----------
    traj : numpy array
        Trajectory representing the path travelled as a `numpy` array of
        shape [frame_size x n_waypoints]. Frame encodes information at
        every time step [x, y, vx, vy, ...]

    Returns
    ---------
    path_length : float
        Length of the path based on Euclidean distance metric
    """

    assert traj.ndim == 2, "Trajectory must be a two dimensional array"

    path_length = 0.0
    for i, j in itertools.izip(range(traj.shape[0]), range(1, traj.shape[0])):
        current, nextstep = traj[i, 0:2], traj[j, 0:2]
        path_length += np.linalg.norm(current - nextstep)

    return path_length


def edist(v1, v2):
    """ Euclidean distance between two 2D vectors """
    return np.hypot(v1[0] - v2[0], v1[1] - v2[1])


def anisotropic_distance(focal_agent, other_agent,
                         phi_ij=None, ak=2.48, bk=1.0,
                         lambda_=0.4, rij=0.9):
    """
    Anisotropic distance based on the Social Force Model (SFM)
    model of pedestrian dynamics.
    """
    ei = np.array([-focal_agent[2], -focal_agent[3]])
    ei = normalize_vector(ei)

    if phi_ij is None:
        phi = np.arctan2(other_agent[1] - focal_agent[1],
                         other_agent[0] - focal_agent[0])
    else:
        phi = phi_ij

    dij = edist(focal_agent, other_agent)
    nij = np.array([np.cos(phi), np.sin(phi)])
    ns = 2
    alpha = ak * np.exp((rij - dij) / bk) * nij
    beta_ = np.tile(np.ones(shape=(1, ns)) * lambda_ + ((1 - lambda_)
                    * (np.ones(shape=(1, ns)) - (np.dot(nij.T, ei)).T) / 2.),
                    [1, 1])
    curve = np.multiply(alpha, beta_).T
    dc = np.hypot(curve[0], curve[1])
    return dc


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


def normangle(theta, start=0):
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
    return normangle(alpha + beta, start=0)


def subangles(alpha, beta):
    return normangle(alpha - beta, start=0)


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


def line_crossing(x1, y1, x2, y2, x3, y3, x4, y4):
    """ Check if line segments cross each other

    check if the line segments [(x1, y1), (x2, y2)] and [(x3, y3), (x4, y4)]
    cross each other

    Parameters
    -----------
    x1, y1, x2, y2, x3, y3, x4, y4 : float
        coordonates of the points defining the line segments

    Returns
    --------
    crossing : int
        0 (False) or 1 (True)
    """
    if x2 == x1:
        if x3 == x4:
            if x1 != x3:
                return 0
            elif max(y3, y4) < min(y1, y2) or max(y1, y2) < min(y3, y4):
                return 0
            else:
                return 1
        else:
            a2 = (y4 - y3) / (x4 - x3)
            b2 = y3 - (a2 * x3)
            if a2 == 0:
                if min(y1, y2) > b2 or max(y1, y2) < b2:
                    return 0
                elif x2 <= max(x3, x4) and x2 >= min(x3, x4):
                    return 1
                else:
                    return 0
            elif a2 * x1 + b2 <= min(max(y3, y4), max(y1, y2)) and \
                    a2 * x1 + b2 >= max(min(y3, y4), min(y1, y2)):
                return 1
            else:
                return 0
    elif x3 == x4:
        if x1 == x2:
            if x1 != x3:
                return 0
            elif max(y3, y4) < min(y1, y2) or max(y1, y2) < min(y3, y4):
                return 0
            else:
                return 1
        else:
            a1 = (y2 - y1) / (x2 - x1)
            b1 = y1 - (a1 * x1)
            if a1 == 0:
                if min(y3, y4) > b1 or max(y3, y4) < b1:
                    return 0
                elif x3 <= max(x1, x2) and x3 >= min(x1, x2):
                    return 1
                else:
                    return 0
            elif a1 * x3 + b1 <= min(max(y1, y2), max(y3, y4)) and \
                    a1 * x3 + b1 >= max(min(y1, y2), min(y3, y4)):
                return 1
            else:
                return 0
    else:
        a1 = (y2 - y1) / (x2 - x1)
        a2 = (y4 - y3) / (x4 - x3)
        if a1 == a2:
            return 0
        else:
            b2 = y3 - (a2 * x3)
            b1 = y1 - (a1 * x1)
            xcommun = (b2 - b1) / (a1 - a2)
            if xcommun >= max(min(x1, x2), min(x3, x4)) and \
                    xcommun <= min(max(x1, x2), max(x3, x4)):
                return 1
            else:
                return 0


def relative_heading(pose1, pose2):
    """ Relative heading between two poses

    Poses are 1D vector of [x, y, theta, speed]
    """
    return normangle(pose1[2] - pose2[2])


def goal_bearing(pose, g):
    """ The goal bearing of a pose

    Goal is given a point in 2D

    """
    xp = pose[0] + pose[2] * np.cos(pose[2])
    yp = pose[1] + pose[2] * np.sin(pose[2])
    bearing = np.arctan2(g[1]-yp, g[0]-xp)
    return bearing


def point_infront_of_body(line, back_point, test_point):
    """ Check if a point is on the front side a convex body given a point
    on the back side. The line represents the face of the body
    """
    Ax = line[0][0]
    Ay = line[0][1]
    Bx = line[1][0]
    By = line[1][1]
    back = np.sign((Bx-Ax)*(back_point[1]-Ay)-(By-Ay)*(back_point[0]-Ax))

    side = np.sign((Bx-Ax)*(test_point[1]-Ay)-(By-Ay)*(test_point[0]-Ax))
    if side != back:
        return True

    return False


def perp_from_point(line, distance, start=True):
    """
    Find a point perpendicular to the given line at the given distance
    and passing through either the start or end of the line segment
    """
    Ax = line[0][0]
    Ay = line[0][1]
    Bx = line[1][0]
    By = line[1][1]
    # - find the y intercept of the perperdicular line
    c = By + ((Bx-Ax)/(By-Ay))*Bx

    if start:
        vec = normalize_vector([(0-Ax), (c-Ay)])
        transform = vec * distance
        new_point = np.array((Ax, Ay)) + transform
        return new_point
    else:
        vec = normalize_vector([(0-Bx), (c-By)])
        transform = vec * distance
        new_point = np.array((Bx, By)) + transform
        return new_point

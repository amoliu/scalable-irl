
from __future__ import division

import numpy as np

try:
    from itertools import izip
except ImportError:
    izip = zip


__all__ = [
    'normangle',
    'distance_to_segment',
    'edist',
    'anisotropic_distance',
    'trajectory_length',
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

    traj = np.asarray(traj)
    assert traj.ndim == 2, "Trajectory must be a two dimensional array"

    path_length = 0.0
    for i, j in izip(range(traj.shape[0]), range(1, traj.shape[0])):
        current, nextstep = traj[i, 0:2], traj[j, 0:2]
        path_length += np.linalg.norm(current - nextstep)

    return path_length


def edist(v1, v2):
    """ Euclidean distance between two 2D vectors """
    return np.hypot(v1[0] - v2[0], v1[1] - v2[1])


def anisotropic_distance(i, j, ak=2.48, bk=1.0, lambda_=0.4, rij=0.9):
    """
    Anisotropic distance based on the Social Force Model (SFM)
    model of pedestrian dynamics.

    """
    ei = np.array([-i[2], -i[3]])
    ei = _normalize_vector(ei)

    phi = np.arctan2(j[1] - i[1],
                     j[0] - i[0])

    dij = edist(i, j)
    nij = np.array([np.cos(phi), np.sin(phi)])
    ns = 2
    alpha = ak * np.exp((rij - dij) / bk) * nij
    beta_ = np.tile(np.ones(shape=(1, ns)) * lambda_ + ((1 - lambda_)
                    * (np.ones(shape=(1, ns)) - (np.dot(nij.T, ei)).T) / 2.),
                    [1, 1])
    curve = np.multiply(alpha, beta_).T
    dc = np.hypot(curve[0], curve[1])
    return dc


def distance_to_segment(point, line_start, line_end):
    xa, ya = line_start[0], line_start[1]
    xb, yb = line_end[0], line_end[1]
    xp, yp = point[0], point[1]

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
    candidates = ((x1, y2), (x2, y2), (x1, y2), (x2, y1))
    distances = (edist(candidates[0], point), edist(candidates[1], point),
                 edist(candidates[2], point), edist(candidates[3], point))
    max_index = np.argmax(distances)
    cand = candidates[max_index]
    dmax = distances[max_index]

    start_cand = (line_start[0]-cand[0], line_start[1]-cand[1])
    end_cand = (line_end[0]-cand[0], line_end[1]-cand[1])
    dotp = (start_cand[0] * end_cand[0]) + (start_cand[1] * end_cand[1])

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


def _normalize_vector(vector):
    """ Returns the unit vector of the vector.  """
    norm = np.linalg.norm(vector)
    if abs(norm-0.0) < 0.5 * 10**(-32):
        return vector
    return vector/norm

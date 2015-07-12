
from __future__ import division

import numpy as np

# from matplotlib.patches import Circle, Ellipse
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import matplotlib as mpl


from ..models import LocalController
from ..models import GraphMDP
from ..models import _controller_duration

from ..utils.geometry import edist
from ..algorithms.mdp_solvers import graph_policy_iteration


########################################################################

class PuddleWorldControler(LocalController):
    """ PuddleWorldControler local controller """
    def __init__(self, kind='linear'):
        super(PuddleWorldControler, self).__init__(kind)

    def __call__(self, state, action, duration):
        """ Run a local controller from a ``state`` using ``action``
        """
        nx = state[0] + np.cos(action * 2 * np.pi) * duration
        ny = state[1] + np.sin(action * 2 * np.pi) * duration

        if self._wconfig.x < nx < self._wconfig.w and\
                self._wconfig.y < ny < self._wconfig.h:
            return (nx, ny)
        return state


########################################################################


class PuddleWorldMDP(GraphMDP):
    """ Puddle world MDP

    Parameters
    ------------
    discount : float
        MDP discount factor
    reward : ``SocialNavReward`` object
        Reward function for social navigation task
    controller : ``SocialNavLocalController`` object
        Local controller for the task
    params : ``GraphMDPParams`` object
        Algorithm parameters for the various steps

    """
    def __init__(self, discount, reward, controller, params, world_config):
        super(PuddleWorldMDP, self).__init__(discount, reward,
                                             controller, params)

    def initialize_state_graph(self, samples):
        raise NotImplementedError('TODO')

    def terminal(self, state):
        """ Check if a state is terminal (goal state) """
        position = self._g.gna(state, 'data')
        if edist(position, self._params.goal_state) < 0.05:
            return True
        return False

    # -------------------------------------------------------------
    # internals
    # -------------------------------------------------------------

    def _setup_puddles(self):
        self.puddles = list()
        self.puddles.append(Puddle(0.1, 0.75, 0.45, 0.75, 0.1))
        self.puddles.append(Puddle(0.45, 0.4, 0.45, 0.8, 0.1))


########################################################################


class Puddle(object):
    """ A puddle in a continous puddle world
    Represented by combinations of a line and semi-circles at each end,
    i.e.
    (-----------)
    Parameters
    R-----------
    x1 : float
        X coordinate of the start of the center line
    x2 : float
        X coordinate of the end of the center line
    y1 : float
        Y coordinate of the start of the center line
    y2 : float
        Y coordinate of the end of the center line
    radius : float
        Thickness/breadth of the puddle in all directions
    Attributes
    -----------
    `start_pose` : array-like
        1D numpy array with the start of the line at the puddle center line
    `end_pose`: array-like
        1D numpy array with the end of the line at the puddle center line
    `radius`: float
        Thickness/breadth of the puddle in all directions
    """
    def __init__(self, x1, y1, x2, y2, radius, **kwargs):
        assert x1 >= 0 and x1 <= 1, 'Puddle coordinates must be in [0, 1]'
        assert x2 >= 0 and x2 <= 1, 'Puddle coordinates must be in [0, 1]'
        assert y1 >= 0 and y1 <= 1, 'Puddle coordinates must be in [0, 1]'
        assert y2 >= 0 and y2 <= 1, 'Puddle coordinates must be in [0, 1]'
        assert radius > 0, 'Puddle radius must be > 0'
        self.start_pose = np.array([x1, y1])
        self.end_pose = np.array([x2, y2])
        self.radius = radius

    def cost(self, x, y):
        dist_puddle, inside = distance_to_segment(self.start_pose,
                                                  self.end_pose, (x, y))
        if inside and dist_puddle < self.radius:
            return -400.0 * (self.radius - dist_puddle)
        return 0.0

    @property
    def location(self):
        return self.start_pose[0], self.start_pose[1],\
            self.end_pose[0], self.end_pose[1]

    @property
    def length(self):
        return edist(self.start_pose, self.end_pose)


from __future__ import division

import numpy as np

from ..models import LocalController
from ..models import MDPReward
from ..models import GraphMDP


########################################################################


class SocialNavLocalController(LocalController):
    """ Social navigation local controller in 2D space """
    def __init__(self, kind='linear'):
        super(SocialNavLocalController, self).__init__(kind)

    def __call__(self, state, action, duration):
        """ Run a local controller from a state

        Run the local controller at the given ``state`` using the ``action``
        represented by an angle, :math:` \alpha \in [0, \pi]` for a time limit
        given by ``duration``

        Parameters
        -----------
        state : array like, shape = [1 x 2]
            2D pose of the start state
        action : float
            Angle representing the action taken
        duration : float
            Real time interval limit for executing the controller

        Returns
        --------
        new_state : 2D array
            New state reached by the controller
        """
        nx = state[0] + np.cos(action) * duration
        ny = state[1] + np.sin(action) * duration
        return (nx, ny)

########################################################################


class SocialNavReward(MDPReward):
    """ Social Navigation Reward Funtion """
    def __init__(self, persons, relations, kind='linfa', resolution=0.1):
        super(SocialNavReward, self).__init__(kind)
        self._persons = persons
        self._relations = relations
        self._resolution = resolution

    def __call__(self, state, action):
        return 0.0

########################################################################


class SocialNavMDP(GraphMDP):
    """docstring for SocialNavMDP"""
    def __init__(self, arg):
        super(SocialNavMDP, self).__init__()
        self.arg = arg

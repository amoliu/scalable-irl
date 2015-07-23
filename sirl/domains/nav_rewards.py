from __future__ import division

import numpy as np

from ..models import MDPReward
from ..models import _controller_duration

from ..utils.geometry import edist, anisotropic_distance
from ..utils.geometry import line_crossing


__all__ = [
    'SimpleReward',
    'ScaledSimpleReward',
    'AnisotropicReward',
]


class SimpleReward(MDPReward):
    """ Social Navigation Reward Funtion
    based on intrusion counts (histogram)

    """
    def __init__(self, persons, relations, goal, weights, discount,
                 kind='linfa', resolution=0.2, hzone=0.45):
        super(SimpleReward, self).__init__(kind)
        self._persons = persons
        self._relations = relations
        self._resolution = resolution
        self._goal = goal
        self._weights = weights
        self._gamma = discount
        self._hzone = hzone

    def __call__(self, state_a, state_b):
        source, target = np.array(state_a), np.array(state_b)
        # increase resolution of action trajectory (option)
        duration = _controller_duration(source, target) * 1.0/self._resolution
        action_traj = [target * t / duration + source * (1 - t / duration)
                       for t in range(int(duration))]
        action_traj.append(target)
        action_traj = np.array(action_traj)

        phi = [self._relation_disturbance(action_traj),
               self._social_disturbance(action_traj),
               self._goal_deviation_count(action_traj)]
        reward = np.dot(phi, self._weights)
        return reward, phi

    @property
    def dim(self):
        return 3

    # -------------------------------------------------------------
    # internals
    # -------------------------------------------------------------

    def _goal_deviation_count(self, action):
        """ Goal deviation measured by counts for every time
        a waypoint in the action trajectory recedes away from the goal
        """
        dist = []
        for i in range(action.shape[0]-1):
            dnow = edist(self._goal, action[i])
            dnext = edist(self._goal, action[i + 1])
            dist.append(max((dnext - dnow) * self._gamma ** i, 0))
        return sum(dist)

    def _social_disturbance(self, action):
        pd = [min([edist(wp, person) for _, person in self._persons.items()])
              for wp in action]
        phi = sum(1 * self._gamma**i
                  for i, d in enumerate(pd) if d < self._hzone)
        return phi

    def _relation_disturbance(self, action):
        # TODO - fix relations to start from 0 instead of 1
        atime = action.shape[0]
        c = [sum(line_crossing(action[t][0],
                 action[t][1],
                 action[t+1][0],
                 action[t+1][1],
                 self._persons[i][0],
                 self._persons[i][1],
                 self._persons[j][0],
                 self._persons[j][1])
             for [i, j] in self._relations) for t in range(int(atime - 1))]
        ec = sum(self._gamma**i * x for i, x in enumerate(c))
        return ec


############################################################################


class ScaledSimpleReward(SimpleReward):
    """ Social Navigation Reward Funtion using Gaussians """
    def __init__(self, persons, relations, goal, weights, discount,
                 kind='linfa', resolution=0.2, hzone=0.45):
        super(ScaledSimpleReward, self).__init__(persons, relations, goal,
                                                 weights, discount,
                                                 resolution, hzone)
    # --- override key functions

    def _social_disturbance(self, action):
        """ social disturbance based on a circle around persons
        that is scaled by the person's speed. Assuming 1m/s = ``_hzone``
        """
        phi = 0
        for _, p in self._persons.items():
            speed = np.hypot(p[2], p[3])
            hz = speed * 0.5 * self._hzone
            for wp in action:
                if edist(wp, p) < hz:
                    phi += 1
        return phi


############################################################################


class AnisotropicReward(SimpleReward):
    """ Simple reward using an Anisotropic circle around persons"""
    def __init__(self, persons, relations, goal, weights, discount,
                 kind='linfa', resolution=0.2, hzone=0.45):
        super(AnisotropicReward, self).__init__(persons, relations, goal,
                                                weights, discount,
                                                resolution, hzone)

    def _social_disturbance(self, action):
        phi = 0
        for _, p in self._persons.items():
            # speed = np.hypot(p[2], p[3])
            # hz = speed * 0.5 * self._hzone
            for wp in action:
                ad = anisotropic_distance(p, wp, ak=2*self._hzone)
                if edist(wp, p) < ad:
                    phi += 1
        return phi

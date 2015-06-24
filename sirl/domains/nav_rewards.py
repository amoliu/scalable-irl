from __future__ import division

import numpy as np

from ..models import MDPReward
from ..models import _controller_duration

from ..utils.geometry import edist, distance_to_segment
from ..utils.geometry import line_crossing
from ..utils.common import eval_gaussian


__all__ = [
    'HistogramSocialNavReward',
    'GaussianSocialNavReward',
]


class HistogramSocialNavReward(MDPReward):
    """ Social Navigation Reward Funtion
    based on intrusion counts (histogram)

    """
    def __init__(self, persons, relations, goal, weights, discount,
                 kind='linfa', resolution=0.1):
        super(HistogramSocialNavReward, self).__init__(kind)
        self._persons = persons
        self._relations = relations
        self._resolution = resolution
        self._goal = goal
        self._weights = weights
        self._gamma = discount

    def __call__(self, state_a, state_b):
        source, target = np.array(state_a), np.array(state_b)
        # increase resolution of action trajectory (option)
        duration = _controller_duration(source, target)
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
        pd = [min([edist(wp, person) for person in self._persons])
              for wp in action]
        phi = sum(1 * self._gamma**i for i, d in enumerate(pd) if d < 0.45)
        return phi

    def _relation_disturbance(self, action):
        # TODO - fix relations to start from 0 instead of 1
        atime = action.shape[0]
        c = [sum(line_crossing(action[t][0],
                 action[t][1],
                 action[t+1][0],
                 action[t+1][1],
                 self._persons[i-1][0],
                 self._persons[i-1][1],
                 self._persons[j-1][0],
                 self._persons[j-1][1])
             for [i, j] in self._relations) for t in range(int(atime - 1))]
        ec = sum(self._gamma**i * x for i, x in enumerate(c))
        return ec


############################################################################


class GaussianSocialNavReward(MDPReward):
    """ Social Navigation Reward Funtion using Gaussians """
    def __init__(self, persons, relations, goal, weights, discount,
                 kind='linfa', resolution=0.1):
        super(GaussianSocialNavReward, self).__init__(kind)
        self._persons = persons
        self._relations = relations
        self._resolution = resolution
        self._goal = goal
        self._weights = weights
        self._gamma = discount

    def __call__(self, state_a, state_b):
        source, target = np.array(state_a), np.array(state_b)
        # increase resolution of action trajectory (option)
        duration = _controller_duration(source, target)
        action_traj = [target * t / duration + source * (1 - t / duration)
                       for t in range(int(duration))]
        action_traj.append(target)
        action_traj = np.array(action_traj)

        phi = [self._relation_disturbance(action_traj),
               self._social_disturbance(action_traj),
               self._goal_deviation_count(action_traj)]
        reward = np.dot(phi, self._weights)
        print(phi, reward)
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
        assert isinstance(action, np.ndarray),\
            'numpy ``ndarray`` expected for action trajectory'
        phi = np.zeros(action.shape[0])
        for i, p in enumerate(action):
            for hp in self._persons:
                ed = edist(hp, p)
                if ed < 1.2:
                    phi[i] = eval_gaussian(ed, sigma=0.5) * self._gamma**i
        return np.sum(phi)

    def _relation_disturbance(self, action):
        assert isinstance(action, np.ndarray),\
            'numpy ``ndarray`` expected for action trajectory'
        phi = np.zeros(action.shape[0])
        for k, act in enumerate(action):
            for (i, j) in self._relations:
                link = ((self._persons[i-1][0], self._persons[i-1][1]),
                        (self._persons[j-1][0], self._persons[j-1][1]))

                sdist, inside = distance_to_segment(act, link[0], link[1])
                if inside and sdist < 0.24:
                    phi[k] = eval_gaussian(sdist, sigma=0.5) * self._gamma**k

        return np.sum(phi)

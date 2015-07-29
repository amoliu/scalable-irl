from __future__ import division

import numpy as np

from ...models.reward import MDPReward

from ...utils.geometry import edist, anisotropic_distance
from ...utils.geometry import line_crossing


__all__ = [
    'SimpleReward',
    'ScaledSimpleReward',
    'AnisotropicReward',
    'FlowMergeReward',
]


class SimpleReward(MDPReward):

    """ Social Navigation Reward Funtion
    based on intrusion counts (histogram)

    """

    def __init__(self, persons, relations, annotations, goal,
                 weights, discount, kind='linfa', hzone=0.45):
        super(SimpleReward, self).__init__(kind)
        self._persons = persons
        self._relations = relations
        self._annotations = annotations
        self._goal = goal
        self._weights = weights
        self._gamma = discount
        self._hzone = hzone

    def __call__(self, state, action):
        phi = [self._feature_relation_disturbance(action),
               self._feature_social_disturbance(action),
               self._feature_goal_deviation_count(action),
               self._feature_annotation_disturbance(action)]
        reward = np.dot(phi, self._weights)
        return reward, phi

    @property
    def dim(self):
        return 4

    # -------------------------------------------------------------
    # internals
    # -------------------------------------------------------------

    def _feature_goal_deviation_count(self, action):
        """ Goal deviation measured by counts for every time
        a waypoint in the action trajectory recedes away from the goal
        """
        dist = []
        for i in range(action.shape[0] - 1):
            dnow = edist(self._goal, action[i])
            dnext = edist(self._goal, action[i + 1])
            dist.append(max((dnext - dnow) * self._gamma ** i, 0))
        return sum(dist)

    def _feature_social_disturbance(self, action):
        pd = [min([edist(wp, person) for _, person in self._persons.items()])
              for wp in action]
        phi = sum(1 * self._gamma**i
                  for i, d in enumerate(pd) if d < self._hzone)
        return phi

    def _feature_relation_disturbance(self, action):
        atime = action.shape[0]
        c = [sum(line_crossing(action[t][0],
                               action[t][1],
                               action[t + 1][0],
                               action[t + 1][1],
                               self._persons[i][0],
                               self._persons[i][1],
                               self._persons[j][0],
                               self._persons[j][1])
                 for [i, j] in self._relations) for t in range(int(atime - 1))]
        ec = sum(self._gamma**i * x for i, x in enumerate(c))
        return ec

    def _feature_annotation_disturbance(self, action):
        phi = 0.0
        for wp in action:
            for _, person in self._persons.items():
                for a in self._annotations:
                    phi += a.disturbance(wp, person)
        return phi

############################################################################


class ScaledSimpleReward(SimpleReward):

    """ Social Navigation Reward Funtion using Gaussians """

    def __init__(self, persons, relations, annotations, goal,
                 weights, discount, kind='linfa', hzone=0.45):
        super(ScaledSimpleReward, self).__init__(persons, relations,
                                                 annotations, goal,
                                                 weights, discount, hzone)
    # --- override key functions

    def _feature_social_disturbance(self, action):
        """ social disturbance based on a circle around persons
        that is scaled by the person's speed. Assuming 1m/s = ``_hzone``
        """
        phi = 0
        for _, p in self._persons.items():
            speed = np.hypot(p[2], p[3])
            hz = speed * self._hzone
            for wp in action:
                if edist(wp, p) < hz:
                    phi += 1
        return phi


############################################################################


class AnisotropicReward(SimpleReward):

    """ Simple reward using an Anisotropic circle around persons"""

    def __init__(self, persons, relations, goal, weights, discount,
                 kind='linfa', hzone=0.45):
        super(AnisotropicReward, self).__init__(persons, relations, goal,
                                                weights, discount, hzone)

    def _feature_social_disturbance(self, action):
        phi = 0
        for _, p in self._persons.items():
            # speed = np.hypot(p[2], p[3])
            # hz = speed * 0.5 * self._hzone
            for wp in action:
                ad = anisotropic_distance(p, wp, ak=2 * self._hzone)
                if edist(wp, p) < ad:
                    phi += 1
        return phi


############################################################################


class FlowMergeReward(MDPReward):

    """Flow reward function for merging and interacting with flows """

    def __init__(self, persons, relations, goal, weights,
                 discount, radius=1.2, kind='linfa'):
        super(FlowMergeReward, self).__init__(kind)
        self._persons = persons
        self._relations = relations
        self._goal = goal
        self._weights = weights
        self._gamma = discount
        self._radius = radius
        self._hzone = 0.55

    def __call__(self, state, action):
        # density, speed, angle, goal-dev
        density, heading_similarity = self._flow_feature(action)
        phi = [density,
               heading_similarity,
               self._feature_goal_deviation(action),
               self._feature_goal_distance(action),
               self._feature_relation_disturbance(action),
               self._feature_social_disturbance(action)]
        reward = np.dot(phi, self._weights)
        return reward, phi

    @property
    def dim(self):
        # TODO - get an automatic way of handling this, via registers
        # - use self dict with pre-names??
        return 6

    # -------------------------------------------------------------
    # internals
    # -------------------------------------------------------------

    def _feature_goal_deviation(self, action):
        # TODO - change to theta/angles
        dist = []
        for i in range(action.shape[0] - 1):
            dnow = edist(self._goal, action[i])
            dnext = edist(self._goal, action[i + 1])
            dist.append(max((dnext - dnow) * self._gamma ** i, 0))
        return sum(dist)

    def _feature_goal_distance(self, action):
        phi = 0
        for i, wp in enumerate(action):
            phi += edist(wp, self._goal) * self._gamma**i

        return phi

    def _flow_feature(self, action):
        phi_d = []
        phi_f = []

        for wp in action:
            density = 0
            flow = 0
            for _, p in self._persons.items():
                dist = edist(wp, p[0:2])
                if dist < self._radius:
                    density += 1
                    flow += self._goal_orientation(p)

            if density > 0:
                flow = flow / density

            phi_d.append(density)
            phi_f.append(flow)

        return sum(phi_d), sum(phi_f)

    def _feature_social_disturbance(self, action):
        phi = 0
        for _, p in self._persons.items():
            speed = np.hypot(p[2], p[3])
            hz = speed * self._hzone
            for wp in action:
                if edist(wp, p) < hz:
                    phi += 1
        return phi

    def _feature_relation_disturbance(self, action):
        atime = action.shape[0]
        c = [sum(line_crossing(action[t][0],
                               action[t][1],
                               action[t + 1][0],
                               action[t + 1][1],
                               self._persons[i][0],
                               self._persons[i][1],
                               self._persons[j][0],
                               self._persons[j][1])
                 for [i, j] in self._relations) for t in range(int(atime - 1))]
        ec = sum(self._gamma**i * x for i, x in enumerate(c))
        return ec

    def _goal_orientation(self, person):
        """ Compute a measure of how close a person will be to the goal in
        the next time step using constant velocity model.
        """
        v = np.hypot(person[3], person[2])
        pnext = (person[0] + v * person[2], person[1] + v * person[3])
        dnow = edist(self._goal, person)
        dnext = edist(self._goal, pnext)
        return dnext - dnow

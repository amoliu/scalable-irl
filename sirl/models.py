
from __future__ import division

import numpy as np

from .state_graph import StateGraph


# TODO
# - Move these social nav specific classes to domains module
# - Create abstract interfaces to define the protocol
# - Move the GraphMDP to some other place, make an interface for it

# class MDPState(object):
#     """ GraphMDP State """
#     def __init__(self, sid, data, cost, priority, Q):
#         self.sid = sid
#         self.data = data
#         self.cost = cost
#         self.priority = priority
#         self.Q = Q


class LocalController(object):
    """ GraphMDP Linear Action (local controller) in 2D """
    def __init__(self, angle, duration):
        self.angle = angle
        self.duration = duration

    def __call__(self, state):
        """ Run the local controller at the given ``state``"""
        nx = state.data[0] + np.cos(self.angle) * self.duration
        ny = state.data[1] + np.sin(self.angle) * self.duration
        return (nx, ny)


class MDPReward(object):
    """ Reward represented as linear function approximation"""
    def __init__(self, weights, persons, relations):
        self.weights = weights
        self.persons = persons
        self.relations = relations

    def __call__(self, state, action):
        pass


class GraphMDP(object):
    """ Adaptive State-Graph MDP representation """
    def __init__(self, discount, reward, transition):
        self.discount = discount
        self.reward = reward
        self.transition = transition

    # make abstract to depend on domain
    def initialize_state_graph(self, init_samples):
        pass

    def run(self, max_states):
        pass

    # internals

    def _update_state_costs(self):
        pass

    def _update_state_priorities(self):
        pass

    def _find_best_policies(self):
        pass

    def _prune_graph(self):
        pass

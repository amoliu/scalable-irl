from __future__ import division

import json


class GraphMDPParams(object):
    """ GraphMDP Algorithm parameters """
    def __init__(self):
        self.n_expand = 1   # No of nodes to be expanded
        self.n_new = 20   # no of new nodes
        self.n_add = 1   # no of nodes to be added
        self.radius = 1.8
        self.exp_thresh = 1.2
        self.max_traj_len = 500
        self.goal_reward = 30
        self.p_best = 0.4
        self.max_samples = 50
        self.max_edges = 360
        self.start_states = []
        self.goal_state = (1, 1)
        self.init_type = 'random'
        self.max_cost = 1000
        self.conc_scale = 1
        self.speed = 1

    @property
    def _to_json(self):
        return self.__dict__

    def load(self, json_file):
        with open(json_file, 'r') as f:
            jdata = json.load(f)
            for k, v in jdata.items():
                self.__dict__[k] = v

    def save(self, filename):
        """ Save the parameters to file """
        with open(filename, 'w') as f:
            json.dump(self._to_json, f)

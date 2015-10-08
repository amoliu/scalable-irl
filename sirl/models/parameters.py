
import json


class ControllerGraphParams(object):
    """ ControllerGraph parameters for representation learning

    Uses json encoding for persistence.

    """

    _PARAMS = [
        'n_expand',
        'n_new',
        'n_add',
        'radius',
        'exp_thresh',
        'max_traj_len',
        'p_best',
        'max_samples',
        'max_edges',
        'init_type',
        'max_cost',
        'conc_scale',
        'speed',
        'tmin',
        'tmax',
        'goal_reward',
    ]

    def __init__(self, **kwargs):
        self.n_expand = kwargs.pop('n_expand', 1)
        self.n_new = kwargs.pop('n_new', 20)
        self.n_add = kwargs.pop('n_add', 1)
        self.radius = kwargs.pop('radius', 1.8)
        self.exp_thresh = kwargs.pop('exp_thresh', 1.2)
        self.max_traj_len = kwargs.pop('max_traj_len', 500)
        self.goal_reward = kwargs.pop('goal_reward', 1)
        self.p_best = kwargs.pop('p_best', 0.4)
        self.max_samples = kwargs.pop('max_samples', 100)
        self.max_edges = kwargs.pop('max_edges', 360)
        self.init_type = kwargs.pop('init_type', 'random')
        self.max_cost = kwargs.pop('max_cost', 1000)
        self.conc_scale = kwargs.pop('conc_scale', 1)
        self.speed = kwargs.pop('speed', 1.0)
        self.tmin = kwargs.pop('tmin', (0.45, 2.4))
        self.tmax = kwargs.pop('tmax', (3.6, 7.2))

    def load(self, json_file):
        """ Load parameters from a json file """
        with open(json_file, 'r') as f:
            jdata = json.load(f)
            for k, v in jdata.items():
                self.__dict__[k] = v

    def save(self, filename):
        """ Save the parameters to file """
        with open(filename, 'w') as f:
            json.dump(self._to_dict(), f, indent=4, sort_keys=True)

    def __repr__(self):
        return self._to_dict()

    def __str__(self):
        d = self._to_dict()
        return ''.join('{}: {}\n'.format(k, v) for k, v in d.items())

    def _to_dict(self):
        return self.__dict__

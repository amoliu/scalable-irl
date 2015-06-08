
from __future__ import division
from collections import namedtuple

import numpy as np

from ..models import LocalController
from ..models import MDPReward
from ..models import GraphMDP


########################################################################


class SocialNavLocalController(LocalController):
    """ Social Navigation linear local controller

    Social navigation task linear local controller, which connects states
    using straight lines as actions (options, here considered deterministic).
    The action is thus fully represented by a single float for the angle of
    the line.

    Parameters
    -----------
    kind : string
        LocalController controller type for book-keeping

     Attributes
    -----------
    _wconfig : ``WorldConfig``
        Configuration of the navigation task world

    """
    def __init__(self, world_config, kind='linear'):
        super(SocialNavLocalController, self).__init__(kind)
        self._wconfig = world_config

    def __call__(self, state, action, duration):
        """ Run a local controller from a state

        Run the local controller at the given ``state`` using the ``action``
        represented by an angle, :math:` \alpha \in [0, \pi]` for a time limit
        given by ``duration``

        Parameters
        -----------
        state : array of shape (2)
            Positional data of the state (assuming 0:2 are coordinates)
        action : float
            Angle representing the action taken
        duration : float
            Real time interval limit for executing the controller

        Returns
        --------
        new_state : array of shape (2)
            New state reached by the controller

        Note
        ----
        If the local controller ends up beyond the limits of the world config,
        then the current state is returned to avoid sampling `outside'.
        """
        nx = state[0] + np.cos(action) * duration
        ny = state[1] + np.sin(action) * duration

        if self._wconfig.x < nx < self._wconfig.w and\
                self._wconfig.y < ny < self._wconfig.h:
            return (nx, ny)
        return state


########################################################################


class SocialNavReward(MDPReward):
    """ Social Navigation Reward Funtion """
    def __init__(self, persons, relations, kind='linfa', resolution=0.1):
        super(SocialNavReward, self).__init__(kind)
        self._persons = persons
        self._relations = relations
        self._resolution = resolution

    def __call__(self, state_a, state_b):
        return 0.0

########################################################################


WorldConfig = namedtuple('WorldConfig', ['x', 'y', 'w', 'h'])
COST_LIMIT = 1000


class SocialNavMDP(GraphMDP):
    """ Social Navigation Adaptive State-Graph (SocialNavMDP)

    Social navigation task MDP represented by an adaptive state graph

    Parameters
    ------------
    discount : float
        MDP discount factor
    reward : ``SocialNavReward`` object
        Reward function for social navigation task
    controller : ``SocialNavLocalController`` object
        Local controller for the task
    params : ``AlgoParams`` object
        Algorithm parameters for the various steps
    world_config : ``WorldConfig`` object
        Configuration of the navigation task world


    Attributes
    -----------
    _wconfig : ``WorldConfig``
        Configuration of the navigation task world

    """
    def __init__(self, discount, reward, controller, params, world_config):
        super(SocialNavMDP, self).__init__(discount, reward,
                                           controller, params)
        assert isinstance(world_config, WorldConfig),\
            'Expects a ``WorldConfig`` object'
        self._wconfig = world_config

    def initialize_state_graph(self, samples):
        """ Initialize graph using set of initial samples """
        # - add start and goal samples to initialization set
        for start in self._params.start_states:
            self._g.add_node(nid=self._node_id, data=start, cost=-COST_LIMIT,
                             priority=1, V=0, pi=0, Q=[0], ntype='start')
            self._node_id += 1

        self._g.add_node(nid=self._node_id, data=self._params.goal_state,
                         cost=-COST_LIMIT, priority=1, V=50, pi=0,
                         Q=[0], ntype='goal')
        self._node_id += 1

        # - add the init samples
        init_samples = list(samples)
        for sample in init_samples:
            self._g.add_node(nid=self._node_id, data=sample, cost=COST_LIMIT,
                             priority=1, V=0, pi=0, Q=[0], ntype='simple')
            self._node_id += 1

        # - add edges between each pair
        for n in self._g.nodes:
            for m in self._g.nodes:
                if n == m:
                    continue
                self._g.add_edge(source=n, target=m, reward=10, duration=10)
                self._g.add_edge(source=m, target=n, reward=1, duration=20)

    def visualize(self):
        pass

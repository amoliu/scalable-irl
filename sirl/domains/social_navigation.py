
from __future__ import division
from collections import namedtuple

import numpy as np

from ..models import LocalController
from ..models import MDPReward
from ..models import GraphMDP
from ..models import _controller_duration

from ..algorithms.mdp_solvers import graph_policy_iteration
from ..utils.geometry import edist, angle_between, distance_to_segment
from ..utils.common import eval_gaussian


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
    def __init__(self, persons, relations, goal, weights, discount,
                 kind='linfa', resolution=0.1):
        super(SocialNavReward, self).__init__(kind)
        self._persons = persons
        self._relations = relations
        self._resolution = resolution
        self._goal = goal
        self._weights = weights
        self._gamma = discount

    def __call__(self, state_a, state_b):
        source, target = np.array(state_a), np.array(state_b)
        duration = _controller_duration(source, target)*10
        action_traj = [target * t / duration + source * (1 - t / duration)
                       for t in range(int(duration))]
        action_traj.append(target)
        action_traj = np.array(action_traj)

        phi = [self.relation_disturbance(action_traj),
               self.social_disturbance(action_traj),
               self.goal_deviation_count(action_traj)]
        reward = np.dot(phi, self._weights)
        return reward

    def goal_deviation_angle(self, action):
        source, target = action[0], action[1]
        v1 = np.array([target[0]-source[0], target[1]-source[1]])
        v2 = np.array([self._goal[0]-source[0], self._goal[1]-source[1]])
        duration = _controller_duration(v1, v2)
        goal_dev = angle_between(v1, v2) * self._gamma ** duration
        return goal_dev

    def goal_deviation_count(self, action):
        """ Goal deviation measured by counts for every time
        a waypoint in the action trajectory recedes away from the goal
        """
        dist = []
        for i in range(action.shape[0]-1):
            dnow = edist(self._goal, action[i])
            dnext = edist(self._goal, action[i + 1])
            dist.append(max((dnext - dnow) * self._gamma ** i, 0))
        return sum(dist)

    def social_disturbance(self, action):
        assert isinstance(action, np.ndarray),\
            'numpy ``ndarray`` expected for action trajectory'
        phi = np.zeros(action.shape[0])
        for i, p in enumerate(action):
            for hp in self._persons:
                ed = edist(hp, p)
                if ed < 2.4:
                    phi[i] = eval_gaussian(ed, sigma=1.2) * self._gamma**i
        return np.sum(phi)

    def relation_disturbance(self, action):
        assert isinstance(action, np.ndarray),\
            'numpy ``ndarray`` expected for action trajectory'
        phi = np.zeros(action.shape[0])
        for k, act in enumerate(action):
            for (i, j) in self._relations:
                link = ((self._persons[i-1][0], self._persons[i-1][1]),
                        (self._persons[j-1][0], self._persons[j-1][1]))

                sdist, inside = distance_to_segment(act, link[0], link[1])
                if inside and sdist < 0.24:
                    phi[k] = eval_gaussian(sdist, sigma=1.2) * self._gamma**k

        return np.sum(phi)

########################################################################


WorldConfig = namedtuple('WorldConfig', ['x', 'y', 'w', 'h'])


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
        COST_LIMIT = self._params.max_cost
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

        # - update graph attributes
        self._update_state_priorities()
        graph_policy_iteration(self._g, gamma=self._gamma)
        self._find_best_policies()

    def terminal(self, state):
        """ Check if a state is terminal (goal state) """
        position = self._g.gna(state, 'data')
        if edist(position, self._params.goal_state) < 0.5:
            return True
        return False

    def visualize(self):
        pass

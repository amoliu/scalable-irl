
from __future__ import division
from abc import abstractmethod, ABCMeta
from collections import namedtuple

import numpy as np

from .state_graph import StateGraph
from utils.common import wchoice
from utils.geometry import edist


########################################################################

class LocalController(object):
    """ GraphMDP local controller """

    __metaclass__ = ABCMeta

    def __init__(self, kind='linear'):
        self.kind = kind

    @abstractmethod
    def __call__(self, state, action, duration):
        raise NotImplementedError('Abstract method')


########################################################################

class MDPReward(object):
    """ Reward  function base class """

    __metaclass__ = ABCMeta

    def __init__(self, kind='linfa'):
        self.kind = kind

    @abstractmethod
    def __call__(self, state_a, state_b):
        raise NotImplementedError('Abstract method')


########################################################################

class GraphMDP(object):
    """ Adaptive State-Graph MDP

    MDP is represented using a weighted adaptive state graph,
    .. math:
        \mathcal{G} = \langle \mathcal{S}, \mathcal{A}, w \rangle

    where the weights :math:`w \in \mathbb{R}` are costs of transitioning from
    one state to another (also intepreted as rewards)

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


    Attributes
    -----------
    _gamma : float
        MDP discount factor
    _reward : ``SocialNavReward`` object
        Reward function for social navigation task
    _controller : ``SocialNavLocalController`` object
        Local controller for the task
    _g : ``StateGraph`` object
        The underlying state graph
    _best_trajs : list of tuples, [(x, y)]
        The best trajectories representing the policies from each start pose to
        a goal pose
    _params : ``AlgoParams`` object
        Algorithm parameters for the various steps

    """

    __metaclass__ = ABCMeta

    def __init__(self, discount, reward, controller, params):
        self._gamma = discount
        self._reward = reward
        self._controller = controller
        self._params = params

        # setup the graph structure
        self._g = StateGraph()
        self._best_trajs = []
        self._node_id = 0

    @abstractmethod
    def initialize_state_graph(self, init_samples):
        """ Initialize graph using set of initial samples """
        raise NotImplementedError('Abstract method')

    def run(self):
        """ Run the adaptive state-graph procedure to solve the mdp """

        while len(self._g.nodes) < self._params.max_samples:
            # - select state expansion set between S_best and S_other
            S_best, S_other = self._generate_state_sets()
            e_set = wchoice([S_best, S_other],
                            [self._params.p_best, 1-self._params.p_best])

            for _ in range(min(len(self._g.nodes), self._params.n_exp)):
                # - select state to expand
                expansion_node = wchoice(e_set.keys(), e_set.values())

                # - expand graph from chosen state(s)
                for _ in range(self._params.n_new):
                    new_state = self._sample_new_state_from(expansion_node)
                    # - compute exploration scores
                    print(new_state)

            # - expand around exploration states (if any)

            # - update state attributes, policies
            self._update_state_costs()
            # replan with Policy iteration
            self._update_state_priorities()
            self._find_best_policies()

            # - prune graph
            self._prune_graph()

        return self

    # internals
    def _sample_new_state_from(self, state):
        """ Sample new node in the neighborhood of a given state (node)

        Sample a controller duration based on the number of nodes currently in
        the state graph, then use the sample time to execute the controller and
        add the resulting state as to the state graph.

        ..math::
            duration = \mathcal{U}(t_{min}(it, m_x), t_{max}(it, m_x))

        Parameters
        ------------
        state : int
            state id from which we will sample a new node

        Returns
        ------------
        new_state : int
            state id of the new sampled state
        """
        gna = self._g.gna
        iteration = len(self._g.nodes)
        duration = _sample_control_time(iteration, self._params.max_samples)
        action = np.random.uniform(0.0, 1.0)
        new_state = self.transition(gna(state, 'data'), action, duration*0.1)

        reward = self._reward(gna(state, 'data'), new_state)
        reward_back = self._reward(new_state, gna(state, 'data'))

        # add new state to graph
        nid = self.node_id + 1
        self._g.add_node(nid=nid, data=new_state,
                         cost=gna(state, 'cost')+reward,
                         pol=0, priority=1, Q=[0], V=10, ntype='simple')

        # add connecting edges/actions (towards and back)
        self._g.add_edge(source=gna(state, 'id'), target=nid,
                         reward=reward, duration=duration)
        self._g.add_edge(source=nid, target=gna(state, 'id'),
                         reward=reward_back, duration=duration)

        return nid

    def _update_state_costs(self):
        pass

    def _update_state_priorities(self):
        pass

    def _find_best_policies(self):
        pass

    def _prune_graph(self):
        pass

    def _generate_state_sets(self):
        """ Generate state sets, S_best and S_other

        Separate states into two sets, :math:`S_{best}, S_{other}` based on
        whether or not the states are part of the best trajectories so far.

        """
        gna = self._g.gna
        all_nodes = set(self._g.nodes)
        best_set = set()
        for traj in self._best_trajs:
            for n in traj:
                best_set.add(n)
        other_set = all_nodes.difference(best_set)

        sum_p_best = sum(gna(n, 'priority') for n in best_set)
        sum_p_other = sum(gna(n, 'priority') for n in other_set)

        S_best = {n: gna(n, 'priority')/sum_p_best for n in best_set}
        S_other = {n: gna(n, 'priority')/sum_p_other for n in other_set}

        return S_best, S_other


#############################################################################

class AlgoParams(object):
    """ Algorithm parameters """
    def __init__(self):
        self.n_exp = 1   # No of nodes to be expanded
        self.n_new = 5   # no of new nodes
        self.n_add = 1   # no of nodes to be added
        self.beta = 1.8
        self.sigma_min = 0.05
        self.max_traj_len = 40000
        self.goal_reward = 20
        self.p_best = 0.4
        self.max_samples = 100
        self.max_edges = 9
        self.start_states = ((0.5, 0.5))
        self.goal_state = (5, 5)
        self.init_type = 'random'

    def load_from_json(self, json_file):
        pass

    def save_to_json(self, filename):
        pass


#############################################################################


def _sample_control_time(iteration, max_iter):
    """ Sample a time iterval for running a local controller

    The time iterval is tempered based on the number of iterations
    """
    max_time = _tmax(iteration, max_iter)
    min_time = _tmin(iteration, max_iter)
    return np.random.uniform(0, 1) * (max_time - min_time + 1) + min_time


def _tmin(it, max_iter):
    return int(25 * (1 - it / float(max_iter)) + 5 * it / float(max_iter))


def _tmax(it, max_iter):
    return _tmin(it, max_iter) + 2


def _controller_duration(source, target):
    """
    Returns the time it takes the controller to go from source to target
    """
    return edist(source, target) / 0.2

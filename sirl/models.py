
from __future__ import division
from abc import abstractmethod, ABCMeta
from collections import namedtuple


from .state_graph import StateGraph
from utils.common import wchoice


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
    def __call__(self, state, action):
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

            # - update state attributes, policies

            # - prune graph

        return self

    # internals
    def _sample_new_state_from(self, state):
        """ Sample a new state from current state """
        pass

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
        all_nodes = set(self._g.nodes)
        best_set = set()
        for traj in self._best_trajs:
            for n in traj:
                best_set.add(n)
        other_set = all_nodes.difference(best_set)

        sum_p_best = sum(n.priority for n in best_set)
        sum_p_other = sum(n.priority for n in other_set)

        S_best = {n: n.priority/sum_p_best for n in best_set}
        S_other = {n: n.priority/sum_p_other for n in other_set}

        return S_best, S_other


# Adaptive state-graph algorithm parameters
AlgoParams = namedtuple('AlgoParams',
                        ['n_add',
                         'n_exp',
                         'beta',
                         'max_samples'])


from __future__ import division
from abc import abstractmethod, ABCMeta


from .state_graph import StateGraph


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
    """ Adaptive State-Graph MDP """

    __metaclass__ = ABCMeta

    def __init__(self, discount, reward, controller):
        self._gamma = discount
        self._reward = reward
        self._controller = controller

        # setup the graph structure
        self._g = StateGraph()

    @abstractmethod
    def initialize_state_graph(self, init_samples):
        """ Initialize graph using set of initial samples """
        raise NotImplementedError('Abstract method')

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


from __future__ import division
from abc import abstractmethod, ABCMeta

import numpy as np

from .state_graph import StateGraph
from algorithms.mdp_solvers import graph_policy_iteration

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
    def initialize_state_graph(self, samples):
        """ Initialize graph using set of initial samples """
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def terminal(self, state):
        """ Check if a state is terminal (goal state) """
        raise NotImplementedError('Abstract method')

    def run(self):
        """ Run the adaptive state-graph procedure to solve the mdp """

        while len(self._g.nodes) < self._params.max_samples:
            # - select state expansion set between S_best and S_other
            S_best, S_other = self._generate_state_sets()
            e_set = wchoice([S_best, S_other],
                            [self._params.p_best, 1-self._params.p_best])

            exp_queue = []
            exp_probs = []
            for _ in range(min(len(self._g.nodes), self._params.n_expand)):
                # - select state to expand
                anchor_node = wchoice(e_set.keys(), e_set.values())

                # - expand graph from chosen state(s)m
                for _ in range(self._params.n_new):
                    new_state = self._sample_new_state_from(anchor_node)
                    # - compute exploration score of the new state
                    es = self._exploration_score(new_state)
                    if es > self._params.exp_thresh:
                        exp_queue.append(new_state)
                        exp_probs.append(es)

            # - expand around exploration states (if any)
            exp_probs = np.array(exp_probs) / sum(exp_probs)
            for _ in range(min(len(exp_queue), self._params.n_add)):
                exploration_node = wchoice(exp_queue, exp_probs)
                self._improve_state(exploration_node)

            # - update state attributes, policies
            self._update_state_costs()
            graph_policy_iteration(self._g, gamma=self._gamma)
            self._update_state_priorities()
            self._find_best_policies()

            # - prune graph
            # self._prune_graph()

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
        print(duration)
        action = np.random.uniform(0.0, 1.0)
        new_state = self._controller(gna(state, 'data'), action, duration)

        reward = self._reward(gna(state, 'data'), new_state)
        reward_back = self._reward(new_state, gna(state, 'data'))

        # add new state to graph
        nid = self._node_id
        self._g.add_node(nid=nid, data=new_state,
                         cost=gna(state, 'cost')+reward,
                         pi=0, priority=1, Q=[0], V=10, ntype='simple')

        # add connecting edges/actions (towards and back)
        self._g.add_edge(state, nid, reward, duration)
        self._g.add_edge(self._node_id, state, reward_back, duration)

        self._node_id += 1
        return nid

    def _update_state_costs(self):
        """ Update the costs of all states in the graph """
        cmax = self._params.max_cost
        G = self._g
        converged = False
        while not converged:
            cost_changed = False
            for node in G.nodes:
                for e in G.edges(node):
                    nn = e[1]
                    cost = G.gna(node, 'cost') + G.gea(e[0], e[1], 'reward')
                    if G.gna(nn, 'cost') < cost and abs(cost) < cmax:
                        G.sna(nn, 'cost', cost)
                        cost_changed = True
            if not cost_changed:
                converged = True

    def _update_state_priorities(self, states=None):
        """ Update priority values for all states

        Updates priority score of each node in the state graph
        """
        G = self._g
        if states is None:
            states = G.nodes

        concentration = [self._node_concentration(state) for state in states]
        concentration = [c / max(concentration) for c in concentration]

        ess = [G.gna(n, 'cost') + G.gna(n, 'V') for n in states]
        if max(ess) - min(ess) > 1e-09:
            ess = [(es - min(ess)) / (max(ess) - min(ess)) for es in ess]

        for i, state in enumerate(states):
            G.sna(state, 'priority', ess[i] + concentration[i])

    def _find_best_policies(self):
        """ Find the best trajectories from starts to goal state """
        self._best_trajs = []
        G = self._g
        for start in G.filter_nodes_by_type(ntype='start'):
            # bt = [(start, G.gna(start, 'data'))]
            bt = [start]
            t = 0
            while t < self._params.max_traj_len and not self.terminal(start):
                action = G.out_edges(start)[G.gna(start, 'pi')]
                next_node = action[1]
                t += max(G.gea(action[0], action[1], 'duration'), 1)
                # bt.append((next_node, G.gna(next_node, 'data')))
                bt.append(next_node)
                start = next_node
            self._best_trajs.append(bt)

    def _improve_state(self, state):
        """ Improve a state's utility by adding connections """
        range_neighbors =\
            self._g.find_neighbors_range(state, distance=self._params.beta)

        for n in range_neighbors:
            if n != state:
                if not self._g.edge_exists(state, n):
                    xs = self._g.gna(state, 'data')
                    xn = self._g.gna(n, 'data')
                    d = edist(xs, xn)
                    reward = self._reward(xs, xn)
                    self._g.add_edge(state, n, d, reward=reward)

                    if not self._g.edge_exists(n, state):
                        reward_back = self._reward(xn, xs)
                        self._g.add_edge(state, n, d, reward=reward_back)

    def _prune_graph(self):
        """ Prune the graph

        Remove edges that are long, in which the probability of a controller
        succeeding is small (taking into account uncertainity)
        """
        G = self._g
        beta = self._params.beta

        for node in self._g.nodes:
            nedges = len(G.out_edges(node))
            if nedges > 2:
                sweep = 0
                for e in G.out_edges(node):
                    if G.gea(e[0], e[1], 'duration') > beta and\
                            len(G.out_edges(node)) > 2 and sweep < nedges:
                        G.remove_edge(e[0], e[1])
                        pol = np.argmax([G.gea(d[0], d[1], 'reward')
                                        for d in G.out_edges(node)])
                        G.sna(node, 'pi', pol)
                sweep += 1

    def _exploration_score(self, state):
        """ Exploration score :math:`p(s)`

        Compute the exporation score of a node

        Parameters
        ------------
        state : int
            State for which we should give exploration score

        Returns
        ---------
        p : float
            Exploration score of a node
        """
        c = self._node_concentration(state)
        v_estimate = 1
        return c + v_estimate

    def _node_concentration(self, state):
        """ Node concentration within a radius

        Get the list of nodes within a certain radius `_beta` from the
        selected node

        Parameters
        ------------
        state : int
            State id to compute concentration on

        Returns
        --------
        concentration : float
            Concentation of the node
        """
        neighbors = self._g.find_neighbors_range(state, self._params.beta)
        concentration = 1.0 / float(1 + len(neighbors))
        return concentration

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
        self.n_expand = 1   # No of nodes to be expanded
        self.n_new = 7   # no of new nodes
        self.n_add = 1   # no of nodes to be added
        self.beta = 1.8
        self.exp_thresh = 1.2
        self.max_traj_len = 400
        self.goal_reward = 20
        self.p_best = 0.4
        self.max_samples = 50
        self.max_edges = 9
        self.start_states = [(1, 1), (9, 5)]
        self.goal_state = (5, 8)
        self.init_type = 'random'
        self.max_cost = 1000

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
    return (np.random.uniform(0, 1) * (max_time - min_time + 1) + min_time)*0.1


def _tmin(it, max_iter):
    return int(50 * (1 - it / float(max_iter)) + 15 * it / float(max_iter))


def _tmax(it, max_iter):
    return _tmin(it, max_iter) + 2


def _controller_duration(source, target):
    """
    Returns the time it takes the controller to go from source to target
    """
    return edist(source, target)

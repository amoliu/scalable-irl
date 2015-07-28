
from __future__ import division

from abc import abstractmethod
from abc import ABCMeta

import json
import numpy as np
from numpy.random import uniform

from .state_graph import StateGraph
from algorithms.mdp_solvers import graph_policy_iteration
from algorithms.function_approximation import gp_predict, gp_covariance

from utils.common import wchoice, map_range, Timer
from utils.geometry import trajectory_length
from .base import ModelMixin


__all__ = ['GraphMDP']


class GraphMDP(ModelMixin):
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
    params : ``GraphMDPParams`` object
        Algorithm parameters for the various steps


    Attributes
    -----------
    gamma : float
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
    _params : ``GraphMDPParams`` object
        Algorithm parameters for the various steps
    _node_id : int
        State id (for keeping track when adding new states)
    _max_conc : float
        Maximum node/state concentration score
    _max_es : float
        Maximum exploration score for a state
    _min_es : float
        Minimum exploration score for a state

    """

    __metaclass__ = ABCMeta

    def __init__(self, discount, reward, controller, params):
        assert 0 <= discount < 1, '``discount`` must be in [0, 1)'
        self.gamma = discount
        self._reward = reward
        self._controller = controller
        self._params = params

        # setup the graph structure and internal variables
        self._g = StateGraph()
        self._best_trajs = []
        self._node_id = 0
        self._max_conc = 1.0
        self._max_es = 1.0
        self._min_es = 0.0

        self.data = dict()
        self.data['timing'] = []

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
        p_b = self._params.p_best
        cscale = self._params.conc_scale
        t = Timer()
        while self._node_id < self._params.max_samples:
            t.__enter__()

            if self._node_id % 10 == 0:
                print('Run: no.nodes = {}'.format(self._node_id))

            # - select state expansion set between S_best and S_other
            S_best, S_other = self._generate_state_sets()
            e_set = S_other
            if uniform(0, 1) > p_b or len(S_other) == 0:
                e_set = S_best

            exp_queue = []
            exp_probs = []
            for _ in range(min(self._node_id, self._params.n_expand)):
                # - select state to expand
                picked = False
                while not picked:
                    anchor_node = wchoice(e_set.keys(), e_set.values())
                    if not self.terminal(anchor_node):
                        picked = True
                        break

                # - expand graph from chosen state(s)m
                for _ in range(self._params.n_new):
                    new_state = self._sample_new_state_from(anchor_node)

                    # - compute exploration score of the new state
                    conc, es, var_es = self._exploration_score(new_state)
                    if conc > self._max_conc:
                        self._max_conc = conc
                    if es > self._max_es:
                        self._max_es = es
                    if es < self._min_es:
                        self._min_es = es
                    conc = conc / float(self._max_conc)
                    es = map_range(es, self._min_es, self._max_es, 0.0, 1.0)
                    # print(conc, es, var_es, self._params.exp_thresh)
                    if var_es > self._params.exp_thresh:
                        exp_queue.append(new_state)
                        exp_probs.append(es + cscale*conc)

            # - expand around exploration states (if any)
            for _ in range(min(len(exp_queue), self._params.n_add)):
                index = wchoice(np.arange(len(exp_queue)), exp_probs)
                sn = exp_queue[index]

                # add the selected node to the graph
                nid = self._node_id
                self._g.add_node(nid=nid, data=sn['data'],
                                 cost=sn['cost'], pi=0, Q=[0], V=sn['V'],
                                 ntype='simple', priority=exp_probs[index])
                self._g.add_edge(source=sn['b_state'], target=nid,
                                 reward=sn['f_reward'], phi=sn['f_phi'],
                                 duration=sn['f_duration'], traj=sn['f_traj'])

                # - compute the missing backwards phi and rewards
                b_traj = self._controller.trajectory(sn['data'], sn['b_data'],
                                                     self._params.speed)
                b_r, b_phi = self._reward(sn['data'], b_traj)
                b_d = _controller_duration(b_traj)
                self._g.add_edge(source=nid, target=sn['b_state'], reward=b_r,
                                 duration=b_d, phi=b_phi, traj=b_traj)

                # remove from queue??
                exp_queue.remove(sn)
                exp_probs.remove(exp_probs[index])

                self._node_id += 1
                self._improve_state(nid)

            # - update state attributes, policies
            self._update_state_costs()
            graph_policy_iteration(self)
            self._update_state_priorities()
            self._find_best_policies()

            t.__exit__()
            delta = t.interval
            self.data['timing'].append(delta)

        return self

    # -------------------------------------------------------------
    # properties
    # -------------------------------------------------------------

    @property
    def graph(self):
        return self._g

    # -------------------------------------------------------------
    # internals
    # -------------------------------------------------------------

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
        state_dict : dict
            dict with the attributes of the new sampled state
        """
        gna = self._g.gna
        iteration = len(self._g.nodes)
        duration = _sample_control_time(iteration, self._params.max_samples)

        cs = gna(state, 'data')
        vmax = self._params.speed

        succeeded = False
        while not succeeded:
            action = uniform(0.0, 2.0*np.pi)
            ns, f_traj = self._controller(cs, action, duration, vmax)
            if f_traj is not None:
                succeeded = True
                break

        # - can be costly, only compute the forward case here
        reward, phi = self._reward(state=gna(state, 'data'), action=f_traj)

        state_dict = dict()
        state_dict['data'] = ns
        state_dict['cost'] = gna(state, 'cost')+reward

        # - forwards info
        state_dict['f_reward'] = reward
        state_dict['f_phi'] = phi
        state_dict['f_traj'] = f_traj
        state_dict['f_duration'] = _controller_duration(f_traj)

        # - backwards info
        state_dict['b_state'] = state
        state_dict['b_data'] = cs
        return state_dict

    def _update_state_costs(self):
        """ Update the costs of all states in the graph

        Given:  (n1) ---r1---- (n2) ----r2---- (n3)

        cost(n1) = cost(n1)
        cost(n2) = cost(n1) + r1
        ...

        """
        # TODO - create a node visitor from starts to goal??
        # - What about multiple goal start cases?
        cmax = self._params.max_cost
        G = self._g
        converged = False
        while not converged:
            cost_changed = False
            for node in G.nodes:
                for e in G.out_edges(node):
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

        cc = [self._node_concentration(state) for state in states]
        self._max_conc = max(cc)
        cc = [c / float(self._max_conc) for c in cc]

        ess = [G.gna(n, 'cost') + G.gna(n, 'V') for n in states]
        self._max_es = max(ess)
        self._min_es = min(ess)
        ess = [(s - self._min_es) / float((self._max_es - self._min_es))
               for s in ess]

        cscale = self._params.conc_scale
        for i, state in enumerate(states):
            G.sna(state, 'priority', ess[i] + cscale*cc[i])

    def _find_best_policies(self):
        """ Find the best trajectories from starts to goal state """
        self._best_trajs = []
        G = self._g
        for start in G.filter_nodes_by_type(ntype='start'):
            bt = [start]
            t = 0
            while t < self._params.max_traj_len and not self.terminal(start):
                action = G.out_edges(start)[G.gna(start, 'pi')]
                next_node = action[1]
                t += max(G.gea(start, next_node, 'duration'), 1.0)
                start = next_node
                if start not in bt:
                    bt.append(start)
            self._best_trajs.append(bt)

    def _improve_state(self, s):
        """ Improve a state's utility by adding connections """
        neighbors = self._g.find_neighbors_range(s, self._params.radius)
        vmax = self._params.speed
        for n in neighbors:
            if n != s:
                xs = self._g.gna(s, 'data')
                xn = self._g.gna(n, 'data')
                if len(self._g.out_edges(s)) < self._params.max_edges:
                    if not self._g.edge_exists(s, n) and not self.terminal(s):
                        traj = self._controller.trajectory(xs, xn, vmax)
                        d = _controller_duration(traj)
                        reward, phi = self._reward(xs, traj)

                        self._g.add_edge(source=s, target=n, phi=phi,
                                         duration=d, reward=reward, traj=traj)
                if len(self._g.out_edges(n)) < self._params.max_edges:
                    if not self._g.edge_exists(n, s) and not self.terminal(n):
                        traj = self._controller.trajectory(xn, xs, vmax)
                        d = _controller_duration(traj)
                        rb, phi = self._reward(xn, traj)
                        self._g.add_edge(source=n, target=s, phi=phi,
                                         duration=d, reward=rb, traj=traj)

    def _exploration_score(self, state_dict):
        """ Exploration score :math:`p(s)`

        Compute the exporation score of a node

        Parameters
        ------------
        state_dict : int
            Dict with parameters of the state for which we should compute
            the exploration score

        Returns
        ---------
        conc : float
            Concentation score for the state
        p : float
            Exploration score of a node
        sigma : float
            Variance of the exploration score
        """
        nn = self._g.find_neighbors_data(state_dict['data'],
                                         self._params.radius)
        concentration = 1.0 / float(1 + len(nn))
        node_cost = state_dict['cost']
        if len(nn) < 1:
            y = 10
            state_dict['V'] = y
            return concentration, node_cost+y, 1

        train_data = [self._g.gna(n, 'data') for n in nn]
        train_values = [self._g.gna(n, 'V') for n in nn]

        gram = gp_covariance(train_data, train_data)
        y, v = gp_predict(state_dict['data'], train_data, gram, train_values)

        state_dict['V'] = y
        return concentration, (node_cost + y), v

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
        neighbors = self._g.find_neighbors_range(state, self._params.radius)
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

#############################################################################


def _sample_control_time(iteration, max_iter):
    """ Sample a time iterval for running a local controller

    The time iterval is tempered based on the number of iterations
    """
    max_time = _tmax(iteration, max_iter)
    min_time = _tmin(iteration, max_iter)
    delta = uniform(min_time, max_time)
    return delta


def _tmin(it, max_iter):
    return 2.4 * (1 - it/float(max_iter)) + 0.45*it/float(max_iter)


def _tmax(it, max_iter):
    return 7.2 * (1 - it/float(max_iter)) + 3.6*it/float(max_iter)


def _controller_duration(action):
    """
    Returns the time it takes the controller to execute an action represented
    by a trajectory.
    """
    return trajectory_length(action)

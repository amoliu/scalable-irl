r"""
ControllerGraph

A representation learning algorithm for MDPs.
MDP is represented using a weighted labeled directed graph,

    .. math:
        \mathcal{G} = \langle \mathcal{S}, \mathcal{A}, \mathbf{W} \rangle

    where the weights :math:`w_{i,j} \in \mathbb{R}` are transition
    probabilities from state (:math:`s_i`) to (:math:`s_j`), i.e.
    :math:`p(s' | s, a)`

Graph is initialized using a variety of procedures, while growing of the graph
is done on-line my mixing exploration and exploitation guided by heuristics

"""
from __future__ import division

import logging
import copy

import numpy as np
from numpy.random import uniform

from ..algorithms.mdp_solvers import graph_policy_iteration
from ..algorithms.function_approximation import gp_predict, gp_covariance

from ..utils.common import wchoice, map_range
from ..utils.common import Logger

from ..utils.geometry import trajectory_length

from ..models.state_graph import StateGraph
from ..models.base import MDPRepresentation


class ControllerGraph(MDPRepresentation, Logger):
    """ A ControllerGraph

    Parameters
    ------------
    mdp : `MDP` object
        MDP object (describing the state, action, transitions, etc)
    controller : `SocialNavLocalController` object
        Local controller for the task
    params : `GraphMDPParams` object
        Algorithm parameters for the various steps


    Attributes
    -----------
    _controller : :class:`SocialNavLocalController` object
        Local controller for the task
    _g : :class:`StateGraph` object
        The underlying state graph
    _best_trajs : list of tuples, [(x, y)]
        The best trajectories representing the policies from each start pose to
        a goal pose
    _params : :class:`GraphMDPParams` object
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
    def __init__(self, mdp, local_controller, params):
        super(ControllerGraph, self).__init__(mdp)

        self._controller = local_controller
        self._params = params

        # setup the graph structure and internal variables
        self._g = StateGraph(state_dim=mdp.state_dimension)
        self._best_trajs = []
        self._node_id = 0
        self._max_conc = 1.0
        self._max_es = 1.0
        self._min_es = 0.0

        self.log_config(logging.DEBUG)

    def initialize_state_graph(self, samples):
        """ Initialize graph using set of initial samples

        If random, samples are states
        If trajectory, samples are trajectories

        """
        self._g.clear()

        if self._params.init_type == 'random':
            self._fixed_init(samples)
        elif self._params.init_type == 'trajectory':
            self._traj_init(samples)

        # - update graph attributes
        self._update_state_costs()
        graph_policy_iteration(self._g, self._mdp.gamma)
        self._update_state_priorities()
        self.find_best_policies()

    def run(self):
        """ Run the adaptive state-graph procedure to solve the mdp """
        p_b = self._params.p_best
        cscale = self._params.conc_scale
        while self._node_id < self._params.max_samples:
            if self._node_id % 10 == 0:
                self.info('Run: no.nodes = {}'.format(self._node_id))

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
                    xn = wchoice(e_set.keys(), e_set.values())
                    if not self._mdp.terminal(self._g.gna(xn, 'data')):
                        picked = True
                        break

                # - expand graph from chosen state(s)m
                for _ in range(self._params.n_new):
                    new_state = self._sample_new_state_from(xn)

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
                    # self.debug('{}, {}, {}, {}'
                    #            .format(conc, es, var_es, len(exp_queue)))
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
                b_r, b_phi = self._mdp.reward(sn['data'], b_traj)
                b_d = trajectory_length(b_traj)
                self._g.add_edge(source=nid, target=sn['b_state'], reward=b_r,
                                 duration=b_d, phi=b_phi, traj=b_traj)

                # remove from queue??
                # exp_queue.remove(sn)
                # exp_probs.remove(exp_probs[index])
                exp_queue = exp_queue[:index] + exp_queue[index+1:]
                exp_probs = exp_probs[:index] + exp_probs[index+1:]

                self._node_id += 1
                self._improve_state(nid)

            # - update state attributes, policies
            self._update_state_costs()
            graph_policy_iteration(self.graph, self.mdp.gamma)
            self._update_state_priorities()
            self.find_best_policies()

        return self

    def find_best_policies(self):
        """ Find the best trajectories from starts to goal state """
        self._best_trajs = []
        G = self._g
        for start in G.filter_nodes_by_type(ntype='start'):
            bt = [start]
            t = 0
            while t < self._params.max_traj_len and \
                    not self._mdp.terminal(G.gna(start, 'data')):
                action = G.out_edges(start)[G.gna(start, 'pi')]
                next_node = action[1]
                t += max(G.gea(start, next_node, 'duration'), 1.0)
                start = next_node
                if start not in bt:
                    bt.append(start)
            self._best_trajs.append(bt)

        return self._best_trajs

    def update_rewards(self, new_reward):
        """ Update the reward for all edges in the graph """
        new_reward = np.asarray(new_reward)
        assert new_reward.size == self.mdp.reward.dim,\
            'weight vector and feature vector dimensions do not match'

        gea = self.graph.gea
        sea = self.graph.sea

        for e in self.graph.all_edges:
            phi = gea(e[0], e[1], 'phi')
            r = np.dot(phi, new_reward)
            sea(e[0], e[1], 'reward', r)

        return self

    def trajectory_quality(self, reward, trajs):
        """ Compute the Q-function of a set of trajectories

        Compute the action-value function of a set of trajectories using the
        specified reward function, on the MDP representation

        """
        G = self.graph
        gr = self._params.goal_reward
        gamma = self._mdp.gamma

        q_trajs = []
        for traj in trajs:
            duration = 0
            q_traj = 0
            for n in traj:
                actions = G.out_edges(n)
                if actions:  # if no edges, use goal reward???
                    e = actions[G.gna(n, 'pi')]
                    r = np.dot(reward, G.gea(e[0], e[1], 'phi'))
                    q_traj += (gamma ** duration) * r
                    duration += G.gea(e[0], e[1], 'duration')
                else:
                    q_traj += (gamma ** duration) * gr
            q_trajs.append(q_traj)
        return q_trajs

    # -------------------------------------------------------------
    # properties
    # -------------------------------------------------------------

    @property
    def graph(self):
        return self._g

    @property
    def policies(self):
        return self._best_trajs

    @property
    def mdp(self):
        return self._mdp

    @property
    def kind(self):
        return 'graph'

    # -------------------------------------------------------------
    # internals
    # -------------------------------------------------------------

    def _fixed_init(self, samples):
        """ Initialize from random samples """
        GR = self._params.goal_reward
        CLIMIT = self._params.max_cost
        # TODO - make these depend on state size (maybe world param?)
        GOAL = list(self._mdp.goal_state) + [0, self._params.speed]
        # GOAL = self._mdp.goal_state

        for start in self._mdp.start_states:
            st = list(start) + [0, self._params.speed]
            # st = start
            self._g.add_node(nid=self._node_id, data=st, cost=0,
                             priority=1, V=GR, pi=0, Q=[], ntype='start')
            self._node_id += 1

        self._g.add_node(nid=self._node_id, data=GOAL, cost=-CLIMIT,
                         priority=1, V=GR, pi=0, Q=[], ntype='goal')
        self._node_id += 1

        # - add the init samples
        init_samples = list(samples)
        for sample in init_samples:
            smp = list(sample) + [0, self._params.speed]
            # smp = sample
            self._g.add_node(nid=self._node_id, data=smp, cost=-CLIMIT,
                             priority=1, V=GR, pi=0, Q=[], ntype='simple')
            self._node_id += 1

        # - add edges between each pair
        for n in self._g.nodes:
            for m in self._g.nodes:
                if n == m or self._mdp.terminal(self._g.gna(n, 'data')):
                    continue
                ndata, mdata = self._g.gna(n, 'data'), self._g.gna(m, 'data')
                traj = self._controller.trajectory(ndata, mdata,
                                                   self._params.speed)
                d = trajectory_length(traj)
                r, phi = self._mdp.reward(ndata, traj)
                self._g.add_edge(source=n, target=m, reward=r,
                                 duration=d, phi=phi, traj=traj)

    def _traj_init(self, trajs):
        """ Initialize from trajectories """
        GR = self._params.goal_reward
        CLIMIT = self._params.max_cost
        GOAL = list(self._mdp.goal_state) + [0, self._params.speed]
        self._g.add_node(nid=self._node_id, data=GOAL,
                         cost=-CLIMIT, priority=1, V=GR, pi=0,
                         Q=[], ntype='goal')
        g = copy.copy(self._node_id)
        self._node_id += 1

        self._params.start_states = []
        vmax = self._params.speed
        for traj in trajs:
            # - add start
            start = traj[0]
            smp = list(start) + [0, self._params.speed]
            self._params.start_states.append(start)
            self._g.add_node(nid=self._node_id, data=smp, cost=0,
                             priority=1, V=GR, pi=0, Q=[], ntype='start')
            n = copy.copy(self._node_id)
            self._node_id += 1

            # - add the rest of the waypoints
            for wp in traj[1:-1]:
                sp = list(wp) + [0, self._params.speed]
                self._g.add_node(nid=self._node_id, data=sp, cost=-CLIMIT,
                                 priority=1, V=GR, pi=0, Q=[], ntype='simple')
                m = copy.copy(self._node_id)
                ndata, mdata = self._g.gna(n, 'data'), self._g.gna(m, 'data')
                traj = self._controller.trajectory(ndata, mdata, vmax)
                d = trajectory_length(traj)
                r, phi = self._mdp.reward(ndata, traj)
                self._g.add_edge(source=n, target=m, reward=r,
                                 duration=d, phi=phi, traj=traj)
                n = copy.copy(self._node_id)
                self._node_id += 1

            fdata, tdata = self._g.gna(m, 'data'), self._g.gna(g, 'data')
            traj = self._controller.trajectory(fdata, tdata, vmax)
            d = trajectory_length(traj)
            r, phi = self._mdp.reward(fdata, traj)
            self._g.add_edge(source=m, target=g, reward=r,
                             duration=d, phi=phi, traj=traj)

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
        duration = self._sample_control_time(iteration,
                                             self._params.max_samples)

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
        reward, phi = self._mdp.reward(state=gna(state, 'data'), action=f_traj)

        state_dict = dict()
        state_dict['data'] = ns
        state_dict['cost'] = gna(state, 'cost')+reward

        # - forwards info
        state_dict['f_reward'] = reward
        state_dict['f_phi'] = phi
        state_dict['f_traj'] = f_traj
        state_dict['f_duration'] = trajectory_length(f_traj)

        # - backwards info
        state_dict['b_state'] = state
        state_dict['b_data'] = cs
        return state_dict

    def _update_state_costs(self):
        """ Update the costs of all states in the graph

        Estimate the costs of all vertices in the graph in an optimistic way

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

    def _update_state_priorities(self):
        """ Update priority values for all states

        Updates priority score of each node in the state graph
        """
        G = self._g
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

    def _improve_state(self, s):
        """ Improve a state's utility by adding connections """
        neighbors = self._g.find_neighbors_range(s, self._params.radius)
        vmax = self._params.speed
        for n in neighbors:
            if n != s:
                xs = self._g.gna(s, 'data')
                xn = self._g.gna(n, 'data')
                if len(self._g.out_edges(s)) < self._params.max_edges:
                    if not self._g.edge_exists(s, n) and\
                            not self._mdp.terminal(self._g.gna(s, 'data')):
                        traj = self._controller.trajectory(xs, xn, vmax)
                        d = trajectory_length(traj)
                        reward, phi = self._mdp.reward(xs, traj)

                        self._g.add_edge(source=s, target=n, phi=phi,
                                         duration=d, reward=reward, traj=traj)
                if len(self._g.out_edges(n)) < self._params.max_edges:
                    if not self._g.edge_exists(n, s) and\
                            not self._mdp.terminal(self._g.gna(n, 'data')):
                        traj = self._controller.trajectory(xn, xs, vmax)
                        d = trajectory_length(traj)
                        rb, phi = self._mdp.reward(xn, traj)
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
        nn = self._g.find_neighbors_from_pose(state_dict['data'],
                                              self._params.radius)
        concentration = 1.0 / float(1 + len(nn))
        node_cost = state_dict['cost']
        if len(nn) < 1:
            y = self._params.goal_reward
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
            Concentration of the node

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
        best_set = set(n for traj in self._best_trajs for n in traj)
        other_set = all_nodes.difference(best_set)

        sum_p_best = sum(gna(n, 'priority') for n in best_set)
        sum_p_other = sum(gna(n, 'priority') for n in other_set)

        S_best = {n: gna(n, 'priority')/sum_p_best for n in best_set}
        S_other = {n: gna(n, 'priority')/sum_p_other for n in other_set}

        return S_best, S_other

    def _sample_control_time(self, i, imax):
        """ Sample a time interval for running a local controller

        The time interval is tempered based on the number of iterations

        """
        imax = float(imax)
        tmin, tmax = self._params.tmin, self._params.tmax
        min_time = tmin[1] * (1 - i/imax) + tmin[0] * i/imax
        max_time = tmax[1] * (1 - i/imax) + tmax[0] * i/imax
        return uniform(min_time, max_time)

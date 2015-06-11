
from __future__ import division
from collections import namedtuple

import numpy as np

from matplotlib.patches import Circle, Ellipse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl


from ..models import LocalController
from ..models import MDPReward
from ..models import GraphMDP
from ..models import _controller_duration

from ..algorithms.mdp_solvers import graph_policy_iteration
from ..utils.geometry import edist, angle_between, distance_to_segment
from ..utils.geometry import line_crossing
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
        nx = state[0] + np.cos(action * 2 * np.pi) * duration * 0.1
        ny = state[1] + np.sin(action * 2 * np.pi) * duration * 0.1

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
        # increase resolution of action trajectory (option)
        duration = _controller_duration(source, target)
        action_traj = [target * t / duration + source * (1 - t / duration)
                       for t in range(int(duration))]
        action_traj.append(target)
        action_traj = np.array(action_traj)

        phi = [self.relation_disturbance(action_traj),
               self.social_disturbance(action_traj),
               # self.goal_deviation_angle((source, target))]
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
        pd = [min([edist(wp, person) for person in self._persons])
              for wp in action]
        phi = sum(1 * self._gamma**i for i, d in enumerate(pd) if d < 0.25)
        return phi

    def social_disturbance2(self, action):
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
        # TODO - fix relations to start from 0 instead of 1
        atime = action.shape[0]
        c = [sum(line_crossing(action[t][0],
                 action[t][1],
                 action[t+1][0],
                 action[t+1][1],
                 self._persons[i-1][0],
                 self._persons[i-1][1],
                 self._persons[j-1][0],
                 self._persons[j-1][1])
             for [i, j] in self._relations) for t in range(int(atime - 1))]
        ec = sum(self._gamma**i * x for i, x in enumerate(c))
        return ec

    def relation_disturbance2(self, action):
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
        GR = self._params.goal_reward
        COST_LIMIT = self._params.max_cost
        for start in self._params.start_states:
            self._g.add_node(nid=self._node_id, data=start, cost=0,
                             priority=1, V=GR, pi=0, Q=[], ntype='start')
            self._node_id += 1

        self._g.add_node(nid=self._node_id, data=self._params.goal_state,
                         cost=-COST_LIMIT, priority=1, V=GR, pi=0,
                         Q=[], ntype='goal')
        self._node_id += 1

        # - add the init samples
        init_samples = list(samples)
        for sample in init_samples:
            self._g.add_node(nid=self._node_id, data=sample, cost=-COST_LIMIT,
                             priority=1, V=GR, pi=0, Q=[], ntype='simple')
            self._node_id += 1

        print(self._g.nodes_data)
        # - add edges between each pair
        for n in self._g.nodes:
            for m in self._g.nodes:
                if n == m:
                    continue
                ndata, mdata = self._g.gna(n, 'data'), self._g.gna(m, 'data')
                r = self._reward(ndata, mdata)
                d = _controller_duration(ndata, mdata)
                self._g.add_edge(source=n, target=m, reward=r, duration=d)
                # rb = self._reward(mdata, ndata)
                # self._g.add_edge(source=m, target=n, reward=rb, duration=d)

        # - update graph attributes
        self._update_state_costs()
        graph_policy_iteration(self._g, gamma=self._gamma)
        self._update_state_priorities()
        self._find_best_policies()
        print(self._g.edges(0))
        print(self._g.out_edges(0))

    def terminal(self, state):
        """ Check if a state is terminal (goal state) """
        position = self._g.gna(state, 'data')
        if edist(position, self._params.goal_state) < 0.05:
            return True
        return False

    def visualize(self, persons, relations):
        """ Visualize the social navigation world

        Allows recording of demonstrations and also display of final
        graph representing the MDP
        """
        self._setup_visuals()

        for p in persons:
            phead = np.degrees(np.arctan2(p[3], p[2]))
            self.ax.add_artist(Ellipse((p[0], p[1]), width=0.3, height=0.6,
                               angle=phead, color='r', fill=False, lw=1.5,
                               aa=True))
            self.ax.add_artist(Circle((p[0], p[1]), radius=0.12, color='w',
                               ec='r', lw=2.5, aa=True))
            self.ax.arrow(p[0], p[1], p[2]/5., p[3]/5., fc='r', ec='r', lw=1.5,
                          head_width=0.14, head_length=0.1)

        for [i, j] in relations:
            x1, x2 = persons[i-1][0], persons[i-1][1]
            y1, y2 = persons[j-1][0], persons[j-1][1]
            self.ax.plot((x1, y1), (x2, y2), ls='-', lw=2.0, c='r', alpha=0.7)

        self._plot_graph_in_world()

        return self.ax

    def _setup_visuals(self):
        """ Prepare figure axes for plotting """
        self.figure = plt.figure(figsize=(12, 9))
        self.ax = plt.axes([0, 0, 0.8, 1])
        self.figure.add_axes(self.ax)

        self.ax.set_xlim([self._wconfig.x, self._wconfig.w])
        self.ax.set_ylim([self._wconfig.y, self._wconfig.h])

        self.record_status = self.figure.text(0.825, 0.3, 'Recording [OFF]',
                                              fontsize=14, color='blue')
        self.figure.text(0.825, 0.2, '#Demos: ', fontsize=10)
        self.demo_count = self.figure.text(0.925, 0.2, '0', fontsize=10)

        # self.figure.canvas.mpl_connect('key_press_event', self._key_press)
        # self.figure.canvas.mpl_connect('button_press_event', self._btn_click)

    def _plot_graph_in_world(self):
        """ Shows the lattest version of the world with MDP
        """
        G = self._g
        gna = G.gna
        gea = G.gea

        rewards = [gea(e[0], e[1], 'reward') for e in G.all_edges()]
        norm = mpl.colors.Normalize(vmin=np.min(rewards), vmax=np.max(rewards))
        m = cm.ScalarMappable(norm=norm, cmap=cm.jet)

        n_nodes = len(G.nodes)
        best_nodes = set()
        for traj in self._best_trajs:
            for state in traj:
                best_nodes.add(state)

        for i, n in enumerate(G.nodes):
            [posx, posy] = gna(n, 'data')
            if [posx, posy] in self._params.start_states:
                color = ['black', 'black']
                nr = 1.0
            elif self.terminal(n):
                color = ['green', 'black']
                nr = 1.5
            elif n in best_nodes:
                color = ['green', 'green']
                nr = 0.5
            else:
                rgcol = _rgb_to_hex(((0, 0, 255 * i / float(n_nodes))))
                color = [rgcol, rgcol]
                nr = 0.5

            self.ax.add_artist(Circle((posx, posy), nr/10., fc=color[0],
                               ec=color[1], lw=1.5))

            p = gna(n, 'pi')
            ndata = gna(n, 'data')
            for i, e in enumerate(G.out_edges(n)):
                t = e[1]
                tdata = gna(t, 'data')
                x1, y1 = ndata[0], ndata[1]
                x2, y2 = tdata[0], tdata[1]
                if n in best_nodes and i == p:
                    self.ax.plot((x1, y1), (x2, y2), ls='-', lw=4.0, c='g')
                else:
                    # self.ax.plot((x1, x2), (y1, y2), ls='-', lw=1.0,
                    #              c='k', alpha=0.5)

                    cost = gea(e[0], e[1], 'reward')
                    self.ax.arrow(x1, y1, 0.97*(x2-x1), 0.97*(y2-y1),
                                  width=0.01, head_width=0.15,
                                  head_length=0.15,
                                  fc=m.to_rgba(cost), ec=m.to_rgba(cost))


def _rgb_to_hex(rgb):
    return ('#%02X%02X%02X' % (rgb[0], rgb[1], rgb[2]))

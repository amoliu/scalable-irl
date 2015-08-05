
from __future__ import division
from collections import namedtuple
from copy import copy

from scipy.spatial import Voronoi

import numpy as np

from matplotlib.patches import Circle, Ellipse, Polygon
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

from ...models.mdp import GraphMDP

from ...utils.geometry import edist
from ...utils.geometry import trajectory_length
from ...algorithms.mdp_solvers import graph_policy_iteration


WorldConfig = namedtuple('WorldConfig', ['x', 'y', 'w', 'h'])


class SocialNavMDP(GraphMDP):
    """ Social Navigation Adaptive State-Graph (SocialNavMDP)

    Social navigation task MDP represented by an adaptive state graph

    Parameters
    ------------
    discount : float
        MDP discount factor
    reward : :class:`SocialNavReward` object
        Reward function for social navigation task
    controller : :class:`SocialNavLocalController` object
        Local controller for the task
    params : :class:`GraphMDPParams` object
        Algorithm parameters for the various steps
    world_config : :class:`WorldConfig` object
        Configuration of the navigation task world


    Attributes
    -----------
    _wconfig : :class:`WorldConfig`
        Configuration of the navigation task world

    """
    def __init__(self, discount, reward, controller, params, world_config,
                 persons, relations):
        super(SocialNavMDP, self).__init__(discount, reward,
                                           controller, params)
        self._wconfig = world_config
        self._persons = persons
        self._relations = relations

        # manual demonstration recording
        self._recording = False
        self._demos = list()

    def initialize_state_graph(self, samples):
        """ Initialize graph using set of initial samples """
        self._g.clear()

        if self._params.init_type == 'random':
            if samples is None:
                samples = []
                udist = np.random.uniform
                x, dx = self._wconfig.x, self._wconfig.w
                y, dy = self._wconfig.y, self._wconfig.h
                for _ in range(10):
                    samples.append([udist(x, x+dx), udist(y, y+dy)])

            self._fixed_init(samples)

        elif self._params.init_type == 'homotopy':
            samples = [[p[0], p[1]] for _, p in self._persons.items()]
            vor = Voronoi(samples)
            self._fixed_init(samples)

        elif self._params.init_type == 'trajectory':
            self._traj_init(samples)

    def terminal(self, state):
        """ Check if a state is terminal (goal state) """
        position = self._g.gna(state, 'data')
        if edist(position, self._params.goal_state) < 0.05:
            return True
        return False

    def visualize(self, annotations=None, fsize=(12, 9)):
        """ Visualize the social navigation world

        Allows recording of demonstrations and also display of final
        graph representing the MDP
        """
        self._setup_visuals(fsize)

        for _, p in self._persons.items():
            phead = np.degrees(np.arctan2(p[3], p[2]))
            self.ax.add_artist(Ellipse((p[0], p[1]), width=0.3, height=0.6,
                               angle=phead, color='r', fill=False, lw=1.5,
                               aa=True, zorder=3))
            self.ax.add_artist(Circle((p[0], p[1]), radius=0.12, color='w',
                               ec='r', lw=2.5, aa=True, zorder=3))
            self.ax.arrow(p[0], p[1], p[2]/5., p[3]/5., fc='r', ec='r', lw=1.5,
                          head_width=0.14, head_length=0.1, zorder=3)

            speed = np.hypot(p[2], p[3])
            hz = speed * 0.55
            self.ax.add_artist(Circle((p[0], p[1]), radius=hz, color='r',
                               ec='r', lw=1, aa=True, alpha=0.2))

        for [i, j] in self._relations:
            x1, y1 = self._persons[i][0], self._persons[i][1]
            x2, y2 = self._persons[j][0], self._persons[j][1]
            self.ax.plot((x1, x2), (y1, y2), ls='-', c='r', lw=2.0, zorder=2)

        # for a in annotations:
        #     self.ax.add_artist(Polygon(a.geometry, hatch='x'))
        #     self.ax.add_artist(Polygon(a.influence_zone, alpha=0.3))

        self._plot_graph_in_world()

        return self.ax

    # -------------------------------------------------------------
    # internals
    # -------------------------------------------------------------

    def _fixed_init(self, samples):
        """ Initialize from random samples """
        GR = self._params.goal_reward
        CLIMIT = self._params.max_cost
        GOAL = list(self._params.goal_state) + [0, self._params.speed]

        for start in self._params.start_states:
            st = list(start) + [0, self._params.speed]
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
            self._g.add_node(nid=self._node_id, data=smp, cost=-CLIMIT,
                             priority=1, V=GR, pi=0, Q=[], ntype='simple')
            self._node_id += 1

        # - add edges between each pair
        for n in self._g.nodes:
            for m in self._g.nodes:
                if n == m or self.terminal(n):
                    continue
                ndata, mdata = self._g.gna(n, 'data'), self._g.gna(m, 'data')
                traj = self._controller.trajectory(ndata, mdata,
                                                   self._params.speed)
                d = trajectory_length(traj)
                r, phi = self._reward(ndata, traj)
                self._g.add_edge(source=n, target=m, reward=r,
                                 duration=d, phi=phi, traj=traj)

        # - update graph attributes
        self._update_state_costs()
        graph_policy_iteration(self)
        self._update_state_priorities()
        self._find_best_policies()

    def _traj_init(self, trajs):
        """ Initialize from trajectories """
        GR = self._params.goal_reward
        CLIMIT = self._params.max_cost

        # - goal state
        self._g.add_node(nid=self._node_id, data=self._params.goal_state,
                         cost=-CLIMIT, priority=1, V=GR, pi=0,
                         Q=[], ntype='goal')
        g = copy(self._node_id)
        self._node_id += 1

        self._params.start_states = []
        for traj in trajs:
            # - add start
            start = traj[0]
            self._params.start_states.append(start)
            self._g.add_node(nid=self._node_id, data=start, cost=0,
                             priority=1, V=GR, pi=0, Q=[], ntype='start')
            n = copy(self._node_id)
            self._node_id += 1

            # - add the rest of the waypoints
            for wp in traj[1:]:
                self._g.add_node(nid=self._node_id, data=wp, cost=-CLIMIT,
                                 priority=1, V=GR, pi=0, Q=[], ntype='simple')
                m = copy(self._node_id)
                ndata, mdata = self._g.gna(n, 'data'), self._g.gna(m, 'data')
                traj = self._controller.trajectory(ndata, mdata,
                                                   self._params.speed)
                d = trajectory_length(traj)
                r, phi = self._reward(ndata, traj)
                self._g.add_edge(source=n, target=m, reward=r,
                                 duration=d, phi=phi, traj=traj)
                n = copy(self._node_id)
                self._node_id += 1

            fdata, tdata = self._g.gna(m, 'data'), self._g.gna(g, 'data')
            traj = self._controller.trajectory(fdata, tdata,
                                               self._params.speed)
            d = trajectory_length(traj)
            r, phi = self._reward(fdata, traj)
            self._g.add_edge(source=m, target=g, reward=r,
                             duration=d, phi=phi, traj=traj)

        # - update graph attributes
        self._update_state_costs()
        graph_policy_iteration(self)
        self._update_state_priorities()
        self._find_best_policies()

    def _setup_visuals(self, fsize=(12, 9)):
        """ Prepare figure axes for plotting """
        self.figure = plt.figure(figsize=fsize)
        self.ax = plt.axes([0, 0, 0.8, 1])
        self.figure.add_axes(self.ax)
        self.ax.set_xlim([self._wconfig.x, self._wconfig.w])
        self.ax.set_ylim([self._wconfig.y, self._wconfig.h])

        self.record_status = self.figure.text(0.825, 0.3, 'Recording [OFF]',
                                              fontsize=14, color='blue')
        self.figure.text(0.825, 0.2, '#Demos: ', fontsize=10)
        self.demo_count = self.figure.text(0.925, 0.2, '0', fontsize=10)

        self.figure.canvas.mpl_connect('key_press_event', self._key_press)
        self.figure.canvas.mpl_connect('button_press_event', self._btn_click)

    def _key_press(self, event):
        """Handler for key press events
        Used for demonstration recording and experiment handling
        """
        if event.key == 'R':
            print('Starting recordning demos')
            if self._recording is False:
                print('Recording new demonstration')
                self.record_status.set_text("Recording [ON]")
                self.new_demo = list()
                self.demo_color = np.random.choice(['r', 'b', 'g', 'k',
                                                   'm', 'y', 'c'])
                self._recording = True
            else:
                print('Done recording demo')
                self._demos.append(np.array(self.new_demo))
                self.record_status.set_text("Recording [OFF]")
                self.demo_count.set_text(str(len(self._demos)))
                self._recording = False

                if len(self._demos):
                    last_demo = self._demos[len(self._demos) - 1]
                    self.ax.plot(last_demo[:, 0], last_demo[:, 1],
                                 color=self.demo_color, ls='-', lw=1)
        elif event.key == 'S':
            if self._recording:
                print('Please finish recording before saving')
            else:
                print('Saving demos as: demos.npy')
                d = np.array(self._demos)
                fname = 'demos_metropolis.npy'
                print(d, fname)
                np.save(fname, d)
        self.figure.canvas.draw()

    def _btn_click(self, event):
        if self._recording:
            self.new_demo.append([event.xdata, event.ydata])
            cc = self.demo_color
            self.ax.add_artist(Circle((event.xdata, event.ydata),
                               0.03, fc=cc, ec=cc))
            self.figure.canvas.draw()

    def _plot_graph_in_world(self, show_rewards=False):
        """ Shows the lattest version of the world with MDP
        """
        G = self._g
        gna = G.gna
        gea = G.gea

        values = [gna(n, 'V') for n in G.nodes]
        nv = mpl.colors.Normalize(vmin=min(values), vmax=max(values))
        mv = cm.ScalarMappable(norm=nv, cmap=cm.jet)

        best_nodes = set()
        for traj in self._best_trajs:
            for state in traj:
                best_nodes.add(state)

        for i, n in enumerate(G.nodes):
            posx, posy, _, _ = gna(n, 'data')
            if gna(n, 'type') == 'start':
                color = 'black'
                nr = 1.0
            elif self.terminal(n):
                color = 'green'
                nr = 1.5
            elif n in best_nodes:
                color = 'green'
                nr = 0.5
            else:
                color = mv.to_rgba(gna(n, 'V'))
                nr = 0.5
            self.ax.add_artist(Circle((posx, posy), nr/10., fc=color,
                               ec=color, lw=1.5, zorder=3))

            p = gna(n, 'pi')
            for i, e in enumerate(G.out_edges(n)):
                if n in best_nodes and i == p:
                    traj = gea(e[0], e[1], 'traj')
                    for wp in traj:
                        v = wp[3]
                        vx, vy = v*np.cos(wp[2]), v*np.sin(wp[2])
                        self.ax.arrow(wp[0], wp[1], 0.5*vx, 0.5*vy, fc='g',
                                      ec='g', lw=1.0, head_width=0.1,
                                      head_length=0.08, zorder=3)


# -------------------------------------------------------------
# simple utils
# -------------------------------------------------------------


def _rgb_to_hex(rgb):
    return ('#%02X%02X%02X' % (rgb[0], rgb[1], rgb[2]))

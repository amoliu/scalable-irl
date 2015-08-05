from __future__ import division

import numpy as np
from scipy.spatial import Voronoi
from copy import deepcopy

from matplotlib.patches import Circle, Ellipse, Polygon
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

from .base import ModelMixin
from .state_graph import StateGraph
from ..utils.geometry import edist, trajectory_length
from ..utils.validation import check_array


class HomotopyMDP(ModelMixin):
    """ MDP with the state graph derived directly using homotopy via
    Voronoi tesselation to get the set of states and actions
    """
    def __init__(self, discount, reward, controller, params, world_config):
        assert 0 <= discount < 1, '``discount`` must be in [0, 1)'
        self.gamma = discount
        self._reward = reward
        self._controller = controller
        self._params = params
        self._wconfig = world_config

        # setup the graph structure and internal variables
        self._g = StateGraph()
        self._best_trajs = []
        self._node_id = 0

    def terminal(self, state):
        """ Check if a state is terminal (goal state) """
        position = self._g.gna(state, 'data')
        if edist(position, self._params.goal_state) < 0.05:
            return True
        return False

    def setup(self, entities):
        """ Set up the graph

        Set up the Voronoi graph using all the semantic entities provided,
        these include: persons, start and goal positions, coordinates of all
        objects (including relation lines, annotations) etc.

        Parameters
        -----------
        entities : array-like, shape (2, N)
            Array with the coordinates of all the entities in R^2

        """
        GR = self._params.goal_reward
        CLIMIT = self._params.max_cost
        VMAX = self._params.speed

        entities = check_array(entities)
        self._vor = Voronoi(entities)

        # add nodes to state graph
        # TODO
        # - add checks for terminal and start states (to modify node types)
        for simplex in self._vor.ridge_vertices:
            simplex = np.asarray(simplex)
            x1x2 = self._vor.vertices[simplex, 0]
            y1y2 = self._vor.vertices[simplex, 1]
            source = (x1x2[0], y1y2[0])
            target = (x1x2[1], y1y2[1])

            if np.all(simplex >= 0) and self._in_world(source) and\
                    self._in_world(target):
                # - add start
                s_id = deepcopy(self._node_id)
                s_state = list(source) + [0, VMAX]
                s_type = 'goal' if self._goal(source) else 'simple'
                s_type = 'start' if self._starting(source) else 'simple'
                self.graph.add_node(nid=s_id, data=s_state, cost=CLIMIT,
                                    priority=1, V=GR, pi=0,
                                    Q=[], ntype=s_type)

                self._node_id += 1

                # - add target
                t_id = deepcopy(self._node_id)
                t_state = list(target) + [0, VMAX]
                t_type = 'goal' if self._goal(target) else 'simple'
                t_type = 'start' if self._starting(target) else 'simple'
                self.graph.add_node(nid=t_id, data=t_state, cost=CLIMIT,
                                    priority=1, V=GR, pi=0,
                                    Q=[], ntype=t_type)

                self._node_id += 1

                # - add conecting edges both directions
                f_traj = self._controller.trajectory(s_state, t_state, VMAX)
                f_d = trajectory_length(f_traj)
                f_r, f_phi = self._reward(s_state, f_traj)
                self._g.add_edge(source=s_id, target=t_id, reward=f_r,
                                 duration=f_d, phi=f_phi, traj=f_traj)

                b_traj = self._controller.trajectory(t_state, s_state, VMAX)
                b_d = trajectory_length(b_traj)
                b_r, b_phi = self._reward(t_state, b_traj)
                self._g.add_edge(source=t_id, target=s_id, reward=b_r,
                                 duration=b_d, phi=b_phi, traj=b_traj)

                print(s_type, t_type)

        self._find_best_policies()

    def prep_entities(self, persons, relations, annotations=None, objects=None):
        """ Prepare the entities for building the voronoi """
        entities = []
        for _, p in persons.items():
            entities.append([p[0], p[1]])

        # add starts and goal poses
        # TODO
        # - undo this (starts and goals need not be centroids)
        # - add the nodes manually, and connect to k-nn neighbors
        for start in self._params.start_states:
            entities.append([start[0], start[1]])

        entities.append([self._params.goal_state[0],
                        self._params.goal_state[1]])

        # TODO - add annotations and objects

        return entities

    def visualize(self, persons, relations, annotations=None, fsize=(12, 9)):
        """ Visualize the social navigation world

        Allows recording of demonstrations and also display of final
        graph representing the MDP
        """
        self._setup_visuals(fsize)

        for _, p in persons.items():
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

        for [i, j] in relations:
            x1, y1 = persons[i][0], persons[i][1]
            x2, y2 = persons[j][0], persons[j][1]
            self.ax.plot((x1, x2), (y1, y2), ls='-', c='r', lw=2.0, zorder=2)

        self._plot_graph_in_world()

        # self.ax.plot(self._vor.vertices[:, 0], self._vor.vertices[:, 1], 'o')
        # for simplex in self._vor.ridge_vertices:
        #     simplex = np.asarray(simplex)
        #     if np.all(simplex >= 0):
        #         self.ax.plot(self._vor.vertices[simplex,0],
        #                      self._vor.vertices[simplex,1], 'k-')
        #         # print(self._vor.vertices[simplex, 2])

        return self.ax


    # -------------------------------------------------------------
    # properties
    # -------------------------------------------------------------

    @property
    def graph(self):
        return self._g

    # -------------------------------------------------------------
    # internals
    # -------------------------------------------------------------

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

    def _in_world(self, pose):
        return self._wconfig.x < pose[0] < self._wconfig.w and\
                self._wconfig.y < pose[1] < self._wconfig.h

    def _starting(self, pose):
        for state in self._params.start_states:
            if edist(state, pose) < 0.05:
                return True
        return False

    def _goal(self, pose):
        return edist(self._params.goal_state, pose) < 0.05

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

        # self.figure.canvas.mpl_connect('key_press_event', self._key_press)
        # self.figure.canvas.mpl_connect('button_press_event', self._btn_click)

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
                else:
                    traj = gea(e[0], e[1], 'traj')
                    for wp in traj:
                        v = wp[3]
                        vx, vy = v*np.cos(wp[2]), v*np.sin(wp[2])
                        self.ax.arrow(wp[0], wp[1], 0.5*vx, 0.5*vy, fc='0.7',
                                      ec='0.7', lw=1.0, head_width=0.07,
                                      head_length=0.05, zorder=3)


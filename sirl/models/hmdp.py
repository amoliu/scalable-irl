from __future__ import division

import numpy as np
from scipy.spatial import Voronoi

from matplotlib.patches import Circle, Ellipse, Polygon
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

from .base import ModelMixin
from .state_graph import StateGraph
from ..utils.geometry import edist
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
        entities = check_array(entities)
        self._vor = Voronoi(entities)
        # print(self._vor.vertices)

    def prep_entities(self, persons, relations, annotations=[], objects=[]):
        """ Prepare the entities for building the voronoi """
        entities = []
        for _, p in persons.items():
            entities.append([p[0], p[1]])

        for [i, j] in relations:
            entities.append([persons[i][0], persons[i][1]])
            entities.append([persons[j][0], persons[j][1]])

        # add starts and goal poses
        for start in self._params.start_states:
            entities.append([start[0], start[1]])

        entities.append([self._params.goal_state[0],
                        self._params.goal_state[1]])

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

        # self._plot_graph_in_world()

        self.ax.plot(self._vor.vertices[:, 0], self._vor.vertices[:, 1], 'o')
        for simplex in self._vor.ridge_vertices:
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                self.ax.plot(self._vor.vertices[simplex,0],
                             self._vor.vertices[simplex,1], 'k-')

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

    def _find_policies(self):
        pass

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

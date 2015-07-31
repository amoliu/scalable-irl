from __future__ import division

import numpy as np
from scipy.spatial import Voronoi

from .base import ModelMixin
from .state_graph import StateGraph
from ..utils.geometry import edist


class HomotopyMDP(ModelMixin):
    """ MDP with the state graph derived directly using homotopy via
    Voronoi tesselation to get the set of states and actions
    """
    def __init__(self, discount, reward, controller, params):
        assert 0 <= discount < 1, '``discount`` must be in [0, 1)'
        self.gamma = discount
        self._reward = reward
        self._controller = controller
        self._params = params

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
        assert entities.ndim == 2, 'Expects a 2D array for entities'
        vor = Voronoi(entities)
        print(vor.vertices)

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

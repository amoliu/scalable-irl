from __future__ import division, absolute_import

import six
import numpy as np

from abc import ABCMeta

from ..utils.common import Logger
from ..models.base import ModelMixin


class SparseSampling(six.with_metaclass(ABCMeta, ModelMixin), Logger):

    """ Sparse sampling algorithm for MDP planning """

    def __init__(self, mdp, world, C, H):
        self._mdp = mdp
        self._world = world

    def plan_from_state(self, state):
        """ Estimate Value for a state using Monte carlo samples """
        pass

    def _set_H_and_C(self):
        """ Set H and C for epsilon optimality guarantee """
        pass

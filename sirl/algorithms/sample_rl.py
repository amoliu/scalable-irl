
from __future__ import division


import numpy as np


from ..state_graph import StateGraph


class SBRL(object):
    """Sample based Reinforcement Learning """
    def __init__(self, model):
        self.model = model

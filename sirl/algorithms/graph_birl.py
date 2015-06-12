
from __future__ import division
import numpy as np

from ..base import ModelMixin


class GraphBIRL(ModelMixin):
    """GraphBIRL algorith

    """
    def __init__(self, demos):
        self._demos = demos

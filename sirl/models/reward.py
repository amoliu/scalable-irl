from __future__ import division

from abc import abstractmethod
from abc import ABCMeta


from .base import ModelMixin


class MDPReward(ModelMixin):
    """ Reward  function base class """

    __metaclass__ = ABCMeta
    _template = '_feature_'

    def __init__(self, kind='linfa'):
        self.kind = kind

    @abstractmethod
    def __call__(self, state, action):
        """ Evaluate the reward function for the (state, action) pair

        Compute :math:`r(s, a) = f(s, a, w)` where :math:`f` is a function
        approximator for the reward parameterized by :math:`w`
        """
        raise NotImplementedError('Abstract method')

    @property
    def dim(self):
        """ Dimension of the reward function """
        # - count all class members named '_feature_{x}'
        features = self.__class__.__dict__
        dim = sum([f[0].startswith(self._template) for f in features])
        return dim

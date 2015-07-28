
from __future__ import division

from abc import abstractmethod
from abc import ABCMeta

from .base import ModelMixin


class LocalController(ModelMixin):
    """ GraphMDP local controller """

    __metaclass__ = ABCMeta

    def __init__(self, kind='linear'):
        self.kind = kind

    @abstractmethod
    def __call__(self, state, action, duration, max_speed):
        """ Execute a local controller at ``state`` using ``action``
        for period lasting ``duration`` and speed limit ``max_speed``
        """
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def trajectory(self, start, target, max_speed):
        """ Generate a trajectory by executing the local controller

        Execute the local controller between the given two states to generate
        a local trajectory which encapsulates the meta-action

        """
        raise NotImplementedError('Abstract method')

from __future__ import division

from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.random import choice, randint

from ...models.base import ModelMixin
from ...utils.common import Logger
from ..mdp_solvers import graph_policy_iteration


########################################################################
# Reward Priors
# ######################################################################


class RewardPrior(ModelMixin):
    """ Reward prior interface """
    __meta__ = ABCMeta

    def __init__(self, dim, name):
        self.name = name
        assert 0 < dim, 'Reward dimension must be > 0'
        self.dim = dim

    @abstractmethod
    def __call__(self, r):
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def log_p(self, r):
        raise NotImplementedError('Abstract method')


class UniformRewardPrior(RewardPrior):
    """ Uniform/flat prior"""
    def __init__(self, dim, name='uniform'):
        super(UniformRewardPrior, self).__init__(dim, name)

    def __call__(self, r):
        rp = np.ones(r.shape[0])
        dist = rp / np.sum(rp)
        return dist

    def log_p(self, r):
        return np.log(self.__call__(r))


class GaussianRewardPrior(RewardPrior):
    """Gaussian reward prior"""
    def __init__(self, dim, name='gaussian', sigma=0.5):
        super(GaussianRewardPrior, self).__init__(dim, name)
        self._sigma = sigma

    def __call__(self, r):
        rp = np.exp(-np.square(r)/(2.0*self._sigma**2)) /\
            np.sqrt(2.0*np.pi)*self._sigma
        return rp / np.sum(rp)

    def log_p(self, r):
        # TODO - make analytical
        return np.log(self.__call__(r))


class LaplacianRewardPrior(RewardPrior):
    """Laplacian reward prior"""
    def __init__(self, dim, name='laplace', sigma=0.5):
        super(LaplacianRewardPrior, self).__init__(dim, name)
        self._sigma = sigma

    def __call__(self, r):
        rp = np.exp(-np.fabs(r)/(2.0*self._sigma)) / (2.0*self._sigma)
        return rp / np.sum(rp)

    def log_p(self, r):
        # TODO - make analytical
        return np.log(self.__call__(r))


class DirectionalRewardPrior(RewardPrior):
    """ Prior that injects direction information

    Useful for cases in which we know the direction of influence of a feature
    as either a penalty or a reward. Defaults to all rewarding features

    """
    def __init__(self, dim, name='directional', directions=None):
        super(DirectionalRewardPrior, self).__init__(dim, name)
        self.directions = directions
        if self.directions is None:
            self.directions = [1 for _ in self.dim]

    def __call__(self, r):
        rp = np.array([r[i] * self.directions[i] for i in range(self.dim)])
        # NOTE - unnormalized
        return rp

    def log_p(self, r):
        return np.log(self.__call__(r))


########################################################################
# MCMC proposals
# ######################################################################


class Proposal(ModelMixin):
    """ Proposal for MCMC sampling """
    __meta__ = ABCMeta

    def __init__(self, dim):
        self.dim = dim

    @abstractmethod
    def __call__(self, loc):
        raise NotImplementedError('Abstract class')


class PolicyWalkProposal(Proposal):
    """ PolicyWalk MCMC proposal """
    def __init__(self, dim, delta, bounded=True):
        super(PolicyWalkProposal, self).__init__(dim)
        self.delta = delta
        self.bounded = bounded
        # TODO - allow setting bounds as list of arrays

    def __call__(self, loc):
        new_loc = np.array(loc)
        changed = False
        while not changed:
            d = choice([-self.delta, self.delta])
            i = randint(self.dim)
            if self.bounded:
                if -1 <= new_loc[i]+d <= 1:
                    new_loc[i] += d
                    changed = True
            else:
                new_loc[i] += d
                changed = True
        return new_loc


########################################################################
# BIRL (Bayesian IRL) base interface
########################################################################


class BIRL(ModelMixin, Logger):
    """ Bayesian Inverse Reinforcement Learning

    BIRL algorithm that seeks to find a reward function underlying a set of
    expert demonstrations by computing the posterior of the reward distribution
    :math:`p(r | \Xi)`.

    The algorithms typically summarize the distribution by taking a single
    value such as the mode, or mean.


    Parameters
    ----------
    demos : array-like
        Expert demonstrations as set of M trajectories of state action pairs.
        Trajectories can be of different lengths.
    rep : A representation object
        The underlying representation of the MDP for the task, can be a
        :class:`ControllerGraph`, or any derivative of the representation
        interface :class:`MDPRepresentation`
    prior : :class:``RewardPrior`` or derivative object
        Reward prior callable object
    loss : A callable object, derivative of :class:`RewardLoss`
        Reward loss callable, for evaluating progress in reward search
    beta : float, optional (default=0.9)
        Expert optimality parameter for the reward likelihood term in the
        product of exponential distributions


    Attributes
    ----------
    _demos : array-like
        Expert demonstrations as set of M trajectories of state action pairs.
        Trajectories can be of different lengths.
    _rep : A representation object
        The underlying representation of the MDP for the task, can be a
        :class:`ControllerGraph`, or any derivative of the representation
        interface :class:`MDPRepresentation`
    _prior : :class:``RewardPrior`` or derivative object
        Reward prior callable object
    _loss : A callable object, derivative of :class:`RewardLoss`
        Reward loss callable, for evaluating progress in reward search
    _beta : float, optional (default=0.9)
        Expert optimality parameter for the reward likelihood term in the
        product of exponential distributions

    """
    __meta__ = ABCMeta

    def __init__(self, demos, rep, prior, loss, beta=0.7):
        self._demos = demos
        self._prior = prior
        self._rep = rep
        self._loss = loss

        assert 0.0 < beta <= 1.0, '*beta* must be in (0, 1]'
        self._beta = beta

    @abstractmethod
    def solve(self):
        """ Find the true reward function """
        raise NotImplementedError('Abstract')

    @abstractmethod
    def initialize_reward(self):
        """ Initialize reward function based on solver """
        raise NotImplementedError('Abstract')

    def _compute_policy(self, reward):
        """ Compute the policy induced by a given reward function """
        if self._rep.kind == 'graph':
            self._rep = self._rep.update_rewards(reward)
            graph_policy_iteration(self._rep.graph,
                                   self._rep.mdp.gamma)
            trajs = self._rep.find_best_policies()
            return trajs
        else:
            raise ValueError('Unsupported representation *kind*')

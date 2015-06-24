from __future__ import division

from abc import ABCMeta, abstractmethod
from copy import deepcopy

from numpy.random import choice, randint, uniform
import numpy as np

from ..base import ModelMixin
from .graph_birl import GBIRL


########################################################################
# MCMC proposals

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
            d = choice([-self.delta, 0, self.delta])
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


class GBIRLPolicyWalk(GBIRL):
    """GraphBIRL algorithm using PolicyWalk MCMC

    Bayesian Inverse Reinforcement Learning on Adaptive State-Graphs using
    PolicyWalk (GBIRL-PW)

    Reward posterior disctribution is computed using MCMC samples via a
    grid walk on the space of rewards.

    Parameters
    ----------
    demos : array-like, shape (M x d)
        Expert demonstrations as M trajectories of state action pairs
    mdp : ``GraphMDP`` object
        The underlying (semi) Markov decision problem
    prior : ``RewardPrior`` object
        Reward prior callable
    loss : callable
        Reward loss callable
    step_size : float, optional (default=0.2)
        Grid walk step size
    burn : float, optional (default=0.2)
        Fraction of MCMC samples to throw away before the chain stabilizes
    max_iter : int, optional (default=10)
        Number of iterations of the GBIRL algorith
    beta : float, optional (default=0.9)
        Expert optimality parameter for softmax Boltzman temperature
    reward_max : float, optional (default=1.0)
        Maximum value of the reward signal (for a single dimension)
    mcmc_iter : int, optional (default=200)
        Number of MCMC samples to use in the PolicyWalk algorithm

    Attributes
    -----------
    _delta : float, optional (default=0.2)
        Grid walk step size
    _rmax : float, optional (default=1.0)
        Maximum value of the reward signal (for a single dimension)
    _mcmc_iter : int, optional (default=200)
        Number of MCMC samples to use in the PolicyWalk algorithm
    _burn : float, optional (default=0.2)
        Fraction of MCMC samples to throw away before the chain stabilizes

    Note
    -----
    Using small step sizes generally implies the need for more samples

    """
    def __init__(self, demos, mdp, prior, loss, step_size=0.3, burn=0.2,
                 max_iter=10, beta=0.9, reward_max=1.0, mcmc_iter=200):
        super(GBIRLPolicyWalk, self).__init__(demos, mdp, prior, loss,
                                              beta, max_iter)
        self._delta = step_size
        self._rmax = reward_max
        self._mcmc_iter = mcmc_iter
        self._burn = burn
        # TODO
        # - examine acceptance rates, convergence
        # - log key steps

    def initialize_reward(self):
        """
        Generate initial reward for the algorithm in $R^{|S| / \delta}$
        """
        rdim = self._mdp._reward.dim
        v = np.arange(-self._rmax, self._rmax+self._delta, self._delta)
        reward = np.zeros(rdim)
        for i in range(rdim):
            reward[i] = choice(v)
        return reward

    def find_next_reward(self, reward, g_trajs):
        """ Compute a new reward based on current generated trajectories """
        result = dict()
        result['trace'] = []
        result['walk'] = []
        result['reward'] = None
        result['accept_ratio'] = 0
        result['mh_ratio'] = []

        return self._policy_walk(reward, g_trajs, result)

    # -------------------------------------------------------------
    # internals
    # -------------------------------------------------------------

    def _policy_walk(self, init_reward, g_trajs, result):
        """ Policy Walk MCMC reward posterior computation """
        r = deepcopy(init_reward)
        # r = self.initialize_reward()
        r_mean = deepcopy(r)
        p_dist = PolicyWalkProposal(r.shape[0], self._delta, bounded=False)

        QE = self._expert_trajectory_quality(r)
        QPi = self._generated_trajectory_quality(r, g_trajs)

        for Q_i in QPi:
            self.data['qloss'].append(sum(Qe - Qp for Qe, Qp in zip(QE, Q_i)))

        burn_point = int(self._mcmc_iter * self._burn / 100)
        for step in range(1, self._mcmc_iter+1):
            # - generate new reward sample
            r_new = p_dist(loc=r_mean)

            # - Compute new trajectory quality scores
            QE_new = self._expert_trajectory_quality(r_new)
            QPi_new = self._generated_trajectory_quality(r_new, g_trajs)

            # - compute acceptance probability for the new reward
            mh_ratio = self._mh_ratio(r_mean, r_new, QE, QE_new, QPi, QPi_new)
            # if uniform(0, 1) < min(1, mh_ratio):
            if mh_ratio > 1.0:
                r_mean = self._iterative_reward_mean(r_mean, r_new, step)
                result['accept_ratio'] += 1

            result['mh_ratio'].append(mh_ratio)

            # - handling sample burning
            if step > burn_point:
                result['walk'].append(r_new)
                result['trace'].append(r_mean)
                result['reward'] = r_mean

            # log progress
            if step % 10 == 0:
                print('It: %s, R: %s, R_mean: %s' % (step, r_new, r_mean))
            # self.debug('It: %s, R: %s, R_mean: %s' % (step, r_new, r_mean))

        return result

    def _mh_ratio(self, r, r_new, QE, QE_new, QPi, QPi_new):
        """ Compute the Metropolis-Hastings acceptance ratio

        Given a new reward (weights), MH ratio is used to determine whether or
        not to accept the reward sample.

        Parameters
        -----------
        r : array-like, shape (reward-dim)
            Current reward
        r_new : array-like, shape (reward-dim)
            New reward sample from the MCMC walk
        QE : array-like
            Quality of the expert trajectories based on reward ``r``
        QE_new : array-like
            Quality of the expert trajectories based on reward ``r_new``
        QPi : array-like
            Quality of the generated trajectories based on reward ``r``
        QPi_new : array-like
            Quality of the generated trajectories based on reward ``r_new``

        Returns
        --------
        mh_ratio : float
            The ratio corresponding to,

            .. math::
                ratio = P(R_new|O) / P(R|O) x P(R_new)/P(R)
        """
        # # reward priors
        # prior_new = np.sum(self._prior(r_new))
        # prior = np.sum(self._prior(r))

        # # likelihoods (un-normalized, since we only need the ratio)
        # lk = 1
        # for i, Qe in enumerate(QE):
        #     lk *= np.exp(self._beta * (Qe)) / \
        #           (np.exp(self._beta * (Qe)) +
        #            np.sum(np.exp(self._beta * (Qn[i])) for Qn in QPi))

        # lk_new = 1
        # for i, Qe_new in enumerate(QE_new):
        #     lk_new *= np.exp(self._beta * (Qe_new)) / \
        #               (np.exp(self._beta * (Qe_new)) +
        #                np.sum(np.exp(self._beta * (Qn[i])) for Qn in QPi_new))

        # self.data['lk'].append(lk)
        # self.data['lk_new'].append(lk_new)

        # mh_ratio = (lk_new / lk) * (prior_new / prior)

        # reward priors
        prior_new = np.sum(self._prior.log_p(r_new))
        prior = np.sum(self._prior.log_p(r))

        # log-likelihoods
        lk = 0
        for QP_i in QPi:
            for q_e in QE:
                qs = np.sum([np.exp(self._beta*(q_i - q_e)) for q_i in QP_i])
                lk += -np.log(qs)

        lk_new = 0
        for QP_n in QPi_new:
            for q_n in QE_new:
                qs = np.sum([np.exp(self._beta*(q_j - q_n)) for q_j in QP_n])
                lk_new += -np.log(qs)

        self.data['lk'].append(lk)
        self.data['lk_new'].append(lk_new)

        mh_ratio = (lk_new + prior_new) / (lk + prior)

        return mh_ratio

    def _iterative_reward_mean(self, r_mean, r_new, step):
        """ Iterative mean reward

        Compute the iterative mean of the reward using the running mean
        and a new reward sample
        """
        r_mean = [((step - 1) / float(step)) * m_r + 1.0 / step * r
                  for m_r, r in zip(r_mean, r_new)]
        return np.array(r_mean)

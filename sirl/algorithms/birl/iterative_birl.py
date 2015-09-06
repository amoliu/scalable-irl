"""
Iterative variants of BIRL using just samples from state and action spaces
as trajectories. Do not explicitely use the full spaces. These sampling is
abstracted away by a representation learning step that capture the relevant
parts of these spaces for the task at hand.

"""

from __future__ import division

from copy import deepcopy

from scipy.misc import logsumexp
import scipy as sp

import numpy as np
from numpy.random import uniform

from .base import GeneratingTrajectoryBIRL
# from .base import SamplingTrajectoryBIRL
from .base import PolicyWalkProposal


########################################################################
# Sampling Trajectory type BIRL Algorithms
#########################################################################


# class LPSampledBIRL(SampledBIRL):
#     """ LP based SampledBIRL """
#     def __init__(self, arg):
#         super(LPSampledBIRL, self).__init__()
#         self.arg = arg


# class MAPSampledBIRL(SampledBIRL):
#     """ LP based SampledBIRL """
#     def __init__(self, arg):
#         super(MAPSampledBIRL, self).__init__()
#         self.arg = arg


########################################################################
# Generating Trajectory type BIRL Algorithms
#########################################################################


class GTBIRLOptim(GeneratingTrajectoryBIRL):
    """ Generating Trajectory BIRL algorithm using direct optimization


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
    max_iter : int, optional (default=10)
        Number of iterations of the GenerativeBIRL algorithm
    beta : float, optional (default=0.9)
        Expert optimality parameter for softmax Boltzman temperature
    reward_max : float, optional (default=1.0)
        Maximum value of the reward signal (for a single dimension)


    Attributes
    -----------
    _rmax : float, optional (default=1.0)
        Maximum value of the reward signal (for a single dimension)
    _bounds : tuple, optional (default=None)
        Box bounds for L-BFGS optimization of the negative log-likelihood,
        specified for each dimension of the reward function vector, e.g.
        ((-1, 1), (-1, 0)) for a 2D reward vector

    """
    def __init__(self, demos, rep, prior, loss, max_iter=10, beta=0.9,
                 reward_max=1.0, bounds=None):
        super(GTBIRLOptim, self).__init__(demos, rep, prior, loss,
                                          beta, max_iter)
        self._rmax = reward_max
        self._bounds = bounds
        if self._bounds is None:
            self._bounds = tuple((-self._rmax, self._rmax)
                                 for _ in range(self._rep.mdp.reward.dim))

    def initialize_reward(self, delta=0.2):
        """
        Generate initial reward
        """
        rdim = self._rep.mdp.reward.dim
        reward = np.array([np.random.uniform(-self._rmax, self._rmax)
                           for _ in range(rdim)])
        return reward

    def find_next_reward(self):
        """ Compute a new reward based on current generated trajectories """
        # initialize the reward TODO - why???
        r_init = self.initialize_reward()

        # run optimization to minimize N_llk
        res = sp.optimize.fmin_l_bfgs_b(self._neg_loglk,
                                        r_init,
                                        approx_grad=1,
                                        bounds=self._bounds)

        self.debug('Solver result: {}'.format(res))
        reward = res[0]

        return reward

    # -------------------------------------------------------------
    # internals
    # -------------------------------------------------------------

    def _neg_loglk(self, r):
        """ Compute the negative log likelihood with respect to the given
        reward and generated trajectories.
        """
        # - prepare the trajectory quality scores
        QE = self._rep.trajectory_quality(r, self._demos)
        QPi = [self._rep.trajectory_quality(r, self._g_trajs[i])
               for i in range(self._iteration)]

        # - the negative log likelihood
        z = []
        for q_e in QE:
            for QP_i in QPi:
                for q_i in QP_i:
                    z.append(self._beta*(q_i - q_e))
        lk = logsumexp(z)

        return lk


########################################################################
# PolicyWalk


class GTBIRLPolicyWalk(GeneratingTrajectoryBIRL):
    """ Generating Trajectory BIRL algorithm using PolicyWalk MCMC

    Reward posterior disctribution is computed using MCMC samples via a
    grid walk on the space of rewards.


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
    loss : callable
        Reward loss callable
    step_size : float, optional (default=0.2)
        Grid walk step size
    burn : float, optional (default=0.2)
        Fraction of MCMC samples to throw away before the chain stabilizes
    max_iter : int, optional (default=10)
        Number of iterations of the GenerativeBIRL algorithm
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
    def __init__(self, demos, rep, prior, loss, step_size=0.3, burn=0.2,
                 max_iter=10, beta=0.9, reward_max=1.0, mcmc_iter=200,
                 cooling=False):
        super(GTBIRLPolicyWalk, self).__init__(demos, rep, prior,
                                               loss, beta, max_iter)
        self._delta = step_size
        self._rmax = reward_max
        self._mcmc_iter = mcmc_iter
        self._burn = burn
        self._tempered = cooling

        # some data for diagnosis
        self.data['trace'] = []
        self.data['walk'] = []
        self.data['accept_ratios'] = []
        self.data['iter_rewards'] = []

    def initialize_reward(self):
        """
        Generate initial reward for the algorithm in $R^{|S| / \delta}$
        """
        rdim = self._rep.mdp.reward.dim
        reward = np.array([np.random.uniform(-self._rmax, self._rmax)
                           for _ in range(rdim)])
        if self._tempered:
            # initialize to the maximum of prior
            prior = self._prior(reward)
            reward = np.array([max(prior) for _ in range(rdim)])

        return reward

    def find_next_reward(self):
        """ Compute a new reward based on current generated trajectories """
        return self._policy_walk()

    # -------------------------------------------------------------
    # internals
    # -------------------------------------------------------------

    def _policy_walk(self):
        """ Policy Walk MCMC reward posterior computation """
        r = self.initialize_reward()
        r_mean = deepcopy(r)
        p_dist = PolicyWalkProposal(r.shape[0], self._delta, bounded=True)

        QE = self._rep.trajectory_quality(r, self._demos)
        QPi = [self._rep.trajectory_quality(r, self._g_trajs[i])
               for i in range(self._iteration)]

        burn_point = int(self._mcmc_iter * self._burn / 100)

        for step in range(1, self._mcmc_iter+1):
            r_new = p_dist(loc=r_mean)
            QE_new = self._rep.trajectory_quality(r_new, self._demos)
            QPi_new = [self._rep.trajectory_quality(r_new, self._g_trajs[i])
                       for i in range(self._iteration)]

            mh_ratio = self._mh_ratio(r_mean, r_new, QE, QE_new, QPi, QPi_new)
            accept_probability = min(1, mh_ratio)
            if self._tempered:
                accept_probability = min(1, mh_ratio) ** self._cooling(step)

            if accept_probability > uniform(0, 1):
                r_mean = self._iterative_reward_mean(r_mean, r_new, step)
                self.data['accept_ratios'].append(1)

            # - handling sample burning
            if step > burn_point:
                self.data['trace'].append(r_mean)
                self.data['walk'].append(r_new)

            if step % 10 == 0:
                print('It: %s, R: %s, R_mean: %s' % (step, r_new, r_mean))
            # self.debug('It: %s, R: %s, R_mean: %s' % (step, r_new, r_mean))

        self.data['iter_rewards'].append(r_mean)
        return r_mean

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
            The ratio corresponding to :math:`P(r_n|O) / P(r|O) x P(r_n)/P(r)`

        """
        # - initialize reward posterior distribution to log priors
        p_new = np.sum(self._prior.log_p(r_new))
        p = np.sum(self._prior.log_p(r))

        # - log-likelihoods
        z = []
        for q_e in QE:
            for QP_i in QPi:
                for q_i in QP_i:
                    z.append(self._beta*(q_i - q_e))
        lk = -logsumexp(z)

        z_new = []
        for q_e_new in QE_new:
            for QP_i_new in QPi_new:
                for q_i_new in QP_i_new:
                    z_new.append(self._beta*(q_i_new - q_e_new))
        lk_new = -logsumexp(z_new)

        mh_ratio = (lk_new + p_new) / (lk + p)
        return mh_ratio

    def _iterative_reward_mean(self, r_mean, r_new, step):
        """ Iterative mean reward

        Compute the iterative mean of the reward using the running mean
        and a new reward sample

        """
        r_mean = [((step - 1) / float(step)) * m_r + 1.0 / step * r
                  for m_r, r in zip(r_mean, r_new)]
        return np.array(r_mean)

    def _cooling(self, step):
        """ Tempering """
        return 5 + step / 50.0

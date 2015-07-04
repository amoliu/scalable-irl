"""
Optimization based (approximate) inference for GBIRL
"""

from __future__ import division

from copy import deepcopy
from random import randrange

from scipy.misc import logsumexp
import numpy as np
import scipy as sp

from .graph_birl import GBIRL


class GradientGBIRL(GBIRL):
    """GraphBIRL algorithm using gradient minimization of the likelihood

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
    max_iter : int, optional (default=10)
        Number of iterations of the GBIRL algorith
    beta : float, optional (default=0.9)
        Expert optimality parameter for softmax Boltzman temperature
    reward_max : float, optional (default=1.0)
        Maximum value of the reward signal (for a single dimension)
    grad_iter : int, optional (default=200)
        Maximum number of iterations for gradient optimization

    Attributes
    -----------
    _rmax : float, optional (default=1.0)
        Maximum value of the reward signal (for a single dimension)
    _grad_iter : int, optional (default=200)
        Number of MCMC samples to use in the PolicyWalk algorithm

    """
    def __init__(self, demos, mdp, prior, loss, max_iter=10, beta=0.9,
                 reward_max=1.0, grad_iter=200):
        super(GradientGBIRL, self).__init__(demos, mdp, prior, loss,
                                            beta, max_iter)
        self._rmax = reward_max
        self._grad_iter = grad_iter

        self.data = dict()
        self.data['qloss'] = []

    def initialize_reward(self, delta=0.2):
        """
        Generate initial reward
        """
        rdim = self._mdp._reward.dim
        loc = [-self._rmax + i * delta
               for i in range(int(self._rmax / delta + 1))]
        r = [loc[randrange(int(self._rmax / delta + 1))] for _ in range(rdim)]
        reward = np.array(r)
        return reward

    def find_next_reward(self, g_trajs):
        """ Compute a new reward based on current generated trajectories """
        # initialize the reward
        r_init = self.initialize_reward()

        # hack - put the g_trajs a member to avoid passing it to the objective
        self.g_trajs = g_trajs

        # run optimization to minimize N_llk
        objective = self._neg_loglk
        # res = sp.optimize.fmin_bfgs(objective, r_init)
        res = sp.optimize.fmin_l_bfgs_b(objective, r_init, approx_grad=1,
                                        bounds=((-1, 1), (-1, 1), (-1, 1)))

        print(res)
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
        QE = self._expert_trajectory_quality(r)
        QPi = self._generated_trajectory_quality(r, self.g_trajs)
        ql = sum([sum(Qe - Qp for Qe, Qp in zip(QE, Q_i)) for Q_i in QPi])
        self.data['qloss'].append(ql)

        # - the N_lk
        z = []
        for q_e in QE:
            for QP_i in QPi:
                for q_i in QP_i:
                    z.append(self._beta*(q_i - q_e))
        lk = logsumexp(z)

        return lk

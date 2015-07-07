"""
Optimization based (approximate) inference for TBIRL
"""

from __future__ import division

from random import randrange

from scipy.misc import logsumexp
import numpy as np
import scipy as sp

from .graph_birl import TBIRL


class TBIRLOpt(TBIRL):
    """TBIRL algorithm using direct optimization on the likelihood

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
        Number of iterations of the TBIRL algorithm
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
    def __init__(self, demos, mdp, prior, loss, max_iter=10, beta=0.9,
                 reward_max=1.0, bounds=None):
        super(TBIRLOpt, self).__init__(demos, mdp, prior, loss,
                                       beta, max_iter)
        self._rmax = reward_max
        self._bounds = bounds
        if self._bounds is None:
            self._bounds = tuple((-self._rmax, self._rmax)
                                 for _ in range(self._mdp._reward.dim))

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
        res = sp.optimize.fmin_l_bfgs_b(objective, r_init, approx_grad=1,
                                        bounds=self._bounds)

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
        self.data['qloss'].append(self._loss(QE,  QPi))

        # - the N_lk
        z = []
        for q_e in QE:
            for QP_i in QPi:
                for q_i in QP_i:
                    z.append(self._beta*(q_i - q_e))
        lk = logsumexp(z)

        return lk

    def get_diff_feature_matrix(self, start_state):
        nb_g_trajs = sum(1 for i in self.g_trajs)
        time = 0
        rdim = self._mdp._reward.dim
        G = self._mdp.graph

        QEf = np.zeros(rdim + 1)
        for n in self._demos[start_state]:
            actions = G.out_edges(n)
            if actions:
                e = actions[G.gna(n, 'pi')]
                tmp = list(G.gea(e[0], e[1], 'phi'))
                tmp.append(0)
                tmp = np.array(tmp)
                time += G.gea(e[0], e[1], 'duration')
                tmp = (self._mdp.gamma ** time) * tmp
                QEf += tmp
                time += G.gea(e[0], e[1], 'duration')
            else:
                tmp = ([0] * rdim)
                tmp.append(1)
                tmp = np.array(tmp)
                tmp = (self._mdp.gamma ** time) * tmp
                QEf += tmp

        Qf = np.zeros((rdim + 1) * nb_g_trajs).reshape(rdim + 1, nb_g_trajs)
        for i, generated_traj in enumerate(self.g_trajs):
            QPif = np.zeros(rdim + 1)
            time = 0
            for n in generated_traj[start_state]:
                if n.get_edges() != []:
                    e = actions[G.gna(n, 'pi')]
                    tmp = list(G.gea(e[0], e[1], 'phi'))
                    tmp.append(0)
                    tmp = np.array(tmp)
                    tmp = (self._mdp.gamma ** time) * tmp
                    QPif += tmp
                    time += G.gea(e[0], e[1], 'duration')
                else:
                    tmp = ([0] * rdim)
                    tmp.append(1)
                    tmp = np.array(tmp)
                    tmp = (self._mdp.gamma ** time) * tmp
                    QPif += tmp
            Qf[:, i] = np.transpose(QPif - QEf)
        return Qf

    def _ais(self, start_state, r):
        goal_reward = self._mdp._params.goal_reward
        G = self._mdp.graph
        time = 0
        QE = 0
        for n in self._demos[start_state]:
            actions = G.out_edges(n)
            if actions:
                e = actions[G.gna(n, 'pi')]
                r = np.dot(r, G.gea(e[0], e[1], 'phi'))
                QE += (self._mdp.gamma ** time) * r
                time += G.gea(e[0], e[1], 'duration')
            else:
                QE += (self._mdp.gamma ** time) * goal_reward

        QPis = []
        for generated_traj in self.g_trajs:
            QPi = 0
            time = 0
            for n in generated_traj[start_state]:
                actions = G.out_edges(n)
                if actions:
                    e = actions[G.gna(n, 'pi')]
                    r = np.dot(r, G.gea(e[0], e[1], 'phi'))
                    QE += (self._mdp.gamma ** time) * r
                    time += G.gea(e[0], e[1], 'duration')
                else:
                    QPi += (self._mdp.gamma ** time) * goal_reward
            QPis.append(QPi)
        QPis = [np.exp(Q - QE) for Q in QPis]
        tot = sum(Q for Q in QPis)
        QPis = [Q / float(1 + tot) for Q in QPis]
        QPis = np.array(QPis)
        return QPis

    def _grad_nloglk(self, r):
        """ Gradient of the negative log likelihood
        """
        num_starts = sum(1 for i in self._demos)
        grad = sum(np.mat(self.get_diff_feature_matrix(i)) *
                   np.transpose(np.mat(self._ais(i, r)))
                   for i in range(num_starts))
        return grad

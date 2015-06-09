
from __future__ import division

import numpy as np

from numpy.random import choice
from scipy.stats import norm


def wchoice(elements, weights):
    """ Choose a single element with probability proportional to its weight """
    # Hack - shift and re-scale to avoid issues with negative V(s)
    w2 = np.array(weights) + 1000
    w2 = w2 / np.sum(w2)
    return choice(elements, 1, p=w2)[0]


def eval_gaussian(x, mu=0.0, sigma=0.2):
    """
    Evaluate a Gaussian at a point
    """
    return norm.pdf(x, loc=mu, scale=sigma)

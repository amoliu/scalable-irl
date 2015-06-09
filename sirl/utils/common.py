
from numpy.random import choice
from scipy.stats import norm


def wchoice(elements, weights):
    """ Choose a single element with probability proportional to its weight """
    return choice(elements, 1, p=weights)[0]


def eval_gaussian(x, mu=0.0, sigma=0.2):
    """
    Evaluate a Gaussian at a point
    """
    return norm.pdf(x, loc=mu, scale=sigma)

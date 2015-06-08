
from numpy.random import choice


def wchoice(elements, weights):
    """ Choose a single element with probability proportional to its weight """
    return choice(elements, 1, p=weights)[0]

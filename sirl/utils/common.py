
from __future__ import division

import time
import os.path
import logging
import traceback

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


def map_range(value, mina, maxa, mint, maxt):
    denom = maxa - mina
    if abs(denom) < 1e-09:
        denom = 1e-09
    return mint + ((value - mina) * (maxt - mint) / denom)


def softmax(x1, x2):
    """ Compute the `Softmax(x1, x2) wrt to elements of x1, and x2"""
    mx = max(x1, x2)
    mn = min(x1, x2)
    return mx + np.log(1 + np.exp(mx-mn))


##########################################################################

class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start


##########################################################################


class Logger(object):
    """
    Logger mixin class adding verbose logging to subclasses.
    """

    show_source_location = False

    def _raw_log(self, logfn, message, exc_info):
        cname = ''
        loc = ''
        fn = ''
        tb = traceback.extract_stack()
        if len(tb) > 2:
            if self.show_source_location:
                loc = '[%s:%d]:' % (os.path.basename(tb[-3][0]), tb[-3][1])
            fn = tb[-3][2]
            if fn != '<module>':
                if self.__class__.__name__ != Logger.__name__:
                    fn = self.__class__.__name__ + '.' + fn
                fn += ' >>>'

        logfn(loc + cname + fn + ' ' + message, exc_info=exc_info)

    def info(self, message, exc_info=False):
        """
        Log a info-level message. If exc_info is True, if an exception
        was caught, show the exception information (message and stack trace).
        """
        self._raw_log(logging.info, message, exc_info)

    def debug(self, message, exc_info=False):
        """
        Log a debug-level message. If exc_info is True, if an exception
        was caught, show the exception information (message and stack trace).
        """
        self._raw_log(logging.debug, message, exc_info)

    def warning(self, message, exc_info=False):
        """
        Log a warning-level message. If exc_info is True, if an exception
        was caught, show the exception information (message and stack trace).
        """
        self._raw_log(logging.warning, message, exc_info)

    def error(self, message, exc_info=False):
        """
        Log an error-level message. If exc_info is True, if an exception
        was caught, show the exception information (message and stack trace).
        """
        self._raw_log(logging.error, message, exc_info)

    def log_config(self, level=logging.DEBUG):
        """
        Apply a basic logging configuration which outputs the log to the
        console (stderr). Optionally, the minimum log level can be set, one
        of DEBUG, WARNING, ERROR (or any of the levels from the logging
        module). If not set, DEBUG log level is used as minimum.
        """
        logging.basicConfig(level=level,
                            format='%(asctime)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')


# Many pieces shamelessly borrowed from scikit-learn
# Licence: BSD

from abc import abstractmethod, abstractproperty
from abc import ABCMeta

import inspect
import warnings

import numpy as np
import six


###############################################################################


class ModelMixin(object):
    """ Base mixin class for models and algorithms

    All RL/IRL algorithms should specify their required parameters in their
    respective  ``__init__`` methods with explicit keyword arguiments only

    """
    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        args, varargs, kw, default = inspect.getargspec(init)
        if varargs is not None:
            raise RuntimeError("skirl algorithms should always "
                               "specify their parameters in the signature"
                               " of their __init__ (no varargs)."
                               " %s doesn't follow this convention."
                               % (cls, ))
        # Remove 'self'
        # XXX: This is going to fail if the init is a staticmethod, but
        # who would do this?
        args.pop(0)
        args.sort()
        return args

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The former have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in six.iteritems(params):
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError('Invalid parameter %s ' 'for estimator %s'
                                     % (key, self.__class__.__name__))
                setattr(self, key, value)
        return self

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(deep=False),
                                               offset=len(class_name),),)


###############################################################################


def _pprint(params, offset=0, printer=repr):
    """Pretty print the dictionary 'params'

    Parameters
    ----------
    params: dict
        The dictionary to pretty print

    offset: int
        The offset in characters to add at the begin of each line.

    printer:
        The function to convert entries to strings, typically
        the builtin str or repr

    """
    # Do a multi-line justified repr:
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset // 2) * ' '
    for i, (k, v) in enumerate(sorted(six.iteritems(params))):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if (this_line_length + len(this_repr) >= 75 or '\n' in this_repr):
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines


########################################################################

# Reward model

class MDPReward(ModelMixin):
    """ Reward  function base class """

    __metaclass__ = ABCMeta
    _template = '_feature_'

    def __init__(self, world, kind='linfa'):
        # keep a reference to parent MDP to get access to S, A
        self._world = world
        self.kind = kind

    @abstractmethod
    def __call__(self, state, action):
        """ Evaluate the reward function for the (state, action) pair

        Compute :math:`r(s, a) = f(s, a, w)` where :math:`f` is a function
        approximator for the reward parameterized by :math:`w`
        """
        raise NotImplementedError('Abstract method')

    @property
    def dim(self):
        """ Dimension of the reward function """
        # - count all class members named '_feature_{x}'
        features = self.__class__.__dict__
        dim = sum([f[0].startswith(self._template) for f in features])
        return dim


# Reward Loss Functions

class RewardLoss(ModelMixin):
    """Reward loss function """

    __meta__ = ABCMeta

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __call__(self, r1, r2):
        """ Reward loss between ``r1`` and ``r2`` """
        raise NotImplementedError('Abstract')


class TrajQualityLoss(RewardLoss):
    """ Trajectory quality loss :math:`||Q(s) - Q(s)||_p` """
    def __init__(self, p=2, name='tqloss'):
        super(TrajQualityLoss, self).__init__(name)
        self.p = p

    def __call__(self, QE, QPi):
        ql = sum([sum((Qe - Qp)**self.p
                 for Qe, Qp in zip(QE, Q_i))
                 for Q_i in QPi])
        return ql

########################################################################


class LocalController(ModelMixin):
    """ GraphMDP local controller """

    __metaclass__ = ABCMeta

    def __init__(self, world, kind='linear'):
        self._world = world
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


########################################################################


class MDP(ModelMixin):
    """ Markov Decision Process Model

    Parameters
    ------------
    discount : float
        MDP discount factor
    reward : `SocialNavReward` object
        Reward function for social navigation task

    Attributes
    -----------
    gamma : float
        MDP discount factor
    _reward : :class:`SocialNavReward` object
        Reward function for social navigation task

    """

    def __init__(self, discount, reward):
        if 0.0 > discount >= 1.0:
            raise ValueError('The `discount` must be in [0, 1)')

        self.gamma = discount
        self.reward = reward

    @abstractmethod
    def terminal(self, state):
        """ Check if a state is terminal (goal state) """
        raise NotImplementedError('Abstract method')

    @abstractproperty
    def state_dimension(self):
        return 0

    @abstractproperty
    def start_states(self):
        return None

    @abstractproperty
    def goal_state(self):
        return None


########################################################################

class World(object):
    """ The environment that the MDP is defined on

    This is largely a data container for all the things in the environment
    that the MDP should care about, for use in computing reward functions
    etc

    Also contains limits of the environment

    """
    def __init__(self):
        pass

    @abstractmethod
    def in_world(self, state):
        raise NotImplementedError('Abstract')

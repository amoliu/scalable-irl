
from nose.tools import assert_equal

from sirl.models.base import LocalController
from sirl.models.base import MDPReward
from sirl.models.mdp import GraphMDP
# from sirl.models import AlgoParams


def test_local_controller():
    try:
        LocalController()
        assert False, 'Abstract method instantiated'
    except TypeError:
        pass

    class ConcreteLC(LocalController):
        def __init__(self, kind):
            super(ConcreteLC, self).__init__(kind)

        def __call__(self, state, action, duration, max_speed):
            return 42

        def trajectory(self, start, target, max_speed):
            return None

    subclass = ConcreteLC(kind='some-name')
    assert_equal(subclass.kind, 'some-name')
    assert_equal(subclass(1, 2, 3, 4), 42)


def test_mdp_reward():
    try:
        MDPReward()
        assert False, 'Abstract method instantiated'
    except TypeError:
        pass

    class ConcreteReward(MDPReward):
        def __init__(self, kind):
            super(ConcreteReward, self).__init__(kind)

        def __call__(self, state_a, state_b):
            return 42

        @property
        def dim(self):
            return 21

    subclass = ConcreteReward(kind='some-name')
    assert_equal(subclass.kind, 'some-name')
    assert_equal(subclass(1, 2), 42)

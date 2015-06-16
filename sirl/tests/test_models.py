
from nose.tools import assert_raises
from nose.tools import assert_equal

from sirl.models import LocalController
from sirl.models import MDPReward
from sirl.models import GraphMDP
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

        def __call__(self, state, action, duration):
            return 42

    subclass = ConcreteLC(kind='some-name')
    assert_equal(subclass.kind, 'some-name')
    assert_equal(subclass(1, 2, 3), 42)


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

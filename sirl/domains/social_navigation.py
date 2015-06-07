
from __future__ import division


# from ..models import MDPState
from ..models import LocalController
from ..models import MDPReward
from ..models import GraphMDP


# Maybe get rid of this because of graph already has the data
# class SocialNavState(MDPState):
#     """ Social Navigation MDP state """
#     def __init__(self, arg):
#         super(SocialNavState, self).__init__()
#         self.arg = arg


class SocialNavLocalController(LocalController):
    """ Social navigation local controller """
    def __init__(self, arg):
        super(SocialNavLocalController, self).__init__()
        self.arg = arg


class SocialNavReward(MDPReward):
    """ Social Navigation Reward Funtion """
    def __init__(self, arg):
        super(SocialNavReward, self).__init__()
        self.arg = arg


class SocialNavMDP(GraphMDP):
    """docstring for SocialNavMDP"""
    def __init__(self, arg):
        super(SocialNavMDP, self).__init__()
        self.arg = arg

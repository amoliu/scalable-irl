
from __future__ import division

import json
import time
import copy
import numpy as np

import matplotlib
matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")

# np.random.seed(42)

from sirl.domains.navigation.social_navigation import SocialNavMDP
from sirl.domains.navigation.local_controllers import POSQLocalController
from sirl.domains.navigation.local_controllers import LinearLocalController
from sirl.domains.navigation.reward_functions import SimpleReward
from sirl.domains.navigation.social_navigation import SocialNavEnvironment

from sirl.algorithms.controller_graph import ControllerGraph
from sirl.models.parameters import ControllerGraphParams

DPATH = '../../experiments/social_rewards/'

params = ControllerGraphParams()
params.load(DPATH+'graph_mdp_params.json')
params.max_cost = 1000
params.max_samples = 80
params.radius = 1.8
params.speed = 1
params.max_edges = 360
params.init_type = 'random'

STARTS = ((0.5, 0.5), (4, 0.1), (2, 3), (8.5, 5.2),
          (8.9, 0.1), (0.1, 8.5), (4, 3))
GOAL = (5.5, 9)
BEHAVIOR = 'polite'
WEIGHTS = {
    # 'polite': [-0.75996794, -0.06025078, -0.99495333],
    'polite': [-1.0, -0.7, -0.85],
    'sociable': [-1.0, +0.2, -0.95]
}

# load world elements
f = open(DPATH+'scenes/metropolis.json', 'r')
scene = json.load(f)
f.close()
persons = scene['persons']
persons = {int(k): v for k, v in persons.items()}
relations = scene['relations']

world = SocialNavEnvironment((0, 0, 10, 10), persons, relations, GOAL, STARTS)

posq_controller = POSQLocalController(world, base=0.4, resolution=0.15)
lin_controller = LinearLocalController(world, resolution=0.1)


def show_graph_reinforcement_learning():
    sreward = SimpleReward(world, WEIGHTS[BEHAVIOR], scaled=False,
                           behavior=BEHAVIOR, anisotropic=False,
                           thresh_p=0.45, thresh_r=0.2)

    mdp = SocialNavMDP(discount=0.95, reward=sreward, world=world)

    cg = ControllerGraph(mdp=mdp,
                         local_controller=lin_controller,
                         params=params)

    # trajs = np.load('demos_metropolis.npy')
    cg.initialize_state_graph(samples=[(5, 5), (1, 3)])
    cg = cg.run()

    mdp.visualize(cg.graph, cg.policies, show_edges=False)

    plt.show()


if __name__ == '__main__':
    show_graph_reinforcement_learning()
    # learn_reward()

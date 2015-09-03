
from __future__ import division

import json
import time
import numpy as np

import matplotlib
matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")

np.random.seed(42)

from sirl.domains.navigation.social_navigation import SocialNavMDP
from sirl.domains.navigation.local_controllers import POSQLocalController

from sirl.domains.navigation.reward_functions import SimpleReward
from sirl.domains.navigation.social_navigation import WorldConfig

from sirl.algorithms.controller_graph import ControllerGraph

from sirl.models.parameters import GraphMDPParams

DPATH = '../../experiments/social_rewards/'

params = GraphMDPParams()
params.load(DPATH+'graph_mdp_params.json')
params.max_cost = 1000
params.max_samples = 240
params.radius = 2.4
params.speed = 1
params.max_edges = 36
# choices: random, homotopy, trajectory
params.init_type = 'random'

params.start_states = [[0.5, 0.5], [4, 0.1], [2, 3], [8.5, 5.2],
                       [8.9, 0.1], [0.1, 8.5], [4, 3]]
params.goal_state = (5.5, 9)

# weights = [-1.0, -0.6, -0.95]  # polite
weights = [-1.0, -0.6, -0.95]  # polite

f = open(DPATH+'scenes/metropolis.json', 'r')
scene = json.load(f)
f.close()

persons = scene['persons']
persons = {int(k): v for k, v in persons.items()}
relations = scene['relations']

# wconfig = WorldConfig(0, 0, 15, 15)
wconfig = WorldConfig(0, 0, 10, 10)

gs = params.goal_state
sreward = SimpleReward(persons, relations, None, gs, weights,
                       discount=1, hzone=0.24, scaled=False)

posq_controller = POSQLocalController(wconfig, gs, base=0.4, resolution=0.15)


def show_graph_reinforcement_learning():
    mdp = SocialNavMDP(discount=0.95, reward=sreward,
                       params=params, world_config=wconfig,
                       persons=persons, relations=relations)

    cg = ControllerGraph(mdp=mdp,
                         local_controller=posq_controller,
                         params=params)

    # trajs = np.load('demos_metropolis.npy')
    cg.initialize_state_graph(samples=[(5, 5), (1, 3)])

    g, policies = cg.run()

    mdp.visualize(g, policies)

    plt.show()


if __name__ == '__main__':
    show_graph_reinforcement_learning()

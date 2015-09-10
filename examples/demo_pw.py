from __future__ import division

import matplotlib
matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")

import numpy as np
# np.random.seed(21)

from sirl.domains.puddle_world.puddle_world import PuddleWorldMDP
from sirl.domains.puddle_world.puddle_world import PuddleWorldEnvironment
from sirl.domains.puddle_world.puddle_world import PuddleWorldControler
from sirl.domains.puddle_world.puddle_world import PuddleReward
from sirl.domains.puddle_world.puddle_world import PuddleRewardOriented

from sirl.algorithms.controller_graph import ControllerGraph
from sirl.models.parameters import ControllerGraphParams


world = PuddleWorldEnvironment(start=[(0.3, 0.65)], goal=(0.97, 0.97))
lin_controller = PuddleWorldControler(world)


def demo_rep_learning():
    # reward = PuddleReward(world)
    reward = PuddleRewardOriented(world, weights=(1.0, -0.2, -0.001))

    mdp = PuddleWorldMDP(discount=0.95, reward=reward, world=world)

    params = ControllerGraphParams()
    params.load('pw_cg_params.json')
    print(params)

    cg = ControllerGraph(mdp=mdp,
                         local_controller=lin_controller,
                         params=params)

    cg.initialize_state_graph(samples=[(0.5, 0.2)])
    cg = cg.run()

    mdp.visualize(cg.graph, cg.policies, show_edges=True)
    plt.show()

if __name__ == '__main__':
    demo_rep_learning()

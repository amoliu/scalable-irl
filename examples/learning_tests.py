
from __future__ import division

import json
import copy

import numpy as np
import matplotlib.pyplot as plt

from sirl.domains.navigation.social_navigation import SocialNavMDP
from sirl.domains.navigation.local_controllers import POSQLocalController
from sirl.domains.navigation.local_controllers import LinearLocalController
from sirl.domains.navigation.reward_functions import SimpleBehaviors
from sirl.domains.navigation.social_navigation import SocialNavEnvironment

from sirl.algorithms.controller_graph import ControllerGraph
from sirl.models.parameters import ControllerGraphParams

from sirl.algorithms.birl.iterative_birl import GTBIRLOptim
from sirl.algorithms.birl import DirectionalRewardPrior
from sirl.models.base import TrajQualityLoss


STARTS = ((0.5, 0.5), (4, 0.1), (2, 3), (8.5, 5.2),
          (8.9, 0.1), (0.1, 8.5), (4, 3))
GOAL = (5.5, 9)
BEHAVIOR = 'polite'
WEIGHTS = {
    # [-0.4125548 ,  0.04691908,  0.18735264]
    # 'polite': [-0.75996794, -0.06025078, -0.99495333],
    'polite': [-1.0, -0.7, -0.85],
    'sociable': [-1.0, +0.2, -0.95]
}

# load world elements
DPATH = '../../experiments/social_rewards/'
f = open(DPATH+'scenes/metropolis.json', 'r')
scene = json.load(f)
f.close()
persons = scene['persons']
persons = {int(k): v for k, v in persons.items()}
relations = scene['relations']

world = SocialNavEnvironment(0, 0, 10, 10, persons, relations, GOAL, STARTS)

posq_controller = POSQLocalController(world, base=0.4, resolution=0.15)
lin_controller = LinearLocalController(world, resolution=0.1)


def learn_reward():
    sreward = SimpleBehaviors(world, WEIGHTS[BEHAVIOR], scaled=False,
                              behavior=BEHAVIOR, anisotropic=False,
                              thresh_p=0.45, thresh_r=0.2)

    mdp = SocialNavMDP(discount=0.95, reward=sreward, world=world)

    params = ControllerGraphParams()
    params.load('social_nav_cg_params.json')
    print(params)

    cg = ControllerGraph(mdp=mdp,
                         local_controller=lin_controller,
                         params=params)
    # mdp.visualize(cg, cg.policies, show_edges=True, recording=True)
    # plt.show()

    trajs = np.load('demos_metropolis.npy')
    cg.initialize_state_graph(samples=trajs)
    # cg.initialize_state_graph(samples=[(5, 5), (1, 3)])
    cg = cg.run()
    mdp.visualize(cg.graph, cg.policies, show_edges=True)
    # plt.savefig('state_graph_traj.pdf')
    plt.show()

    demos = copy.deepcopy(cg.policies)
    loss = TrajQualityLoss(p=2)
    # prior = UniformRewardPrior()
    prior = DirectionalRewardPrior(dim=sreward.dim, directions=[-1, -1, -1])

    bounds = ((-1, 0), (-1, 0), (-1, 0))
    irl_algo = GTBIRLOptim(demos, cg, prior, loss=loss,
                           beta=0.9, max_iter=20, bounds=bounds)

    # irl_algo = GTBIRLPolicyWalk(demos, cg, prior, loss=loss,
    #                             beta=0.9, max_iter=30)
    r = irl_algo.solve()
    print('Learned reward, {}'.format(r))

    # np.save('qloss', irl_algo.data['qloss'])
    # np.save('QE', irl_algo.data['QE'])
    # np.save('QPi', irl_algo.data['QPi'])

    # use found reward to generate policies for visualization
    cg = cg.update_rewards(r[-1])
    policies = cg.find_best_policies()

    # np.save('loss', irl_algo.data['loss'])

    mdp.visualize(cg.graph, policies, show_edges=False)

    plt.figure(figsize=(9, 6))
    plt.plot(irl_algo.data['loss'])

    plt.show()


if __name__ == '__main__':
    learn_reward()


from __future__ import division

import json
import argparse

import numpy as np

import matplotlib
matplotlib.use('Qt4Agg')

from matplotlib import pyplot as plt

from sirl.domains.navigation.social_navigation import SocialNavMDP
from sirl.domains.navigation.local_controllers import POSQLocalController
from sirl.domains.navigation.local_controllers import LinearLocalController
from sirl.domains.navigation.reward_functions import SimpleBehaviors
from sirl.domains.navigation.social_navigation import SocialNavEnvironment

from sirl.algorithms.controller_graph import ControllerGraph
from sirl.models.parameters import ControllerGraphParams


# behavior parameters
BEHAVIOR = 'polite'
WEIGHTS = {
    'polite': [-1.0, -0.7, -0.85],
    'sociable': [-1.0, +0.2, -0.95]
}

# world parameters
STARTS = ((0.5, 0.5), (4, 0.1), (2, 3), (8.5, 5.2),
          (8.9, 0.1), (0.1, 8.5), (4, 3))
GOAL = (5.5, 9)

f = open('metropolis.json', 'r')
scene = json.load(f)
f.close()
persons = scene['persons']
persons = {int(k): v for k, v in persons.items()}
relations = scene['relations']

world = SocialNavEnvironment(0, 0, 10, 10, persons, relations, GOAL, STARTS)


def demo_representation_learning(init_type, controller):
    sreward = SimpleBehaviors(world, WEIGHTS[BEHAVIOR], scaled=False,
                              behavior=BEHAVIOR, anisotropic=False,
                              thresh_p=0.45, thresh_r=0.2)

    mdp = SocialNavMDP(discount=0.95, reward=sreward, world=world)

    if controller == 'posq':
        local_controller = POSQLocalController(world, base=0.4,
                                               resolution=0.1)
    elif controller == 'linear':
        local_controller = LinearLocalController(world, resolution=0.1)
    else:
        raise ValueError('Invalid controller value: {}, \
                         expecting (posq, linear)'.format(controller))

    # load controller graph parameters
    params = ControllerGraphParams()
    params.load('cg_params.json')
    params.init_type = init_type

    cg = ControllerGraph(mdp=mdp, local_controller=local_controller,
                         params=params)

    if init_type == 'trajectory':
        samples = np.load('expert_demos.npy')
    elif init_type == 'random':
        samples = [(5, 5), (1, 3)]
    else:
        raise ValueError('Invalid initialization type')

    # explicit flag for additional state attributes e.g. speed, orientation
    cg.initialize_state_graph(samples=samples, extra_state_attr=True)
    cg = cg.run()

    mdp.visualize(cg.graph, cg.policies, show_edges=True, show_waypoints=True)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--controller", type=str, required=True,
                        help="Controller type, [posq, linear]")
    parser.add_argument("-i", "--init_type", type=str, required=True,
                        help="Initialization type, [trajectory, random]")

    args = parser.parse_args()
    demo_representation_learning(args.init_type, args.controller)

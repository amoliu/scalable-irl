# sample-irl
Sample based (hierarchical) Bayesian inverse reinforcement learning using adaptive state graphs for MDP representation.

## Features
- Efficient and flexible graph based (hierarchical) representation.
- Ability to incoporate task specific constrains directly into the MDP representation.
- Admits efficient IRL algorithms on sampled trajectories via MCMC of direct optimization using L-BFGS.

## Installation
```bash
git clone https://github.com/makokal/sample-irl.git
python setup.py build_ext -i
python setup.py develop  # For local development without global install
[sudo] python setup.py install  # for global install
```

## Roadmap
- [ ] More value approximation/projection methods (e.g. Nystrom)
- [ ] More guided sampling strategies/heuristics
- [ ] Model-free RL solvers

## Authors
1. [Billy Okal](https://github.com/makokal)

If you use this software, please cite the following paper

```
@InProceedings{okalRSSLfd15,
   author = {Okal, Billy and Gilbert, Hugo and Arras, Kai O.},
    title = {Efficient Inverse Reinforcement Learning using Adaptive State Graphs},
    booktitle={Robotics: Science and Systems (RSS), Workshop on Learning from Demonstration: Inverse optimal control, Reinforcement learning and Lifelong learning},
    address = {Rome, Italy},
    year={2015},
}
```

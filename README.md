# Scalable inverse reinforcement learning 
This repo contains code for scalable IRL approach that involves two key steps:
   1. Representation learning via adaptive state graphs
   2. IRL over (infinite) state and action spaces via sampled trajectories


## Features
- Efficient and flexible graph based (hierarchical) representation.
- Ability to incoporate task specific constrains directly into the MDP representation (the graph).
- Admits efficient IRL algorithms on sampled trajectories.
   - Currently on BIRL variants available

## Installation
```bash
git clone https://github.com/makokal/scalable-irl.git
python setup.py build_ext -i
python setup.py develop  # For local development without global install
[sudo] python setup.py install  # for global install
```

## Roadmap
- [ ] More value approximation/projection methods (e.g. Nystrom)
- [ ] More guided sampling strategies/heuristics
- [ ] Model-free RL solvers
- [ ] Additional IRL variants, e.g. LP, MaxEnt

## Contributions
Pull requests, issues are always welcome

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

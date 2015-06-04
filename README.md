# sample-irl
Sample based (hierarchical, continuous) inverse reinforcement learning using adaptive state graphs for MDP representation.

## Features
- Efficient and flexible graph based representation
- Ability to incoporate task specific constrains directly into the representation
- Admits efficient IRL algorithms on sampled trajectories

## Installation
```bash
git clone [repo url] 
python setup.py build_ext -i
python setup.py develop  # For local development without global install
```

## Roadmap
- [ ] More projection methods (e.g. Nystrom)
- [ ] More guided sampling strategies
- [ ] Use with other IRL algorithms e.g. SCIRL, LMDP-IOC with do not require solving the forward problem



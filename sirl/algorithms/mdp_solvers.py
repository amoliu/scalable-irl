r"""
Markov Decision Process (MDP) Solvers
=======================================

A set of MDP solvers, including
    * Policy iteration
    * Prioritized Sweeping (pending)
"""


def graph_policy_iteration(mdp, epsilon=1e-05, maxit=20):
    """ Graph policy iteration for use with adaptive state graphs

    Perform policy iteration on the MDP graph. The value function and
    action-value function (Q-function) are stored in the nodes/states
    because the number of actions vary per state depending on the no
    of outgoing edges present.

    .. math::
        V(s) = r(s, \pi(s)) + \gamma^d V(\pi(s))

    The reward depends both on state and action (edges)

    Parameters
    ----------
    mdp : ``GraphMDP`` derivative object
        The state graph representing the MDP
    epsilon : float, optional (default: 1e-05)
        Value change threshold for Bellman backup
    maxit : int
        Maximum number of iterations of the policy iteration sweeps


    Note
    -----
    The resulting values and Q functions are modified in-place on the graph
    """
    it = 0
    policy_stable = False

    G = mdp.graph
    gna = G.gna
    gea = G.gea
    sna = G.sna
    gamma = mdp.gamma

    while not policy_stable:
        finished = False
        while not finished:
            max_changed = 0
            # computation of value function
            for n in G.nodes:
                if len(G.out_edges(n)) > 0:
                    e = G.out_edges(n)[gna(n, 'pi')]
                    nn = e[1]
                    old_V = gna(n, 'V')
                    nV = gea(e[0], e[1], 'reward') +\
                        (gamma**max(gea(e[0], e[1], 'duration'), 1)) *\
                        gna(nn, 'V')
                    sna(n, 'V', nV)

                    if abs(nV - old_V) > max_changed:
                        max_changed = abs(nV - old_V)

            if max_changed < epsilon:
                finished = True

        changed = False
        for n in G.nodes:
            if len(G.out_edges(n)) > 0:
                rewards = [gea(e[0], e[1], 'reward') for e in G.out_edges(n)]
                times = [gea(e[0], e[1], 'duration') for e in G.out_edges(n)]
                next_nodes = [e[1] for e in G.out_edges(n)]

                nQ = [r + (gamma ** max(t, 1)) * gna(nn, 'V')
                      for r, t, nn in zip(rewards, times, next_nodes)]
                sna(n, 'Q', nQ)
                old_pol = gna(n, 'pi')
                sna(n, 'pi', nQ.index(max(nQ)))
                if gna(n, 'pi') != old_pol:
                    changed = True
        it += 1

        if changed is False or it == maxit:
            policy_stable = True

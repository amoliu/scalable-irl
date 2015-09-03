

from nose.tools import assert_equal
from numpy.testing import assert_array_equal

from sirl.models.state_graph import StateGraph


def make_test_graph():
    return StateGraph(state_dim=2)


def test_add_node():
    g = make_test_graph()
    g.add_node(nid=0, data=(1, 1), cost=1,
               priority=1, Q=[], V=1, pi=0, ntype='simple')
    assert_equal(len(g.nodes), 1)
    g.add_node(nid=1, data=(3, 2), cost=3,
               priority=1, Q=[], V=10, pi=0, ntype='simple')
    assert_equal(len(g.nodes), 2)


def test_add_edge():
    g = make_test_graph()
    g.add_node(nid=0, data=(1, 1), cost=1,
               priority=1, Q=[], V=1, pi=0, ntype='simple')
    g.add_node(nid=1, data=(3, 3), cost=3,
               priority=1, Q=[], V=10, pi=0, ntype='simple')
    phi = [1, 2, 3]
    traj = [(0, 0), (1, 1)]
    g.add_edge(0, 1, 3, 20, phi, traj)
    g.add_edge(1, 0, 3, 40, phi, traj)
    assert_equal(len(g.all_edges), 2)
    assert_equal(g.edge_exists(0, 1), True)
    assert_equal(g.edge_exists(1, 0), True)
    assert_equal(g.edge_exists(1, 1), False)
    assert_equal(g.edge_exists(0, 0), False)


def test_node_attributes():
    g = make_test_graph()
    g.add_node(nid=0, data=(1, 1), cost=1,
               priority=1, Q=[], V=1, pi=0, ntype='simple')
    g.sna(0, 'priority', 5)
    g.sna(0, 'data', (4, 9))
    assert_equal(g.G.node[0]['priority'], 5)
    assert_equal(g.gna(0, 'priority'), 5)
    assert_array_equal(g.G.node[0]['data'], (4, 9))
    assert_equal(g.gna(0, 'data'), (4, 9))


def test_edge_attributes():
    g = make_test_graph()
    g.add_node(nid=0, data=(1, 1), cost=1,
               priority=1, Q=[], V=1, pi=0, ntype='simple')
    g.add_node(nid=1, data=(3, 3), cost=3,
               priority=1, Q=[], V=10, pi=0, ntype='simple')
    phi = [1, 2, 3]
    traj = [(0, 0), (1, 1)]
    g.add_edge(0, 1, 3, 20, phi, traj)
    g.add_edge(1, 0, 3, 40, phi, traj)
    g.sea(0, 1, 'reward', 100)
    g.sea(0, 1, 'duration', 1000)
    assert_equal(g.G.edge[0][1]['reward'], 100)
    assert_equal(g.gea(0, 1, 'reward'), 100)
    assert_equal(g.G.edge[0][1]['duration'], 1000)
    assert_equal(g.gea(0, 1, 'duration'), 1000)


def test_out_edges():
    g = make_test_graph()
    g.add_node(nid=0, data=(1, 1), cost=1,
               priority=1, Q=[], V=1, pi=0, ntype='simple')
    g.add_node(nid=1, data=(3, 3), cost=3,
               priority=1, Q=[], V=10, pi=0, ntype='simple')
    g.add_node(nid=2, data=(2, 6), cost=3,
               priority=1, Q=[], V=10, pi=0, ntype='simple')
    phi = [1, 2, 3]
    traj = [(0, 0), (1, 1)]
    g.add_edge(0, 1, 3, 20, phi, traj)
    g.add_edge(1, 0, 3, 40, phi, traj)
    g.add_edge(0, 2, 4, 50, phi, traj)
    assert_equal(len(g.out_edges(0)), 2)
    assert_equal(len(g.out_edges(1)), 1)
    assert_equal(len(g.out_edges(2)), 0)


def test_filter_nodes_by_type():
    g = make_test_graph()
    g.add_node(nid=0, data=(1, 1), cost=1,
               priority=1, Q=[], V=1, pi=0, ntype='simple')
    g.add_node(nid=1, data=(3, 3), cost=3,
               priority=1, Q=[], V=10, pi=0, ntype='goal')
    g.add_node(nid=2, data=(2, 6), cost=3,
               priority=1, Q=[], V=10, pi=0, ntype='start')
    assert_equal(len(g.filter_nodes_by_type('simple')), 1)
    assert_equal(len(g.filter_nodes_by_type('goal')), 1)
    assert_equal(len(g.filter_nodes_by_type('start')), 1)
    assert_equal(len(g.filter_nodes_by_type('path')), 0)


def test_find_neighbors_range():
    g = make_test_graph()
    g.add_node(nid=0, data=(1, 1), cost=1,
               priority=1, Q=[], V=1, pi=0, ntype='simple')
    g.add_node(nid=1, data=(3.5, 1), cost=3,
               priority=1, Q=[], V=10, pi=0, ntype='goal')
    g.add_node(nid=2, data=(6, 1), cost=3,
               priority=1, Q=[], V=10, pi=0, ntype='start')
    assert_equal(len(g.find_neighbors_range(0, 4)), 1)
    assert_equal(len(g.find_neighbors_range(0, 7)), 2)
    assert_equal(len(g.find_neighbors_range(0, 2)), 0)

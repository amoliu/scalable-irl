

import warnings

import networkx as nx
from math import sqrt


class StateGraph(object):
    """ State Graph """

    _node_attrs = ('id', 'data', 'cost', 'priority', 'Q', 'V', 'pi', 'type')
    _edge_attrs = ('source', 'target', 'duration', 'reward')

    def __init__(self):
        self._g = nx.DiGraph()
        self._node_ids = set()  # keep track of node ids

    def add_node(self, nid, data, cost, priority, Q, V, pi, ntype):
        """
        Add a new node to the graph
        """
        if nid not in self.G:
            self.G.add_node(nid, data=data, cost=cost, priority=priority,
                            Q=Q, V=V, pi=pi, type=ntype)
            self._node_ids.add(nid)
        else:
            warnings.warn('Node already exits in the graph, not added')

    def add_edge(self, source, target, duration, reward):
        """
        Add a new edge into the graph
        """
        if source == target:
            warnings.warn('WARN: source and target nodes are the same')

        elif not self.G.has_edge(source, target):
            self.G.add_edge(source, target, duration=duration, reward=reward)
        else:
            warnings.warn('Edge ({}--{}) already exists in the graph'
                          .format(source, target))

    def remove_edge(self, source, target):
        """ Remove an edge from the graph """
        if source == target:
            warnings.warn('WARN: source and target nodes are the same')

        self.G.remove_edge(source, target)

    def remove_node(self, node):
        """ Remove a node from the graph """
        self.G.remove_node(node)

    def edge_exists(self, source, target):
        """ Check if an edge already exists in the graph """
        return self.G.has_edge(source, target)

    def gna(self, node_id, attribute):
        """
        Get a single attribute of a single node
        Parameters
        ------------
        node_id : int
        attribute : string
        """
        self._check_node_attributes(node_id, attribute)
        return self.G.node[node_id][attribute]

    def sna(self, node_id, attribute, value):
        """
        Set a single attribute of a node
        Parameters
        ------------
        node_id : int
        attribute : string
        value : any
        """
        self._check_node_attributes(node_id, attribute)
        self.G.node[node_id][attribute] = value

    def gea(self, source, target, attribute):
        """
        Get a single attribute of a single edge
        """
        self._check_edge_attributes(source, target, attribute)
        return self.G.edge[source][target][attribute]

    def sea(self, source, target, attribute, value):
        """
        Set a single attribute of a edge between source and target
        """
        self._check_edge_attributes(source, target, attribute)
        self.G.edge[source][target][attribute] = value

    def find_neighbors_range(self, nid, distance):
        """ Find node neigbors within distance range"""
        cn = self.gna(nid, 'data')
        neigbors = filter(lambda n: eud(self.gna(n, 'data'), cn) <= distance,
                          self.G.nodes())
        return neigbors

    def find_neighbors_k(self, nid, k):
        """ Find k nearest neighbors based on Euclidean distance """
        cn = self.gna(nid, 'data')
        distances = {n: eud(self.gna(n, 'data'), cn) for n in self.G.nodes()}
        sorted_neighbors = sorted(distances.items(), key=lambda x: x[1])
        k_neighbors = sorted_neighbors[:k]
        return list(n[0] for n in k_neighbors)

    def neighbors(self, nid):
        """ Get the connected node neighbors """
        return self.G.neighbors(nid)

    def edges(self, nid):
        """ Return the edges of a node """
        return self.G.edges(nid)

    def filter_nodes_by_type(self, ntype):
        """ Filter nodes by node type """
        sns = filter(lambda n: self.gna(n, 'type') == ntype, self.nodes)
        return sns

    def policy(self):
        """ Return the policy function for the whole graph  """
        return [self.gna(n, 'pi') for n in self.nodes]

    def plot_graph(self, ax=None, path=[]):
        """
        Save the graph to file
        """
        ncolors = {'simple': 'orange', 'path': 'r', 'start': 'b', 'goal': 'g'}
        ecolors = {'simple': 'k', 'path': 'r', 'start': 'b', 'goal': 'g'}

        nodes = self.G.nodes(data=True)
        node_list = list()
        node_color_array = list()
        for n in nodes:
            node_list.append(n[0])
            if n in path:
                node_color_array.append(ncolors['path'])
            else:
                node_color_array.append(ncolors[n[1]['type']])
                # node_color_array.append(ncolors['simple'])

        edges = self.G.edges(data=True)
        edge_list = list()
        edge_color_array = list()
        for e in edges:
            edge_list.append((e[0], e[1]))
            # edge_color_array.append(ecolors[e[2]['type']])
            # if e[0] in path or e[1] in path:
            # edge_color_array.append(ecolors['path'])
            # else:
            edge_color_array.append(ecolors['start'])

        nx.draw_networkx(self.G,
                         pos=nx.get_node_attributes(self.G, 'data'),
                         width=0.7,
                         edge_list=edge_list,
                         edge_color=edge_color_array,
                         node_list=node_list,
                         node_color=node_color_array,
                         node_size=160,
                         fontsize=8,
                         ax=ax)

    def _check_node_attributes(self, node_id, attribute):
        assert attribute in self._node_attrs,\
            'Attribute [{}] is invalid | Expected:{}'\
            .format(attribute, self._node_attrs)
        assert node_id in self._node_ids, \
            'Node ({}) not in the graph'.format(node_id)

    def _check_edge_attributes(self, source, target, attribute):
        assert attribute in self._edge_attrs, \
            'Attribute [{}] is invalid | Expected:{}'\
            .format(attribute, self._edge_attrs)
        assert self.G.has_edge(source, target),\
            'Edge [{}-{}] does not exist in the graph'.format(source, target)

    @property
    def G(self):
        return self._g

    @property
    def nodes(self):
        return self.G.nodes()

    @property
    def nodes_data(self):
        return self.G.nodes(data=True)


def eud(data1, data2):
    return sqrt((data1[0]-data2[0])**2 + (data1[1]-data2[1])**2)


import igraph


class StateGraph(object):

    """ The State Graph representing an MDP """

    _graph = None

    def __init__(self):
        self._graph = igraph.Graph(directed=True)

    def add_node(self, **params):
        """
        Add a new node to the graph with properties
        Parameters
        -------------
        id : int
            Node id for unique indentification
        data : numpy array
            State data (e.g. location, configuration in high dimentions)
        cost : float
            Cost of the node (from some initial node)
        priority : float
            Node priority for exploration
        TODO - finish doc
        """
        self._graph.add_vertex(name=params.get('id'),
                               data=params.get('data'),
                               cost=params.get('cost', 0),
                               priority=params.get('priority', 0),
                               quality=params.get('quality', [0]),
                               value=params.get('value', 0),
                               policy=params.get('policy', 0),
                               )

    def add_edge(self, source, target, duration, reward):
        """
        Add a new edge into the graph
        """
        self._graph.add_edge(source=source,
                             target=target,
                             duration=duration,
                             reward=reward)

    def get_node_attribute(self, node_id, attribute):
        """
        Get node attribute
        """
        attr = self._graph.vs[node_id][attribute][0]
        return attr

    def remove_node(self, node_id):
        pass

    def remove_edge(self, source, target):
        pass

    @property
    def G(self):
        return self._graph

    @property
    def nodes(self):
        return self.G.vs

    @property
    def edges(self):
        return self.G.es


# TODO - make a state graph visitor class for seaching for best paths - see cookbook
#
#

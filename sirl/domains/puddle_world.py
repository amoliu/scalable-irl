
from __future__ import division

import numpy as np

# from matplotlib.patches import Circle, Ellipse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge, Circle
import matplotlib.cm as cm
import matplotlib as mpl


from ..models import LocalController
from ..models import GraphMDP
from ..models import MDPReward
from ..models import _controller_duration

from ..utils.geometry import edist, distance_to_segment
from ..algorithms.mdp_solvers import graph_policy_iteration


########################################################################

class PuddleWorldControler(LocalController):
    """ PuddleWorldControler local controller """
    def __init__(self, kind='linear'):
        super(PuddleWorldControler, self).__init__(kind)

    def __call__(self, state, action, duration):
        """ Run a local controller from a ``state`` using ``action``
        """
        nx = state[0] + np.cos(action * 2 * np.pi) * duration
        ny = state[1] + np.sin(action * 2 * np.pi) * duration

        if 0 < nx < 1 and 0 < ny < 1:
            return (nx, ny)
        return state

########################################################################


class PuddleReward(MDPReward):
    """ Reward function for the puddle world """
    def __init__(self, puddles, discount, kind='linfa'):
        super(PuddleReward, self).__init__(kind)
        self._puddles = puddles
        self._gamma = discount

    def __call__(self, state_a, state_b):
        source, target = np.array(state_a), np.array(state_b)
        # increase resolution of action trajectory (option)
        duration = _controller_duration(source, target)
        action_traj = [target * t / duration + source * (1 - t / duration)
                       for t in range(int(duration))]
        action_traj.append(target)
        action_traj = np.array(action_traj)

        reward = []
        for i, wp in enumerate(action_traj):
            reward.append(sum(p.cost(wp[0], wp[1])
                          for p in self._puddles)*self._gamma**i)

        return sum(reward), reward

    @property
    def dim(self):
        return 1

########################################################################


class PuddleWorldMDP(GraphMDP):
    """ Puddle world MDP

    Parameters
    ------------
    discount : float
        MDP discount factor
    reward : ``SocialNavReward`` object
        Reward function for social navigation task
    controller : ``SocialNavLocalController`` object
        Local controller for the task
    params : ``GraphMDPParams`` object
        Algorithm parameters for the various steps

    """
    def __init__(self, discount, reward, controller, params):
        super(PuddleWorldMDP, self).__init__(discount, reward,
                                             controller, params)
        self._setup_puddles()
        self._recording = False
        self._demos = list()

    def initialize_state_graph(self, samples):
        GR = self._params.goal_reward
        CLIMIT = self._params.max_cost
        for start in self._params.start_states:
            self._g.add_node(nid=self._node_id, data=start, cost=0,
                             priority=1, V=GR, pi=0, Q=[], ntype='start')
            self._node_id += 1

        self._g.add_node(nid=self._node_id, data=self._params.goal_state,
                         cost=-CLIMIT, priority=1, V=GR, pi=0,
                         Q=[], ntype='goal')
        self._node_id += 1

        # - add the init samples
        init_samples = list(samples)
        for sample in init_samples:
            self._g.add_node(nid=self._node_id, data=sample, cost=-CLIMIT,
                             priority=1, V=GR, pi=0, Q=[], ntype='simple')
            self._node_id += 1

        # - add edges between each pair
        for n in self._g.nodes:
            for m in self._g.nodes:
                if n == m or self.terminal(n):
                    continue
                ndata, mdata = self._g.gna(n, 'data'), self._g.gna(m, 'data')
                r, phi = self._reward(ndata, mdata)
                d = _controller_duration(ndata, mdata)
                self._g.add_edge(source=n, target=m, reward=r,
                                 duration=d, phi=phi)

        # - update graph attributes
        self._update_state_costs()
        graph_policy_iteration(self)
        self._update_state_priorities()
        self._find_best_policies()

    def terminal(self, state):
        """ Check if a state is terminal (goal state) """
        position = self._g.gna(state, 'data')
        if edist(position, self._params.goal_state) < 0.05:
            return True
        return False

    def visualize(self):
        self._setup_visuals()
        self._plot_graph_in_world()

    # -------------------------------------------------------------
    # internals
    # -------------------------------------------------------------

    def _setup_puddles(self):
        self.puddles = list()
        self.puddles.append(Puddle(0.1, 0.75, 0.45, 0.75, 0.1))
        self.puddles.append(Puddle(0.45, 0.4, 0.45, 0.8, 0.1))

    def _setup_visuals(self):
        """Setup visual elements
        """
        self.figure = plt.figure(figsize=(12, 9))
        self.ax = plt.axes([0, 0, 0.8, 1])
        self.figure.add_axes(self.ax)

        self.record_status = self.figure.text(0.825, 0.3, 'Recording [OFF]',
                                              fontsize=14, color='blue')
        self.figure.text(0.825, 0.2, '#Demos: ', fontsize=10)
        self.demo_count = self.figure.text(0.925, 0.2, '0', fontsize=10)
        # catch events
        self.figure.canvas.mpl_connect('key_press_event', self._key_press)
        self.figure.canvas.mpl_connect('button_press_event', self._btn_press)

        # Main rectangle showing the environment
        self.ax.add_artist(Rectangle((0, 0), width=1, height=1, color='c',
                           zorder=0, ec='k', lw=8, fill=False))

        # draw goal region
        points = [[1, 1], [1, 0.95], [0.95, 1]]
        goal_polygon = plt.Polygon(points, color='green')
        self.ax.add_patch(goal_polygon)

        # draw puddles
        x1 = self.puddles[0].start_pose[0]
        y1 = self.puddles[0].start_pose[1]-0.05
        pd1 = Rectangle((x1, y1), height=0.1, width=self.puddles[0].length,
                        color='brown', alpha=0.7, aa=True, lw=0)
        self.ax.add_artist(pd1)
        self.ax.add_artist(Wedge(self.puddles[0].start_pose, 0.05, 90, 270,
                           fc='brown', alpha=0.7, aa=True, lw=0))
        self.ax.add_artist(Wedge(self.puddles[0].end_pose, 0.05, 270, 90,
                           fc='brown', alpha=0.7, aa=True, lw=0))

        x2 = self.puddles[1].start_pose[0]-0.05
        y2 = self.puddles[1].start_pose[1]
        pd2 = Rectangle((x2, y2), width=0.1, height=self.puddles[1].length,
                        color='brown', alpha=0.7)
        self.ax.add_artist(pd2)
        self.ax.add_artist(Wedge(self.puddles[1].start_pose, 0.05, 180, 360,
                           fc='brown', alpha=0.7, aa=True, lw=0))
        self.ax.add_artist(Wedge(self.puddles[1].end_pose, 0.05, 0, 180,
                           fc='brown', alpha=0.7, aa=True, lw=0))

        # draw the agent at initial pose
        robot_start = (0.3, 0.65)
        robot_visual = Circle(robot_start, 0.01, fc='b', ec='k', zorder=3)
        self.ax.add_artist(robot_visual)
        self.robot = Agent(position=robot_start, orientation=(1, 1),
                           visual=robot_visual)
        self.ax.set_xlim([0, 1])
        self.ax.set_ylim([0, 1])
        self.ax.set_xticks([])
        self.ax.set_yticks([])

    def _key_press(self, event):
        """Handler for key press events"""
        if event.key == 'R':
            print('Starting recordning demos')
            if self._recording is False:
                print('Recording new demonstration')
                self.record_status.set_text("Recording [ON]")
                self.new_demo = list()
                self.demo_color = np.random.choice(['r', 'b', 'g',
                                                   'k', 'm', 'y', 'c'])
                self._recording = True
            else:
                print('Done recording demo')
                self._demos.append(np.array(self.new_demo))
                self.record_status.set_text("Recording [OFF]")
                self.demo_count.set_text(str(len(self._demos)))
                self._recording = False

                if len(self._demos):
                    last_demo = self._demos[len(self._demos) - 1]
                    self.ax.plot(last_demo[:, 0], last_demo[:, 1],
                                 color=self.demo_color, ls='-', lw=1)
        elif event.key == 'S':
            if self._recording:
                print('Please finish recording before saving')
            else:
                print('Saving demos as: demos.npy')
                d = np.array(self._demos)
                print(d)
                # np.save('demos.npy', d)
        self.figure.canvas.draw()

    def _btn_press(self, event):
        if self._recording:
            self.new_demo.append([event.xdata, event.ydata])
            cc = self.demo_color
            self.ax.add_artist(Circle((event.xdata, event.ydata),
                               0.005, fc=cc, ec=cc))
            self.figure.canvas.draw()

    def _plot_graph_in_world(self, show_rewards=False):
        """ Shows the lattest version of the world with MDP
        """
        G = self._g
        gna = G.gna
        gea = G.gea

        if show_rewards:
            rewards = [gea(e[0], e[1], 'reward') for e in G.all_edges]
            n = mpl.colors.Normalize(vmin=min(rewards), vmax=max(rewards))
            m = cm.ScalarMappable(norm=n, cmap=cm.jet)

        values = [gna(n, 'V') for n in G.nodes]
        nv = mpl.colors.Normalize(vmin=min(values), vmax=max(values))
        mv = cm.ScalarMappable(norm=nv, cmap=cm.jet)

        best_nodes = set()
        for traj in self._best_trajs:
            for state in traj:
                best_nodes.add(state)

        for i, n in enumerate(G.nodes):
            [posx, posy] = gna(n, 'data')
            if gna(n, 'type') == 'start':
                color = ['black', 'black']
                nr = 1.0
            elif self.terminal(n):
                color = ['green', 'black']
                nr = 1.5
            elif n in best_nodes:
                color = ['green', 'green']
                nr = 0.5
            else:
                # rgcol = _rgb_to_hex(((0, 0, 255 * i / float(n_nodes))))
                rgcol = mv.to_rgba(gna(n, 'V'))
                color = [rgcol, rgcol]
                nr = 0.5
            self.ax.add_artist(Circle((posx, posy), nr/100., fc=color[0],
                               ec=color[1], lw=1.5, zorder=3))

            p = gna(n, 'pi')
            ndata = gna(n, 'data')
            for i, e in enumerate(G.out_edges(n)):
                t = e[1]
                tdata = gna(t, 'data')
                x1, y1 = ndata[0], ndata[1]
                x2, y2 = tdata[0], tdata[1]
                if n in best_nodes and i == p:
                    self.ax.plot((x1, x2), (y1, y2), ls='-',
                                 lw=2.0, c='g', zorder=3)
                else:
                    if not show_rewards:
                        self.ax.plot((x1, x2), (y1, y2), ls='-', lw=1.0,
                                     c='0.7', alpha=0.5)
                    else:
                        cost = gea(e[0], e[1], 'reward')
                        self.ax.arrow(x1, y1, 0.97*(x2-x1), 0.97*(y2-y1),
                                      width=0.01, head_width=0.15,
                                      head_length=0.15,
                                      fc=m.to_rgba(cost), ec=m.to_rgba(cost))


########################################################################


class Agent(object):
    """ A agent object """
    def __init__(self, position, orientation, visual, **kwargs):
        self.position = position
        self.orientation = orientation
        self.visual = visual


class Puddle(object):
    """ A puddle in a continous puddle world
    Represented by combinations of a line and semi-circles at each end,
    i.e. (-----------)

    Parameters
    -----------
    x1 : float
        X coordinate of the start of the center line
    x2 : float
        X coordinate of the end of the center line
    y1 : float
        Y coordinate of the start of the center line
    y2 : float
        Y coordinate of the end of the center line
    radius : float
        Thickness/breadth of the puddle in all directions

    Attributes
    -----------
    start_pose : array-like
        1D numpy array with the start of the line at the puddle center line
    end_pose: array-like
        1D numpy array with the end of the line at the puddle center line
    radius: float
        Thickness/breadth of the puddle in all directions
    """
    def __init__(self, x1, y1, x2, y2, radius, **kwargs):
        assert x1 >= 0 and x1 <= 1, 'Puddle coordinates must be in [0, 1]'
        assert x2 >= 0 and x2 <= 1, 'Puddle coordinates must be in [0, 1]'
        assert y1 >= 0 and y1 <= 1, 'Puddle coordinates must be in [0, 1]'
        assert y2 >= 0 and y2 <= 1, 'Puddle coordinates must be in [0, 1]'
        assert radius > 0, 'Puddle radius must be > 0'
        self.start_pose = np.array([x1, y1])
        self.end_pose = np.array([x2, y2])
        self.radius = radius

    def cost(self, x, y):
        dist_puddle, inside = distance_to_segment(self.start_pose,
                                                  self.end_pose, (x, y))
        if inside and dist_puddle < self.radius:
            return -400.0 * (self.radius - dist_puddle)
        return 0.0

    @property
    def location(self):
        return self.start_pose[0], self.start_pose[1],\
            self.end_pose[0], self.end_pose[1]

    @property
    def length(self):
        return edist(self.start_pose, self.end_pose)

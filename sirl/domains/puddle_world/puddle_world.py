
from __future__ import division

import numpy as np

# from matplotlib.patches import Circle, Ellipse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge, Circle
import matplotlib.cm as cm
import matplotlib as mpl


from ...models.base import LocalController
from ...models.base import MDP
from ...models.base import MDPReward
from ...models.base import Environment

from ...utils.geometry import edist, distance_to_segment


__all__ = [
    'PuddleWorldControler',
    'PuddleReward',
    'PuddleRewardOriented',
    'PuddleWorldEnvironment',
    'PuddleWorldMDP',
    'Puddle',
]


########################################################################

class PuddleWorldControler(LocalController):
    """ PuddleWorldControler local controller """
    def __init__(self, world, kind='linear'):
        super(PuddleWorldControler, self).__init__(world, kind)

        # the agents moves by 0.05 at every step as per the original
        # task definition
        self._resolution = 0.01

    def __call__(self, state, action, duration, *others):
        """ Run a local controller from a ``state`` using ``action``
        """
        nx = state[0] + np.cos(action) * duration
        ny = state[1] + np.sin(action) * duration

        if self._world.in_world((nx, ny)):
            target = [nx, ny]
            traj = self.trajectory(state, target)
            return target, traj

        return state, None

    def trajectory(self, source, target, *others):
        """ Compute trajectories between two states"""
        source = np.asarray(source)
        target = np.asarray(target)
        duration = edist(source, target)
        dt = duration * 1.0 / self._resolution
        traj = [target * t / dt + source * (1 - t / dt)
                for t in range(int(dt))]
        traj.append(source)
        traj = np.array(traj)
        return traj


########################################################################


class PuddleReward(MDPReward):
    """ Reward function for the puddle world """
    def __init__(self, world, kind='linfa'):
        super(PuddleReward, self).__init__(world, kind)

    def __call__(self, state, action):
        gamma = 0.98
        reward = []
        for i, wp in enumerate(action):
            reward.append(sum(p.cost(wp[0], wp[1])
                          for p in self._world.puddles)*gamma**i)

        return sum(reward), reward

    @property
    def dim(self):
        return 1


class PuddleRewardOriented(MDPReward):
    """ Reward function for the puddle world (adding goal orientation) """
    def __init__(self, world, weights, kind='linfa'):
        super(PuddleRewardOriented, self).__init__(world, kind)
        weights = np.asarray(weights)
        assert weights.size == self.dim,\
            'Expecting {}-dim weight vector'.format(self.dim)
        self._weights = weights
        self._gamma = 0.9

    def __call__(self, state, action):
        phi = [self._puddle_penalty(action),
               self._goal_orientation(action),
               100.0*action.shape[0]]
        r = np.dot(phi, self._weights)
        return r, phi

    @property
    def dim(self):
        return 3

    def _puddle_penalty(self, action):
        pen = 0.0
        for i, wp in enumerate(action):
            pen += (sum(p.cost(wp[0], wp[1])
                    for p in self._world.puddles)*self._gamma**i)
        return pen

    def _goal_orientation(self, action):
        dist = 0.0
        for i in range(action.shape[0] - 1):
            dnow = edist(self._world.goal, action[i])
            dnext = edist(self._world.goal, action[i + 1])
            dist += max((dnext - dnow), 0) * 100 * self._gamma**i

        return dist

########################################################################


class PuddleWorldEnvironment(Environment):
    """ Social Navigation World """
    def __init__(self, start, goal, puddles=None, **kwargs):
        super(PuddleWorldEnvironment, self).__init__(start, goal)

        if puddles is not None:
            self.puddles = puddles
        else:
            self._setup_default_puddles()

    def in_world(self, state):
        return 0.0 < state[0] < 1.0 and 0.0 < state[1] < 1.0

    def _setup_default_puddles(self):
        self.puddles = list()
        self.puddles.append(Puddle(0.1, 0.75, 0.45, 0.75, 0.1))
        self.puddles.append(Puddle(0.45, 0.4, 0.45, 0.8, 0.1))


class PuddleWorldMDP(MDP):
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
    def __init__(self, discount, reward, world):
        super(PuddleWorldMDP, self).__init__(discount, reward)

        self._world = world
        self._recording = False
        self._demos = list()

    def terminal(self, state):
        """ Check if a state is terminal (goal state) """
        if edist(state, self._world.goal) < 0.05:
            return True
        return False

    @property
    def state_dimension(self):
        return 2

    @property
    def start_states(self):
        return self._world.start

    @property
    def goal_state(self):
        return self._world.goal

    def visualize(self, G, policies, show_edges=False):
        self._setup_visuals()
        self._plot_graph_in_world(G, policies, show_edges)

    # -------------------------------------------------------------
    # internals
    # -------------------------------------------------------------

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
        x1 = self._world.puddles[0].start[0]
        y1 = self._world.puddles[0].start[1]-0.05
        width = self._world.puddles[0].length
        height = self._world.puddles[1].length
        pd1 = Rectangle((x1, y1), height=0.1, width=width,
                        color='brown', alpha=0.7, aa=True, lw=0)
        self.ax.add_artist(pd1)
        self.ax.add_artist(Wedge(self._world.puddles[0].start, 0.05, 90, 270,
                           fc='brown', alpha=0.7, aa=True, lw=0))
        self.ax.add_artist(Wedge(self._world.puddles[0].end, 0.05, 270, 90,
                           fc='brown', alpha=0.7, aa=True, lw=0))

        x2 = self._world.puddles[1].start[0]-0.05
        y2 = self._world.puddles[1].start[1]
        pd2 = Rectangle((x2, y2), width=0.1, height=height,
                        color='brown', alpha=0.7)
        self.ax.add_artist(pd2)
        self.ax.add_artist(Wedge(self._world.puddles[1].start, 0.05, 180, 360,
                           fc='brown', alpha=0.7, aa=True, lw=0))
        self.ax.add_artist(Wedge(self._world.puddles[1].end, 0.05, 0, 180,
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

    def _plot_graph_in_world(self, G, policies, show_edges):
        """ Shows the lattest version of the world with MDP
        """
        gna = G.gna
        values = [gna(n, 'V') for n in G.nodes]
        nv = mpl.colors.Normalize(vmin=min(values), vmax=max(values))
        mv = cm.ScalarMappable(norm=nv, cmap=cm.jet)

        best_nodes = set()
        for traj in policies:
            for state in traj:
                best_nodes.add(state)

        for i, n in enumerate(G.nodes):
            posx, posy = gna(n, 'data')
            if gna(n, 'type') == 'start':
                color = 'black'
                nr = 1.0
            elif self.terminal((posx, posy)):
                color = 'green'
                nr = 1.5
            elif n in best_nodes:
                color = 'green'
                nr = 0.5
            else:
                color = mv.to_rgba(gna(n, 'V'))
                nr = 0.5
            self.ax.add_artist(Circle((posx, posy), nr/100., fc=color,
                               ec=color, lw=1.5, zorder=3))

            p = gna(n, 'pi')
            ndata = gna(n, 'data')
            for i, e in enumerate(G.out_edges(n)):
                tdata = gna(e[1], 'data')
                x1, y1 = ndata[0], ndata[1]
                x2, y2 = tdata[0], tdata[1]

                if n in best_nodes and i == p:
                    self.ax.plot((x1, x2), (y1, y2), ls='-',
                                 lw=2.0, c='g', zorder=3)
                else:
                    if show_edges:
                        self.ax.plot((x1, x2), (y1, y2), ls='-', lw=1.0,
                                     c='0.7', alpha=0.5)


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
    x1, x2, y1, y2 : float
        Coordinates of the puddle midline
    radius : float
        Thickness/breadth of the puddle in all directions

    Attributes
    -----------
    start : array-like
        1D numpy array with the start of the line at the puddle center line
    end: array-like
        1D numpy array with the end of the line at the puddle center line
    radius: float
        Thickness/breadth of the puddle in all directions

    """
    PCOST = 100

    def __init__(self, x1, y1, x2, y2, radius, **kwargs):
        assert x1 >= 0 and x1 <= 1, 'Puddle coordinates must be in [0, 1]'
        assert x2 >= 0 and x2 <= 1, 'Puddle coordinates must be in [0, 1]'
        assert y1 >= 0 and y1 <= 1, 'Puddle coordinates must be in [0, 1]'
        assert y2 >= 0 and y2 <= 1, 'Puddle coordinates must be in [0, 1]'
        assert radius > 0, 'Puddle radius must be > 0'
        self.start = np.array([x1, y1])
        self.end = np.array([x2, y2])
        self.radius = radius

    def cost(self, x, y):
        dist_puddle, inside = distance_to_segment((x, y), (self.start,
                                                  self.end))
        if inside:
            if dist_puddle < self.radius:
                return -self.PCOST * (self.radius - dist_puddle)
        else:
            d = min(edist((x, y), self.start), edist((x, y), self.end))
            if d < self.radius:
                return -self.PCOST * (self.radius - d)
        return 0.0

    @property
    def location(self):
        return self.start[0], self.start[1],\
            self.end[0], self.end[1]

    @property
    def length(self):
        return edist(self.start, self.end)


from __future__ import division

import numpy as np

from matplotlib.patches import Circle, Ellipse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

from ...models.base import MDP
from ...models.base import Environment
from ...utils.geometry import edist


__all__ = ['SocialNavEnvironment', 'SocialNavMDP']


########################################################################

class SocialNavEnvironment(Environment):
    """ Social Navigation World """
    def __init__(self, x, y, w, h, persons, relations,
                 goal, starts, **kwargs):
        super(SocialNavEnvironment, self).__init__(starts, goal)
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.persons = persons
        self.relations = relations

    def in_world(self, state):
        return self.x < state[0] < self.w and\
                self.y < state[1] < self.h


########################################################################


class SocialNavMDP(MDP):
    """ Social Navigation Adaptive State-Graph (SocialNavMDP)

    Social navigation task MDP represented by an adaptive state graph

    Parameters
    ------------
    discount : float
        MDP discount factor
    reward : :class:`SocialNavReward` object
        Reward function for social navigation task
    world : :class:`SocialNavWorld` object
        Configuration of the navigation task world


    Attributes
    -----------
    _world : :class:`SocialNavWorld`
        Configuration of the navigation task world

    """
    def __init__(self, discount, reward, world):
        super(SocialNavMDP, self).__init__(discount, reward)

        self._world = world

        # manual demonstration recording
        self._recording = False
        self._demos = list()

    def terminal(self, state):
        """ Check if a state is terminal (goal state)
        state is a vector of information such as position
        """
        if edist(state, self._world.goal) < 0.05:
            return True
        return False

    @property
    def state_dimension(self):
        return 4

    @property
    def start_states(self):
        return self._world.start

    @property
    def goal_state(self):
        return self._world.goal

    def visualize(self, G, policies, fsize=(12, 9),
                  show_edges=False, show_waypoints=False,
                  recording=False):
        """ Visualize the social navigation world

        Allows recording of demonstrations and also display of final
        graph representing the MDP
        """
        self._setup_visuals(fsize)

        for _, p in self._world.persons.items():
            phead = np.degrees(np.arctan2(p[3], p[2]))
            self.ax.add_artist(Ellipse((p[0], p[1]), width=0.3, height=0.6,
                               angle=phead, color='r', fill=False, lw=1.5,
                               aa=True, zorder=3))
            self.ax.add_artist(Circle((p[0], p[1]), radius=0.12, color='w',
                               ec='r', lw=2.5, aa=True, zorder=3))
            self.ax.arrow(p[0], p[1], p[2]/5., p[3]/5., fc='r', ec='r', lw=1.5,
                          head_width=0.14, head_length=0.1, zorder=3)

            # speed = np.hypot(p[2], p[3])
            # hz = speed * 0.55
            # self.ax.add_artist(Circle((p[0], p[1]), radius=hz, color='r',
            #                    ec='r', lw=1, aa=True, alpha=0.2))

        for [i, j] in self._world.relations:
            x1, y1 = self._world.persons[i][0], self._world.persons[i][1]
            x2, y2 = self._world.persons[j][0], self._world.persons[j][1]
            self.ax.plot((x1, x2), (y1, y2), ls='-', c='r', lw=2.0, zorder=2)

        if recording:
            g = G.mdp.goal_state
            starts = G.mdp.start_states
            self.ax.add_artist(Circle((g[0], g[1]), 1.5/10., fc='g',
                               ec='g', lw=1.5, zorder=3))

            for s in starts:
                self.ax.add_artist(Circle((s[0], s[1]), 1.5/10., fc='k',
                                   ec='k', lw=1.5, zorder=3))

            return self.ax

        self._plot_graph_in_world(G, policies, show_edges, show_waypoints)

        return self.ax

    # -------------------------------------------------------------
    # internals
    # -------------------------------------------------------------

    def _setup_visuals(self, fsize=(12, 9)):
        """ Prepare figure axes for plotting """
        self.figure = plt.figure(figsize=fsize)
        self.ax = plt.axes([0, 0, 0.8, 1])
        # self.ax = plt.axes([0, 0, 1, 1])
        self.figure.add_axes(self.ax)
        self.ax.set_xlim([self._world.x, self._world.w])
        self.ax.set_ylim([self._world.y, self._world.h])

        self.record_status = self.figure.text(0.825, 0.3, 'Recording [OFF]',
                                              fontsize=14, color='blue')
        self.figure.text(0.825, 0.2, '#Demos: ', fontsize=10)
        self.demo_count = self.figure.text(0.925, 0.2, '0', fontsize=10)

        self.figure.canvas.mpl_connect('key_press_event', self._key_press)
        self.figure.canvas.mpl_connect('button_press_event', self._btn_click)

    def _key_press(self, event):
        """Handler for key press events
        Used for demonstration recording and experiment handling
        """
        if event.key == 'R':
            print('Starting recording demos')
            if self._recording is False:
                print('Recording new demonstration')
                self.record_status.set_text("Recording [ON]")
                self.new_demo = list()
                self.demo_color = np.random.choice(['r', 'b', 'g', 'k',
                                                   'm', 'y', 'c'])
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
                fname = 'demos_metropolis2.npy'
                print(d, fname)
                np.save(fname, d)
        self.figure.canvas.draw()

    def _btn_click(self, event):
        if self._recording:
            self.new_demo.append([event.xdata, event.ydata])
            cc = self.demo_color
            self.ax.add_artist(Circle((event.xdata, event.ydata),
                               0.03, fc=cc, ec=cc))
            self.figure.canvas.draw()

    def _plot_graph_in_world(self, G, policies, show_edges, show_waypoints):
        """ Shows the latest version of the world with MDP
        """
        gna = G.gna
        gea = G.gea

        values = [gna(n, 'V') for n in G.nodes]
        nv = mpl.colors.Normalize(vmin=min(values), vmax=max(values))
        mv = cm.ScalarMappable(norm=nv, cmap=cm.viridis)

        best_nodes = set()
        for traj in policies:
            for state in traj:
                best_nodes.add(state)

        for i, n in enumerate(G.nodes):
            posx, posy, _, _ = gna(n, 'data')
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
            self.ax.add_artist(Circle((posx, posy), nr/10., fc=color,
                               ec=color, lw=1.5, zorder=3))

            p = gna(n, 'pi')
            ndata = gna(n, 'data')
            for i, e in enumerate(G.out_edges(n)):
                tdata = gna(e[1], 'data')
                x1, y1 = ndata[0], ndata[1]
                x2, y2 = tdata[0], tdata[1]

                if not show_waypoints:
                    if n in best_nodes and i == p:
                        self.ax.plot((x1, x2), (y1, y2), ls='-',
                                     lw=2.0, c='g', zorder=3)
                    else:
                        if show_edges:
                            self.ax.plot((x1, x2), (y1, y2), ls='--', lw=1.0,
                                         c='0.7', alpha=0.5)
                else:
                    if n in best_nodes and i == p:
                        traj = gea(e[0], e[1], 'traj')
                        for wp in traj:
                            v = wp[3]
                            vx, vy = v*np.cos(wp[2]), v*np.sin(wp[2])
                            self.ax.arrow(wp[0], wp[1], 0.2*vx, 0.2*vy, fc='g',
                                          ec='g', lw=1.0, head_width=0.08,
                                          head_length=0.05, zorder=3)
                    else:
                        if show_edges:
                            traj = gea(e[0], e[1], 'traj')
                            for wp in traj:
                                v = wp[3]
                                vx, vy = v*np.cos(wp[2]), v*np.sin(wp[2])
                                self.ax.arrow(wp[0], wp[1], 0.1*vx, 0.1*vy,
                                              fc='0.7', ec='0.7', lw=0.5,
                                              head_width=0.04,
                                              head_length=0.03, zorder=1)


# -------------------------------------------------------------
# simple utils
# -------------------------------------------------------------


def _rgb_to_hex(rgb):
    return ('#%02X%02X%02X' % (rgb[0], rgb[1], rgb[2]))

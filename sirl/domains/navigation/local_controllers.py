from __future__ import division

import numpy as np
import copy

from ...models import LocalController
from ...utils.geometry import normangle, edist
from .social_navigation import WorldConfig


class LinearLocalController(LocalController):

    """ Social Navigation linear local controller

    Social navigation task linear local controller, which connects states
    using straight lines as actions (options, here considered deterministic).
    The action is thus fully represented by a single float for the angle of
    the line.

    Parameters
    -----------
    kind : string
        LocalController controller type for book-keeping

     Attributes
    -----------
    _wconfig : ``WorldConfig``
        Configuration of the navigation task world

    """

    def __init__(self, world_config, resolution=0.2, kind='linear'):
        super(LinearLocalController, self).__init__(kind)
        assert isinstance(world_config, WorldConfig), 'Expect WorldConfig'
        self._wconfig = world_config
        self._resolution = resolution

    def __call__(self, state, action, duration, max_speed):
        """ Run a local controller from a state

        Run the local controller at the given ``state`` using the ``action``
        represented by an angle, :math:` \\alpha \in [0, \pi]` for a time limit
        given by ``duration``

        Parameters
        -----------
        state : array of shape (2)
            Positional data of the state (assuming 0:2 are coordinates)
        action : float
            Angle representing the action taken
        duration : float
            Real time interval limit for executing the controller
        max_speed : float
            Local speed limit

        Returns
        --------
        new_state : array of shape (2)
            New state reached by the controller
        trajectory : array of shape(N, 2)
            Local trajectory result
        Note
        ----
        If the local controller ends up beyond the limits of the world config,
        then the current state is returned to avoid sampling `outside' and
        `None` is returned as trajectory.
        """
        nx = state[0] + np.cos(action) * duration
        ny = state[1] + np.sin(action) * duration

        if self._wconfig.x < nx < self._wconfig.w and\
                self._wconfig.y < ny < self._wconfig.h:
            dt = (max_speed * duration) * 1.0 / self._resolution
            start = np.array([state[0], state[1]])
            target = np.array([nx, ny])
            traj = [
                target * t / dt + start * (1 - t / dt) for t in range(int(dt))]
            traj.append(target)
            traj = np.array(traj)
            return target, traj

        return state, None

    def trajectory(self, start, target, max_speed):
        """ Compute trajectories between two states"""
        start = np.array(start)
        target = np.array(target)
        duration = edist(start, target)
        dt = (max_speed * duration) * 1.0 / self._resolution
        traj = [target * t / dt + start * (1 - t / dt) for t in range(int(dt))]
        traj.append(target)
        traj = np.array(traj)
        return traj


########################################################################


class POSQLocalController(LocalController):

    """ Local controller based on Two-point boundary value problem solver"""

    def __init__(self, world_config, resolution=0.1, base=0.4, kind='linear'):
        super(POSQLocalController, self).__init__(kind)
        self._wconfig = world_config
        self._resolution = resolution  # deltaT
        self._base = base

    def __call__(self, state, action, duration, max_speed):
        nx = state[0] + np.cos(action) * duration
        ny = state[1] + np.sin(action) * duration

        if self._wconfig.x < nx < self._wconfig.w and\
                self._wconfig.y < ny < self._wconfig.h:
            start = np.array([state[0], state[1], 0])
            target = np.array([nx, ny, np.pi/2])

            print('Running posq with: {}, {}'.format(start, target))
            traj = self.trajectory(start, target, max_speed)
            return target, traj

        return state, traj

    def trajectory(self, start, target, max_speed):
        """ Compute trajectories between two states using POSQ"""
        direction = 0
        initT = 0

        traj, speedvec, vel, inct = self._posq_integrate(
            start, target, direction, self._resolution,
            self._base, initT, nS=0)

        return traj

    def _posq_integrate(self, xstart, xend, direction, deltaT, b, initT, nS=0):
        """ POSQ Integration procedure to general full trajectory """
        assert xstart.shape == xend.shape, 'Expect similar vector sizes'
        assert xstart.size == xend.size == 3, 'Expect 1D array (x, y, theta)'

        # if size(xstart) == [1, 3], xstart = xstart.T
        # if size(xend)   == [1, 3], xend   = xend.T
        vel = np.zeros(shape=(1, 2))
        # Initialize variables
        sl, sr = 0, 0
        oldSl, oldSr = 0, 0
        xvec = np.zeros(shape=(1, 3))  # pose vectors for trajectory
        speedvec = np.zeros(shape=(1, 2))  # velocities during trajectory
        encoders = [0, 0]
        t = initT  # initialize global timer
        ti = 0  # initialize local timer
        eot = 0  # initialize end-of-trajectory flag
        xcurrent = [xstart[0], xstart[1], xstart[2]]
        oldBeta = 0

        while not eot:
            # Calculate distances for both wheels
            dSl = sl - oldSl
            dSr = sr - oldSr
            dSm = (dSl + dSr) / 2
            dSd = (dSr - dSl) / b

            # Integrate robot position
            xcurrent[0] = xcurrent[0] + dSm * np.cos(xcurrent[2] + dSd / 2.0)
            xcurrent[1] = xcurrent[1] + dSm * np.sin(xcurrent[2] + dSd / 2.0)
            xcurrent[2] = normangle(xcurrent[2] + dSd, -np.pi)

            # implementation of the controller
            vl, vr, eot, vm, vd, oldBeta = self._posq_step(ti, xcurrent, xend,
                                                           direction, b,
                                                           oldBeta)
            vel = np.row_stack((vel, [vm, vd]))
            speeds = np.array([vl, vr])
            speedvec = np.row_stack((speedvec, speeds))
            xvec = np.row_stack((xvec, xcurrent))

            # Increase timers
            ti = ti + deltaT
            t = t + deltaT

            # Increase accumulated encoder values
            # simulated encoders of robot
            delta_dist1 = speeds[0] * deltaT
            delta_dist2 = speeds[1] * deltaT
            encoders[0] += delta_dist1
            encoders[1] += delta_dist2

            # print(encoders)

            # Keep track of previous wheel positions
            oldSl = sl
            oldSr = sr

            # noise on the encoders
            sl = encoders[0] + nS * np.random.uniform(0, 1)
            sr = encoders[1] + nS * np.random.uniform(0, 1)

        inct = t  # at the end of the trajectory the time elapsed is added

        return [xvec, speedvec, vel, inct]

    def _posq_step(self, t, xcurrent, xend, direction, b, oldBeta):
        Kv = 1
        Krho = 3    # Condition: Kalpha + 5/3*Kbeta - 2/pi*Krho > 0 !
        Kalpha = 1
        Kbeta = -1
        Vmax = Krho                # [m/s]
        RhoEndCondition = 0.0510      # [m]

        if t == 0:
            oldBeta = 0

        # extract coordinates
        xc, yc, tc = xcurrent[0], xcurrent[1], xcurrent[2]
        xe, ye, te = xend[0], xend[1], xend[2]

        # rho
        dx = xe - xc
        dy = ye - yc
        rho = np.sqrt(dx**2 + dy**2)
        fRho = rho
        if fRho > (Vmax / Krho):
            fRho = Vmax / Krho

        # alpha
        alpha = np.arctan2(dy, dx) - tc
        alpha = normangle(alpha, -np.pi)

        # direction (forward or backward)
        if direction == 0:
            if alpha > np.pi / 2:
                fRho = -fRho                   # backwards
                alpha = alpha - np.pi
            elif alpha <= -np.pi / 2:
                fRho = -fRho                   # backwards
                alpha = alpha + np.pi
        elif direction == -1:                  # arrive backwards
            fRho = -fRho
            alpha = alpha + np.pi
            if alpha > np.pi:
                alpha = alpha - 2 * np.pi

        # phi
        phi = te - tc
        phi = normangle(phi, -np.pi)

        beta = normangle(phi - alpha, -np.pi)
        if abs(oldBeta - beta) > np.pi:           # avoid instability
            beta = oldBeta
        oldBeta = beta

        # New version
        vm = Krho * np.tanh(fRho * Kv)
        vd = (Kalpha * alpha + Kbeta * beta)
        eot = (rho < RhoEndCondition)

        if eot:
            print('t:{}  x:{}  y:{}  theta:{}'
                  .format(t, xc, yc, tc * 180 / np.pi))

        # print('t:{}  x:{}  y:{}  theta:{}, oldBeta:{}'
        #       .format(t, xc, yc, tc * 180 / np.pi, oldBeta))

        # Convert speed to wheel speeds
        vl = vm - vd * b / 2
        if abs(vl) > Vmax:
            vl = Vmax * np.sign(vl)

        vr = vm + vd * b / 2
        if abs(vr) > Vmax:
            vr = Vmax * np.sign(vr)

        return vl, vr, eot, vm, vd, oldBeta

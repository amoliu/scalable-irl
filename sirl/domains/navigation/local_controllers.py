from __future__ import division

import numpy as np

from ...models.base import LocalController
from ...utils.geometry import normangle, edist
from ...utils.validation import asarray


__all__ = ['LinearLocalController', 'POSQLocalController']


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

    def __init__(self, world, resolution=0.2, kind='linear'):
        super(LinearLocalController, self).__init__(world, kind)
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

        if self._world.in_world((nx, ny)):
            target = [nx, ny, action, max_speed]
            traj = self.trajectory(state, target, max_speed)
            return target, traj

        return state, None

    def trajectory(self, source, target, max_speed):
        """ Compute trajectories between two states"""
        source = asarray(source)
        target = asarray(target)
        duration = edist(source, target)
        dt = (max_speed * duration) * 1.0 / self._resolution
        theta = np.arctan2(target[1]-source[1], target[0]-source[0])

        traj = [target[0:2] * t / dt + source[0:2] * (1 - t / dt)
                for t in range(int(dt))]
        traj = [t.tolist()+[theta, max_speed] for t in traj]
        traj = np.array(traj)
        return traj


########################################################################


class POSQLocalController(LocalController):

    """ Local controller based on Two-point boundary value problem solver"""

    def __init__(self, world, resolution=0.1,
                 base=0.4, kind='linear'):
        super(POSQLocalController, self).__init__(world, kind)
        self._resolution = resolution  # deltaT
        self._base = base

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

        if self._world.in_world((nx, ny)):
            target = [nx, ny, action, max_speed]
            traj = self.trajectory(state, target, max_speed)
            return target, traj

        return state, None

    def trajectory(self, source, target, max_speed):
        """ Compute trajectories between two states using POSQ"""
        theta = np.arctan2(target[1]-source[1], target[0]-source[0])
        # source = asarray([source[0], source[1], source[2]])
        source = asarray([source[0], source[1], theta])
        target = asarray([target[0], target[1], theta])

        direction = 1
        init_t = 0
        traj, speedvec, vel, inct = self._posq_integrate(
            source, target, direction, self._resolution,
            self._base, init_t, max_speed, nS=0)

        speeds = np.hypot(speedvec[:, 0], speedvec[:, 1])
        traj = np.column_stack((traj, speeds))

        return traj

    # -------------------------------------------------------------
    # internals
    # -------------------------------------------------------------

    def _posq_integrate(self, xstart, xend, direction, deltaT,
                        b, initT, vmax, nS=0):
        """ POSQ Integration procedure to general full trajectory """
        assert xstart.shape == xend.shape, 'Expect similar vector sizes'
        assert xstart.size == xend.size == 3, 'Expect 1D array (x, y, theta)'

        vel = np.zeros(shape=(1, 2))
        sl, sr = 0, 0
        old_sl, old_sr = 0, 0
        xvec = np.zeros(shape=(1, 3))  # pose vectors for trajectory
        speedvec = np.zeros(shape=(1, 2))  # velocities during trajectory
        encoders = [0, 0]
        t = initT  # initialize global timer
        ti = 0  # initialize local timer
        eot = 0  # initialize end-of-trajectory flag
        xnow = [xstart[0], xstart[1], xstart[2]]
        old_beta = 0

        while not eot:
            # Calculate distances for both wheels
            dSl = sl - old_sl
            dSr = sr - old_sr
            dSm = (dSl + dSr) / 2
            dSd = (dSr - dSl) / self._base

            # Integrate robot position
            xnow[0] = xnow[0] + dSm * np.cos(xnow[2] + dSd / 2.0)
            xnow[1] = xnow[1] + dSm * np.sin(xnow[2] + dSd / 2.0)
            xnow[2] = normangle(xnow[2] + dSd, -np.pi)

            # implementation of the controller
            vl, vr, eot, vm, vd, old_beta = self._posq_step(ti, xnow, xend,
                                                            direction,
                                                            old_beta, vmax)
            vel = np.row_stack((vel, [vm, vd]))
            speeds = np.array([vl, vr])
            speedvec = np.row_stack((speedvec, speeds))
            xvec = np.row_stack((xvec, xnow))

            # Increase timers
            ti = ti + deltaT
            t = t + deltaT

            # Increase accumulated encoder values
            # simulated encoders of robot
            delta_dist1 = speeds[0] * deltaT
            delta_dist2 = speeds[1] * deltaT
            encoders[0] += delta_dist1
            encoders[1] += delta_dist2

            # Keep track of previous wheel positions
            old_sl = sl
            old_sr = sr

            # noise on the encoders
            sl = encoders[0] + nS * np.random.uniform(0, 1)
            sr = encoders[1] + nS * np.random.uniform(0, 1)

        inct = t  # at the end of the trajectory the time elapsed is added

        return xvec, speedvec, vel, inct

    def _posq_step(self, t, xnow, xend, direction, old_beta, vmax):
        """ POSQ single step """
        k_v = 3.8
        k_rho = 1    # Condition: k_alpha + 5/3*k_beta - 2/pi*k_rho > 0 !
        k_alpha = 6
        k_beta = -1
        rho_end = 0.00510      # [m]

        if t == 0:
            old_beta = 0

        # extract coordinates
        xc, yc, tc = xnow[0], xnow[1], xnow[2]
        xe, ye, te = xend[0], xend[1], xend[2]

        # rho
        dx = xe - xc
        dy = ye - yc
        rho = np.sqrt(dx**2 + dy**2)
        f_rho = rho
        if f_rho > (vmax / k_rho):
            f_rho = vmax / k_rho

        # alpha
        alpha = normangle(np.arctan2(dy, dx) - tc, -np.pi)

        # direction (forward or backward)
        if direction == 1:
            if alpha > np.pi / 2:
                f_rho = -f_rho                   # backwards
                alpha = alpha - np.pi
            elif alpha <= -np.pi / 2:
                f_rho = -f_rho                   # backwards
                alpha = alpha + np.pi
        elif direction == -1:                  # arrive backwards
            f_rho = -f_rho
            alpha = alpha + np.pi
            if alpha > np.pi:
                alpha = alpha - 2 * np.pi

        # phi, beta
        phi = te - tc
        phi = normangle(phi, -np.pi)
        beta = normangle(phi - alpha, -np.pi)
        if abs(old_beta - beta) > np.pi:           # avoid instability
            beta = old_beta
        old_beta = beta

        vm = k_rho * np.tanh(f_rho * k_v)
        vd = (k_alpha * alpha + k_beta * beta)
        eot = (rho < rho_end)

        # Convert speed to wheel speeds
        vl = vm - vd * self._base / 2
        if abs(vl) > vmax:
            vl = vmax * np.sign(vl)

        vr = vm + vd * self._base / 2
        if abs(vr) > vmax:
            vr = vmax * np.sign(vr)

        return vl, vr, eot, vm, vd, old_beta

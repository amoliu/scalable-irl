from __future__ import division

import numpy as np

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
            traj = [target * t/dt + start * (1 - t/dt) for t in range(int(dt))]
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
        traj = [target * t/dt + start * (1 - t/dt) for t in range(int(dt))]
        traj.append(target)
        traj = np.array(traj)
        return traj


########################################################################


class POSQLocalController(LocalController):
    """ Local controller based on Two-point boundary value problem solver"""
    def __init__(self, world_config, resolution=0.2, base=0.4, kind='linear'):
        super(POSQLocalController, self).__init__(kind)
        self._wconfig = world_config
        self._resolution = resolution  # deltaT
        self._base = base

    def __call__(self, state, action, duration, max_speed):
        nx = state[0] + np.cos(action) * duration
        ny = state[1] + np.sin(action) * duration

        if self._wconfig.x < nx < self._wconfig.w and\
                self._wconfig.y < ny < self._wconfig.h:
            start = np.array([state[0], state[1]])
            target = np.array([nx, ny])
            traj = self.trajectory(start, target, max_speed)
            return target, traj

        return state, None

    def trajectory(self, start, target, max_speed):
        """ Compute trajectories between two states using POSQ"""
        traj = None
        # - add POSQ
        return traj

    def _posq_step(self, t, xcurrent, xend, dir, b, oldBeta):
        Kv = 5.9
        Krho = 0.2    # Condition: Kalpha + 5/3*Kbeta - 2/pi*Krho > 0 !
        Kalpha = 6.91
        Kbeta = -1
        Vmax = Krho                # [m/s]
        RhoEndCondition = 0.00000510      # [m]

        if t == 0:
            oldBeta = 0

        # extract coordinates
        xc = xcurrent[0]
        yc = xcurrent[1]
        tc = xcurrent[2]
        xe = xend[0]
        ye = xend[1]
        te = xend[2]
        Verbose = 1

        # rho
        dx = xe - xc
        dy = ye - yc
        rho = np.sqrt(dx**2 + dy**2)
        fRho = rho
        if fRho > (Vmax/Krho):
            fRho = Vmax/Krho

        # alpha
        alpha = np.arctan2(dy, dx) - tc
        alpha = normangle(alpha, -np.pi)

        # direction
        if dir == 0:              # controller choose the forward direction
            if alpha > np.pi/2:
                fRho = -fRho                   # backwards
                alpha = alpha-np.pi
            elif alpha <= -np.pi/2:
                fRho = -fRho                   # backwards
                alpha = alpha+np.pi
        elif dir == -1:                    # arrive backwards
            fRho = -fRho
            alpha = alpha+np.pi
            if alpha > np.pi:
                alpha = alpha - 2*np.pi

        # phi
        phi = te-tc
        phi = normangle(phi, -np.pi)

        beta = normangle(phi-alpha, -np.pi)
        if abs(oldBeta-beta) > np.pi:           # avoid instability
            beta = oldBeta
        oldBeta = beta

        # New version
        vm = Krho*np.tanh(fRho*Kv)
        vd = (Kalpha*alpha + Kbeta*beta)
        eot = (rho < RhoEndCondition)

        if eot and Verbose:
            print('t:{} sec  x:{}  y:{}  theta:{}'
                  .format(t, xc, yc, tc*180/np.pi))

        # Convert speed to wheel speeds
        vl = vm - vd*b/2
        if abs(vl) > Vmax:
            vl = Vmax*np.sign(vl)

        vr = vm + vd*b/2
        if abs(vr) > Vmax:
            vr = Vmax*np.sign(vr)

        return vl, vr, eot, vm, vd, oldBeta

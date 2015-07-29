from __future__ import division

from shapely.geometry import Polygon, Point
import numpy as np

from ..utils.geometry import ray_segment_intersection
from ..utils.geometry import distance_to_segment
from ..utils.geometry import edist


class Annotation(object):
    """ Annotation in a scene e.g. info screen, kiosk, etc

    Modelled as polygonal objects with a face (indicating side from which
    engagements are possible), and a zone (indicating influence region)

    Parameters
    -----------
    geometry : array-like, shape (2 x N)
        Array of polygon vertices in a consistent order
    face : array-like, shape (2 x 2)
        Line indicating the engagement side of the annotation
    zone : float, optional (default: 3)
        Range of influence in the direction of the face (perpendicularly)

    Attributes
    -----------
    _geom : array-like, shape (2 x N)
        Array of polygon vertices in a consistent order
    _face : array-like, shape (2 x 2)
        Line indicating the engagement side of the annotation
    _zone : float
        Range of influence in the direction of the face (perpendicularly)
    """
    def __init__(self, geometry, face, zone=3):
        self._geom = geometry
        self._face = face
        self._zone = zone
        self._compute_influence_area()

    def engaged(self, person):
        """ Check is a person is engaged to an annotation,
        e.g by looking/facing it like in the case of screens
        or kiosks and also being in the influence zone
        """
        ray_origin = (person[0], person[1])
        ray_direction = (person[2], person[3])
        looking = ray_segment_intersection(ray_origin,
                                           ray_direction,
                                           self._face[0],
                                           self._face[1])
        dist = distance_to_segment(ray_origin, self._face[0], self._face[1])
        if dist < self._zone and looking:
            return True

        return False

    def disturbance(self, waypoint, person):
        """ Compute the disturbance induced by a robot stepping into
        the influence zone of an annotation

        Requires that the robot come in between the annotation and at
        least one person engaged by it
        """
        if self.engaged(person) and self._point_in_zone(waypoint):
            dist = distance_to_segment(waypoint, self._face[0], self._face[1])
            return dist

        return 0.0

    @property
    def influence_zone(self):
        """ Get the polygon representing the influence area """
        return list(self._poly.exterior.coords)

    def _point_in_zone(self, point):
        """ Check if a waypoint is in the influence zone"""
        return self._poly.contains(Point(point[0], point[1]))

    def _compute_influence_area(self):
        a, b = self._face[0], self._face[1]
        h = np.hypot(edist(a, b), self._zone)
        theta_a = np.arctan2(edist(a, b), self._zone)
        aprime = (b[0] + h*np.cos(theta_a), b[1] + h*np.sin(theta_a))
        bprime = (a[0] + h*np.cos(theta_a), a[1] + h*np.sin(theta_a))
        self._poly = Polygon([a, b, bprime, aprime])

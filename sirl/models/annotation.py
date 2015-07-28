from __future__ import division


class Annotation(object):
    """ Annotation in a scene e.g. info screen, kiosk, etc"""
    def __init__(self, geometry, zone):
        self.geom = geometry
        self.zone = zone

    def point_in_zone(self, point):
        """ Check if a waypoint is in the influence zone"""
        pass

    def engaged(self, person):
        """ Check is a person is engaged to an annotation,
        e.g by looking/facing it like in the case of screens
        or kiosks
        """
        pass

    def disturbance(self, person):
        """ Compute the disturbance induced by a robot stepping into
        the influence zone of an annotation

        Requires that the robot come in between the annotation and at
        least one person engaged by it
        """
        pass


from nose.tools import assert_equal

from sirl.utils.geometry import edist
from sirl.utils.geometry import distance_to_segment
from sirl.utils.geometry import normangle
from sirl.utils.geometry import trajectory_length
from sirl.utils.geometry import anisotropic_distance


import numpy as np


def test_distance_to_segment():
    # test points
    x1 = np.array([2.0, 2.0])  # colinear inside
    x2 = np.array([4.0, 0.0])  # colinear outside
    x3 = np.array([4.0, 1.0])  # outside not colinear
    x4 = np.array([0.0, 1.0])  # inside not colinear
    x5 = np.array([1.0, 2.0])  # inside not colinear
    x6 = np.array([2.7, 2.7])  # inside not colinear

    # line
    ls = np.array([1.0, 3.0])
    le = np.array([3.0, 1.0])

    assert_equal(distance_to_segment(x1, ls, le)[1], True)
    assert_equal(distance_to_segment(x1, ls, le)[0], 0.0)

    assert_equal(distance_to_segment(x2, ls, le)[1], False)
    assert_equal(distance_to_segment(x3, ls, le)[1], False)

    assert_equal(distance_to_segment(x6, ls, le)[1], True)
    assert_equal(distance_to_segment(x4, ls, le)[1], True)
    assert_equal(distance_to_segment(x5, ls, le)[1], True)


def test_edist():
    # toy data
    pose1 = np.array([2, 2])
    pose2 = np.array([12, 2])
    assert_equal(10, edist(pose1, pose2))
    assert_equal(10, edist(pose1.tolist(), pose2.tolist()))
    assert_equal(10, edist((2, 2), (12, 2)))


def test_normangle():
    assert_equal(normangle(3*np.pi), np.pi)
    assert_equal(normangle(np.pi), np.pi)
    assert_equal(normangle(2*np.pi), 0.0)
    assert_equal(normangle(-np.pi), np.pi)


def test_trajectory_length():
    traj1 = [(0, 0), (3, 0)]
    traj2 = [(0, 0), (3, 4)]
    assert_equal(trajectory_length(traj1), 3.0)
    assert_equal(trajectory_length(traj2), 5.0)


def test_anisotropic_distance():
    pass

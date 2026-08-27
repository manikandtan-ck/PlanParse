import math

from planparse.geometry import angle_difference_deg, merge_collinear_segments, perpendicular_distance, projected_overlap
from planparse.models import LineSegment


def test_angle_normalization():
    assert angle_difference_deg(179, 1) == 2


def test_parallel_segment_distance_and_overlap():
    a = LineSegment((0, 0), (100, 0))
    b = LineSegment((20, 10), (80, 10))
    assert math.isclose(perpendicular_distance(a, b), 10)
    assert math.isclose(projected_overlap(a, b), 0.6)


def test_collinear_merge():
    merged = merge_collinear_segments([LineSegment((0, 0), (40, 0)), LineSegment((50, 1), (100, 1))], gap_px=15)
    assert len(merged) == 1
    assert merged[0].length > 99


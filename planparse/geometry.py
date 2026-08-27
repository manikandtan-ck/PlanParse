import math
from itertools import combinations

import numpy as np

from .models import LineSegment


def angle_difference_deg(a: float, b: float) -> float:
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def unit_direction(line: LineSegment) -> np.ndarray:
    v = np.array([line.p2[0] - line.p1[0], line.p2[1] - line.p1[1]], dtype=float)
    n = np.linalg.norm(v)
    return v / n if n else np.zeros(2)


def point_line_distance(point: tuple[float, float], line: LineSegment) -> float:
    v = np.array([line.p2[0] - line.p1[0], line.p2[1] - line.p1[1]], dtype=float)
    w = np.array([point[0] - line.p1[0], point[1] - line.p1[1]], dtype=float)
    n = np.linalg.norm(v)
    return float(abs(v[0] * w[1] - v[1] * w[0]) / n) if n else float("inf")


def perpendicular_distance(a: LineSegment, b: LineSegment) -> float:
    return point_line_distance(b.p1, a)


def projected_overlap(a: LineSegment, b: LineSegment) -> float:
    d = unit_direction(a)
    if not np.any(d):
        return 0.0
    origin = np.array(a.p1)
    a_proj = sorted([0.0, float(np.dot(np.array(a.p2) - origin, d))])
    b_proj = sorted([float(np.dot(np.array(b.p1) - origin, d)), float(np.dot(np.array(b.p2) - origin, d))])
    overlap = max(0.0, min(a_proj[1], b_proj[1]) - max(a_proj[0], b_proj[0]))
    return float(overlap / max(1e-9, max(a.length, b.length)))


def midpoint_line(a: LineSegment, b: LineSegment) -> LineSegment:
    d = unit_direction(a)
    if np.dot(d, unit_direction(b)) < 0:
        b = LineSegment(b.p2, b.p1, b.source_paths)
    origin = np.array(a.p1)
    t = sorted([0.0, float(np.dot(np.array(a.p2) - origin, d))])
    b_t = sorted([float(np.dot(np.array(b.p1) - origin, d)), float(np.dot(np.array(b.p2) - origin, d))])
    start, end = min(t[0], b_t[0]), max(t[1], b_t[1])
    normal = np.array([-d[1], d[0]])
    offset = float(np.dot(np.array(b.p1) - np.array(a.p1), normal))
    p1 = np.array(a.p1) + d * start + normal * (offset / 2)
    p2 = np.array(a.p1) + d * end + normal * (offset / 2)
    return LineSegment(tuple(p1), tuple(p2), tuple(sorted(set(a.source_paths + b.source_paths))))


def merge_collinear_segments(segments: list[LineSegment], angle_deg: float = 3.0, offset_px: float = 3.0, gap_px: float = 18.0) -> list[LineSegment]:
    """Merge nearby collinear segments with a quadratic pairwise scan."""
    remaining = list(segments)
    merged: list[LineSegment] = []
    while remaining:
        current = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            for i, other in enumerate(remaining):
                if angle_difference_deg(current.angle_deg, other.angle_deg) > angle_deg:
                    continue
                if point_line_distance(other.p1, current) > offset_px and point_line_distance(other.p2, current) > offset_px:
                    continue
                if projected_overlap(current, other) > 0 or _projected_gap(current, other) <= gap_px:
                    current = _merge_extent(current, other)
                    remaining.pop(i)
                    changed = True
                    break
        merged.append(current)
    return merged


def _projected_gap(a: LineSegment, b: LineSegment) -> float:
    d = unit_direction(a)
    origin = np.array(a.p1)
    a_t = sorted([0.0, float(np.dot(np.array(a.p2) - origin, d))])
    b_t = sorted([float(np.dot(np.array(b.p1) - origin, d)), float(np.dot(np.array(b.p2) - origin, d))])
    return max(0.0, max(a_t[0], b_t[0]) - min(a_t[1], b_t[1]))


def _merge_extent(a: LineSegment, b: LineSegment) -> LineSegment:
    d = unit_direction(a)
    origin = np.array(a.p1)
    points = [np.array(a.p1), np.array(a.p2), np.array(b.p1), np.array(b.p2)]
    values = [float(np.dot(p - origin, d)) for p in points]
    return LineSegment(tuple(origin + d * min(values)), tuple(origin + d * max(values)), tuple(sorted(set(a.source_paths + b.source_paths))))


def paired_wall_candidates(segments: list[LineSegment], config) -> list:
    from .models import Wall

    walls = []
    for a, b in combinations(segments, 2):
        if min(a.length, b.length) < config.min_segment_length_px:
            continue
        if angle_difference_deg(a.angle_deg, b.angle_deg) > config.max_parallel_angle_deg:
            continue
        spacing = perpendicular_distance(a, b)
        if not config.min_wall_spacing_px <= spacing <= config.max_wall_spacing_px:
            continue
        overlap = projected_overlap(a, b)
        if overlap < config.min_pair_overlap:
            continue
        angle_error = angle_difference_deg(a.angle_deg, b.angle_deg)
        angle_score = max(0.0, 1.0 - angle_error / config.max_parallel_angle_deg)
        overlap_score = min(1.0, overlap)
        spacing_midpoint = (config.min_wall_spacing_px + config.max_wall_spacing_px) / 2
        spacing_half_range = max(1.0, (config.max_wall_spacing_px - config.min_wall_spacing_px) / 2)
        spacing_score = max(0.0, 1.0 - abs(spacing - spacing_midpoint) / spacing_half_range)
        length_score = min(1.0, min(a.length, b.length) / max(config.min_segment_length_px * 5, 1.0))
        centerline = midpoint_line(a, b)
        if centerline.length < config.min_wall_length_px:
            continue
        vector_score = (
            config.vector_angle_weight * angle_score
            + config.vector_overlap_weight * overlap_score
            + config.vector_spacing_weight * spacing_score
            + config.vector_length_weight * length_score
        )
        walls.append(Wall(centerline, spacing, vector_score, vector_score=vector_score, source_paths=tuple(sorted(set(a.source_paths + b.source_paths))), line_a=a, line_b=b, angle_difference_deg=angle_error, projected_overlap=overlap))
    return walls


def suppress_duplicate_walls(walls: list, angle_deg: float = 3.0, center_offset_px: float = 6.0, overlap: float = 0.70, thickness_delta_px: float = 8.0) -> list:
    """Keep the strongest geometrically equivalent candidate."""
    kept = []
    for wall in sorted(walls, key=lambda item: item.confidence, reverse=True):
        duplicate = any(
            angle_difference_deg(wall.centerline.angle_deg, other.centerline.angle_deg) <= angle_deg
            and perpendicular_distance(wall.centerline, other.centerline) <= center_offset_px
            and projected_overlap(wall.centerline, other.centerline) >= overlap
            and abs(wall.thickness_px - other.thickness_px) <= thickness_delta_px
            for other in kept
        )
        if not duplicate:
            kept.append(wall)
    return kept


def line_to_points(line: LineSegment) -> tuple[tuple[int, int], tuple[int, int]]:
    return (round(line.p1[0]), round(line.p1[1])), (round(line.p2[0]), round(line.p2[1]))

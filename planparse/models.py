from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LineSegment:
    p1: tuple[float, float]
    p2: tuple[float, float]
    source_paths: tuple[int, ...] = ()

    @property
    def length(self) -> float:
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        return float((dx * dx + dy * dy) ** 0.5)

    @property
    def angle_deg(self) -> float:
        import math

        return math.degrees(math.atan2(self.p2[1] - self.p1[1], self.p2[0] - self.p1[0])) % 180.0

    def as_dict(self) -> dict[str, Any]:
        return {"p1": [round(self.p1[0], 3), round(self.p1[1], 3)], "p2": [round(self.p2[0], 3), round(self.p2[1], 3)]}


@dataclass
class Wall:
    centerline: LineSegment
    thickness_px: float
    confidence: float
    source: str = "vector"
    vector_score: float = 0.0
    raster_support: float = 0.0
    source_paths: tuple[int, ...] = ()
    line_a: LineSegment | None = None
    line_b: LineSegment | None = None
    angle_difference_deg: float = 0.0
    projected_overlap: float = 0.0

    def as_dict(self, index: int) -> dict[str, Any]:
        return {
            "id": f"wall_{index:04d}",
            "centerline": self.centerline.as_dict(),
            "thickness_px": round(self.thickness_px, 3),
            "angle_deg": round(self.centerline.angle_deg, 3),
            "length_px": round(self.centerline.length, 3),
            "vector_score": round(self.vector_score, 3),
            "raster_support": round(self.raster_support, 3),
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "source_paths": list(self.source_paths),
            "angle_difference_deg": round(self.angle_difference_deg, 3),
            "projected_overlap": round(self.projected_overlap, 3),
            "paired_lines": {"a": self.line_a.as_dict(), "b": self.line_b.as_dict()} if self.line_a and self.line_b else None,
        }


@dataclass
class WallDetectionConfig:
    min_segment_length_px: float = 18.0
    max_parallel_angle_deg: float = 8.0
    min_pair_overlap: float = 0.45
    min_wall_spacing_px: float = 2.0
    max_wall_spacing_px: float = 80.0
    merge_angle_deg: float = 3.0
    merge_offset_px: float = 3.0
    merge_gap_px: float = 18.0
    min_wall_length_px: float = 50.0
    min_vector_stroke_width: float = 0.5
    vector_weight: float = 0.65
    raster_weight: float = 0.35
    acceptance_threshold: float = 0.60
    weak_threshold: float = 0.45
    vector_angle_weight: float = 0.30
    vector_overlap_weight: float = 0.35
    vector_spacing_weight: float = 0.20
    vector_length_weight: float = 0.15


@dataclass
class RasterResult:
    wall_mask: Any
    centerlines: list[LineSegment] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PdfPageResult:
    image: Any
    vectors: list[LineSegment]
    walls: list[Wall]
    diagnostics: dict[str, Any]
    merged_vectors: list[LineSegment] = field(default_factory=list)
    candidates: list[Wall] = field(default_factory=list)
    weak_walls: list[Wall] = field(default_factory=list)
    raw_vectors: list[LineSegment] = field(default_factory=list)

from __future__ import annotations

from dataclasses import dataclass

from .models import LineSegment


@dataclass
class VectorPrimitive:
    path_index: int
    item_index: int
    line: LineSegment
    stroke_width: float | None = None
    color: tuple | None = None


def extract_line_primitives(drawings, min_length: float = 2.0) -> list[VectorPrimitive]:
    primitives = []
    for path_index, drawing in enumerate(drawings):
        width = drawing.get("width")
        color = drawing.get("color")
        for item_index, item in enumerate(drawing.get("items", [])):
            if not item or item[0] != "l":
                continue
            _, start, end = item[:3]
            line = LineSegment((float(start.x), float(start.y)), (float(end.x), float(end.y)), (path_index,))
            if line.length >= min_length:
                primitives.append(VectorPrimitive(path_index, item_index, line, width, color))
    return primitives

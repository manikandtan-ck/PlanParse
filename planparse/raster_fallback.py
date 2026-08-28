"""Small raster-only wall candidate experiment; production code is untouched."""

from __future__ import annotations

import math
import cv2
import numpy as np

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

from planparse.geometry import line_to_points, merge_collinear_segments, paired_wall_candidates, suppress_duplicate_walls
from planparse.models import LineSegment, PdfPageResult, Wall, WallDetectionConfig
from planparse.pdf import analyze_pdf, render_page
from planparse.raster import raster_stages


ORANGE = (0, 165, 255)


def _line_support(mask: np.ndarray, line: LineSegment, width: int = 3) -> float:
    band = np.zeros(mask.shape[:2], np.uint8)
    cv2.line(band, *line_to_points(line), 1, max(1, width), cv2.LINE_AA)
    return float(mask[band > 0].mean()) if np.any(band) else 0.0


def _hough_segments(response: np.ndarray, orientation: str) -> list[LineSegment]:
    h, w = response.shape[:2]
    minimum = max(24, round(min(h, w) * 0.035))
    gap = max(6, round(min(h, w) * 0.012))
    threshold = max(12, round(min(h, w) * 0.025))
    rows = cv2.HoughLinesP(response, 1, np.pi / 180, threshold, minLineLength=minimum, maxLineGap=gap)
    segments = []
    for row in np.asarray(rows).reshape(-1, 4) if rows is not None else []:
        x1, y1, x2, y2 = map(float, row)
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if orientation == "horizontal" and min(angle, 180 - angle) > 5:
            continue
        if orientation == "vertical" and abs(90 - angle) > 5:
            continue
        segments.append(LineSegment((x1, y1), (x2, y2), ()))
    return segments


def raster_wall_detection(image: np.ndarray) -> PdfPageResult:
    """Extract axis-aligned line pairs from the rendered page image."""
    h, w = image.shape[:2]
    short = min(h, w)
    line_kernel = max(15, round(short * 0.012))
    closing_kernel = 3
    stages = raster_stages(image, line_kernel=line_kernel, closing_kernel=closing_kernel, min_component_area=0)
    horizontal_lines = _hough_segments(stages["horizontal_response"], "horizontal")
    vertical_lines = _hough_segments(stages["vertical_response"], "vertical")
    raw_lines = horizontal_lines + vertical_lines
    merge_offset = max(3, round(short * 0.003))
    merge_gap = max(10, round(short * 0.012))
    merged = merge_collinear_segments(raw_lines, angle_deg=3.0, offset_px=merge_offset, gap_px=merge_gap)
    config = WallDetectionConfig(
        min_segment_length_px=max(24.0, short * 0.035),
        max_parallel_angle_deg=5.0,
        min_pair_overlap=0.45,
        min_wall_spacing_px=max(2.0, short * 0.002),
        max_wall_spacing_px=min(120.0, max(40.0, short * 0.06)),
        min_wall_length_px=max(45.0, short * 0.055),
    )
    raw_candidates = paired_wall_candidates(merged, config)
    binary = stages["binary"] > 0
    for wall in raw_candidates:
        boundary_support = (_line_support(binary, wall.line_a, 3) + _line_support(binary, wall.line_b, 3)) / 2 if wall.line_a and wall.line_b else 0.0
        wall.raster_support = boundary_support
        wall.confidence = 0.55 * wall.vector_score + 0.45 * boundary_support
        wall.source = "raster"
    for wall in raw_candidates:
        wall.source = "raster"
    candidates = suppress_duplicate_walls(raw_candidates, center_offset_px=max(5.0, short * 0.006), thickness_delta_px=max(8.0, short * 0.012))
    candidates.sort(key=lambda item: item.confidence, reverse=True)
    walls = [wall for wall in candidates if wall.confidence >= 0.55]
    weak = [wall for wall in candidates if 0.40 <= wall.confidence < 0.55]
    diagnostics = {
        "document_mode": "raster-only",
        "raster_line_kernel": line_kernel,
        "raster_closing_kernel": closing_kernel,
        "raster_raw_line_count": len(raw_lines),
        "raster_merged_line_count": len(merged),
        "wall_candidate_count": len(raw_candidates),
        "deduplicated_candidate_count": len(candidates),
        "accepted_wall_count": len(walls),
        "weak_wall_count": len(weak),
        "raster_confidence_weights": {"geometry": 0.55, "boundary_support": 0.45},
        "width_px": w,
        "height_px": h,
    }
    diagnostics["debug_stages"] = stages
    return PdfPageResult(image, merged, walls, diagnostics, merged, candidates, weak, raw_lines)


def render_raster_page(file_bytes_or_path, page_idx: int = 0) -> tuple[np.ndarray, dict]:
    """Render one PDF page without extracting native vector candidates."""
    doc = pymupdf.open(stream=file_bytes_or_path, filetype="pdf") if isinstance(file_bytes_or_path, (bytes, bytearray)) else pymupdf.open(str(file_bytes_or_path))
    if not 0 <= page_idx < len(doc):
        doc.close()
        raise IndexError(f"page_idx {page_idx} outside document with {len(doc)} pages")
    page = doc[page_idx]
    image, scale = render_page(page)
    metadata = {"page_count": len(doc), "page_index": page_idx, "width_pt": page.rect.width, "height_pt": page.rect.height, "width_px": image.shape[1], "height_px": image.shape[0], "scale_px_per_pt": scale, "vector_path_count": len(page.get_drawings()), "image_count": len(page.get_images(full=True)), "image_coverage": 0.0, "text_block_count": len(page.get_text("blocks"))}
    doc.close()
    return image, metadata


def analyze_experimental(file_bytes_or_path, page_idx: int = 0, mode: str = "Auto") -> tuple[PdfPageResult, str]:
    auto_mode = mode in ("Auto", "Auto (recommended)")
    if mode in ("Raster", "Raster only"):
        image, metadata = render_raster_page(file_bytes_or_path, page_idx)
        raster_result = raster_wall_detection(image)
        raster_result.diagnostics.update(metadata)
        return raster_result, "Raster"
    vector_mode = "vector" if mode == "Vector" else "hybrid"
    vector_result = analyze_pdf(file_bytes_or_path, page_idx, mode=vector_mode)
    useful_vectors = vector_result.diagnostics.get("filtered_line_segment_count", 0) >= 4
    if mode == "Vector":
        return vector_result, "Vector"
    if mode == "Hybrid" or (auto_mode and useful_vectors):
        return vector_result, "Hybrid"
    raster_result = raster_wall_detection(vector_result.image)
    raster_result.diagnostics.update({key: vector_result.diagnostics.get(key) for key in ("page_index", "page_count", "width_pt", "height_pt", "image_count", "image_coverage", "text_block_count", "scale_px_per_pt", "vector_path_count")})
    return raster_result, "Raster"


def draw_raster_debug(result: PdfPageResult) -> dict[str, np.ndarray]:
    stages = result.diagnostics.get("debug_stages", {})
    original = result.image.copy()
    raw = original.copy()
    for line in result.raw_vectors:
        cv2.line(raw, *line_to_points(line), ORANGE, 1, cv2.LINE_AA)
    pairs = original.copy()
    for wall in result.candidates:
        cv2.line(pairs, *line_to_points(wall.centerline), ORANGE, max(2, round(wall.thickness_px / 2)), cv2.LINE_AA)
    accepted = original.copy()
    for wall in result.walls:
        cv2.line(accepted, *line_to_points(wall.centerline), (0, 210, 0), max(2, round(wall.thickness_px)), cv2.LINE_AA)
    return {
        "foreground_mask": stages.get("binary", np.zeros(original.shape[:2], np.uint8)),
        "horizontal_response": stages.get("horizontal_response", np.zeros(original.shape[:2], np.uint8)),
        "vertical_response": stages.get("vertical_response", np.zeros(original.shape[:2], np.uint8)),
        "raw_raster_lines": raw,
        "raster_wall_pairs": pairs,
        "accepted_raster_candidates": accepted,
    }

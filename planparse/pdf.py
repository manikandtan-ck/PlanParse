from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    import pymupdf
except ImportError:  # older package name
    import fitz as pymupdf

from .geometry import merge_collinear_segments, paired_wall_candidates, suppress_duplicate_walls
from .models import PdfPageResult, WallDetectionConfig
from .raster import raster_stages
from .vectors import extract_line_primitives

MAX_VECTOR_PATHS = 50_000


def render_page(page, dpi: int = 150, max_dimension: int = 2500) -> tuple[np.ndarray, float]:
    scale = dpi / 72.0
    max_scale = max_dimension / max(page.rect.width, page.rect.height)
    scale = min(scale, max_scale)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, scale


def pdf_to_pixel(point: tuple[float, float], scale: float) -> tuple[float, float]:
    return point[0] * scale, point[1] * scale


def pdf_page_count(file_bytes_or_path) -> int:
    doc = pymupdf.open(stream=file_bytes_or_path, filetype="pdf") if isinstance(file_bytes_or_path, (bytes, bytearray)) else pymupdf.open(str(file_bytes_or_path))
    count = len(doc)
    doc.close()
    return count


def analyze_pdf(file_bytes_or_path, page_idx: int = 0, config: WallDetectionConfig | None = None) -> PdfPageResult:
    config = config or WallDetectionConfig()
    doc = pymupdf.open(stream=file_bytes_or_path, filetype="pdf") if isinstance(file_bytes_or_path, (bytes, bytearray)) else pymupdf.open(str(file_bytes_or_path))
    if not 0 <= page_idx < len(doc):
        raise IndexError(f"page_idx {page_idx} outside document with {len(doc)} pages")
    page = doc[page_idx]
    image, scale = render_page(page)
    drawings = page.get_drawings()
    if len(drawings) > MAX_VECTOR_PATHS:
        doc.close()
        raise ValueError(f"Page contains {len(drawings):,} vector paths; limit is {MAX_VECTOR_PATHS:,}.")
    primitives = extract_line_primitives(drawings)
    raw_segments = [p.line for p in primitives]
    raw_pixel_segments = [type(s)(pdf_to_pixel(s.p1, scale), pdf_to_pixel(s.p2, scale), s.source_paths) for s in raw_segments]
    filtered_primitives = [
        p for p in primitives
        if p.line.length * scale >= config.min_segment_length_px
        and (p.stroke_width is None or p.stroke_width >= config.min_vector_stroke_width)
    ]
    filtered = [p.line for p in filtered_primitives]
    pixel_segments = [type(s)(pdf_to_pixel(s.p1, scale), pdf_to_pixel(s.p2, scale), s.source_paths) for s in filtered]
    # Construction sheets commonly reserve a bottom title block; keep this narrow and explainable.
    pixel_segments = [s for s in pixel_segments if not (s.p1[1] > image.shape[0] * 0.86 and s.p2[1] > image.shape[0] * 0.86)]
    merged_segments = merge_collinear_segments(pixel_segments, config.merge_angle_deg, config.merge_offset_px, config.merge_gap_px)
    raw_candidates = paired_wall_candidates(merged_segments, config)
    raster_mask = raster_stages(image, min_component_area=0)["binary"] > 0
    for wall in raw_candidates:
        wall.raster_support = _raster_support(raster_mask, wall.centerline, wall.thickness_px)
        wall.confidence = config.vector_weight * wall.vector_score + config.raster_weight * wall.raster_support
        wall.source = "hybrid"
    candidates = suppress_duplicate_walls(raw_candidates)
    candidates.sort(key=lambda w: w.confidence, reverse=True)
    walls = [wall for wall in candidates if wall.confidence >= config.acceptance_threshold]
    weak_walls = [wall for wall in candidates if config.weak_threshold <= wall.confidence < config.acceptance_threshold]
    image_coverage = _image_coverage(page)
    diagnostics = {"page_count": len(doc), "page_index": page_idx, "width_pt": page.rect.width, "height_pt": page.rect.height, "width_px": image.shape[1], "height_px": image.shape[0], "vector_path_count": len(drawings), "image_count": len(page.get_images(full=True)), "image_coverage": image_coverage, "text_block_count": len(page.get_text("blocks")), "raw_line_segment_count": len(raw_segments), "filtered_line_segment_count": len(pixel_segments), "merged_line_segment_count": len(merged_segments), "wall_candidate_count": len(raw_candidates), "deduplicated_candidate_count": len(candidates), "accepted_wall_count": len(walls), "weak_wall_count": len(weak_walls), "rejected_wall_count": len(raw_candidates) - len(candidates), "document_mode": "hybrid" if image_coverage >= 0.35 else ("vector-rich" if raw_segments else ("raster-only" if not page.get_images(full=True) else "hybrid")), "scale_px_per_pt": scale, "min_vector_stroke_width": config.min_vector_stroke_width}
    doc.close()
    return PdfPageResult(image, pixel_segments, walls, diagnostics, merged_segments, candidates, weak_walls, raw_pixel_segments)


def _raster_support(mask: np.ndarray, line, thickness: float) -> float:
    band = np.zeros(mask.shape, np.uint8)
    p1 = tuple(round(v) for v in line.p1)
    p2 = tuple(round(v) for v in line.p2)
    cv2.line(band, p1, p2, 1, max(1, round(thickness)))
    dilated = cv2.dilate(band, np.ones((3, 3), np.uint8)) > 0
    return float(mask[dilated].mean()) if dilated.any() else 0.0


def _image_coverage(page) -> float:
    """Approximate page area covered by embedded image rectangles."""
    page_area = max(1.0, page.rect.width * page.rect.height)
    rectangles = page.get_image_rects(page.get_images(full=True)[0][0]) if page.get_images(full=True) else []
    for image in page.get_images(full=True)[1:]:
        rectangles.extend(page.get_image_rects(image[0]))
    # Images in the inspected public drawings are tiled without significant overlap.
    covered = sum(max(0.0, r.width) * max(0.0, r.height) for r in rectangles)
    return round(min(1.0, covered / page_area), 3)


def create_synthetic_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=720, height=480)
    for y in (120, 132, 280, 292):
        page.draw_line((80, y), (560, y), color=(0, 0, 0), width=1.5)
    for x in (80, 92, 560, 572):
        page.draw_line((x, 120), (x, 292), color=(0, 0, 0), width=1.5)
    page.draw_line((80, 360), (580, 360), color=(0.6, 0.6, 0.6), width=0.5)
    page.draw_rect((600, 380, 700, 450), color=(0.5, 0.5, 0.5), width=0.5)
    data = doc.tobytes()
    doc.close()
    return data

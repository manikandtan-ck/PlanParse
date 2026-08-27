from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .geometry import line_to_points


GREEN = (0, 210, 0)
MAGENTA = (220, 0, 180)
YELLOW = (0, 210, 210)
LIGHT_GRAY = (165, 165, 165)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 4:
        rgb = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        return np.clip(rgb * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _muted_base(image: np.ndarray) -> np.ndarray:
    base = _to_bgr(image)
    hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = (hsv[:, :, 1].astype(np.float32) * 0.48).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _mask_overlay(base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    mask_bool = np.asarray(mask).astype(bool)
    layer = base.copy()
    layer[mask_bool] = color
    return cv2.addWeighted(layer, alpha, base, 1.0 - alpha, 0)


def _legend(canvas: np.ndarray, items: list[tuple[str, tuple[int, int, int]]], x: int = 24, y: int = 36) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(canvas.shape[:2]) / 1900.0)
    thickness = max(1, round(font_scale * 2))
    row_height = max(24, round(font_scale * 44))
    for index, (label, color) in enumerate(items):
        row_y = y + index * row_height
        cv2.rectangle(canvas, (x, row_y - 15), (x + 18, row_y + 3), color, -1)
        cv2.putText(canvas, label, (x + 28, row_y), font, font_scale, (25, 25, 25), thickness, cv2.LINE_AA)


def render_wall_overlay(
    image: np.ndarray,
    prediction: np.ndarray | None,
    ground_truth: np.ndarray | None = None,
    title: str | None = None,
    metric: float | None = None,
    show_legend: bool = True,
) -> np.ndarray:
    """Render a consistent public wall-overlay view without changing masks."""
    base = _muted_base(image)
    canvas = base.copy()
    legend_items: list[tuple[str, tuple[int, int, int]]] = []
    if ground_truth is not None:
        canvas = _mask_overlay(canvas, ground_truth, MAGENTA, 0.55)
        legend_items.append(("Ground truth", MAGENTA))
    if prediction is not None:
        canvas = _mask_overlay(canvas, prediction, GREEN, 0.82)
        prediction_bool = np.asarray(prediction).astype(bool)
        contours, _ = cv2.findContours(prediction_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline_width = max(2, round(min(canvas.shape[:2]) / 360.0))
        cv2.drawContours(canvas, contours, -1, GREEN, outline_width, cv2.LINE_AA)
        legend_items.insert(0, ("Detected wall", GREEN))
    if title or metric is not None:
        text = title or ""
        if metric is not None:
            text = f"{text}  F1@3px: {metric:.3f}".strip()
        cv2.putText(canvas, text, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, max(0.55, min(canvas.shape[:2]) / 1550.0), (20, 20, 20), 2, cv2.LINE_AA)
    if show_legend and legend_items:
        _legend(canvas, legend_items, y=76 if title or metric is not None else 36)
    return canvas


def draw_wall_overlay(image, walls, opacity: float = 0.75, show_thickness: bool = True) -> np.ndarray:
    base = _to_bgr(image)
    overlay = base.copy()
    for wall in walls:
        p1, p2 = line_to_points(wall.centerline)
        color = GREEN
        width = max(2, round(wall.thickness_px)) if show_thickness else 2
        cv2.line(overlay, p1, p2, color, width, cv2.LINE_AA)
    return cv2.addWeighted(overlay, opacity, base, 1 - opacity, 0)


def draw_vector_overlay(image, segments, color=(150, 150, 150), thickness=1) -> np.ndarray:
    out = _to_bgr(image)
    for segment in segments:
        cv2.line(out, *line_to_points(segment), color, thickness, cv2.LINE_AA)
    return out


def draw_candidate_overlay(image, candidates, show_pairs: bool = True, show_centerlines: bool = True, merged_segments=None) -> np.ndarray:
    out = _to_bgr(image)
    if merged_segments is not None:
        out = draw_vector_overlay(out, merged_segments, LIGHT_GRAY, 1)
    for wall in candidates:
        color = YELLOW
        if show_pairs and wall.line_a and wall.line_b:
            cv2.line(out, *line_to_points(wall.line_a), YELLOW, 1, cv2.LINE_AA)
            cv2.line(out, *line_to_points(wall.line_b), YELLOW, 1, cv2.LINE_AA)
        if show_centerlines:
            cv2.line(out, *line_to_points(wall.centerline), color, max(2, round(wall.thickness_px / 2)), cv2.LINE_AA)
    return out


def benchmark_comparison(image, gt_mask, pred_mask) -> np.ndarray:
    return render_wall_overlay(image, pred_mask, gt_mask, title="Benchmark comparison")


def save_image(path: str | Path, image: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)

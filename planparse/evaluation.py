from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize


def _binary(mask) -> np.ndarray:
    return np.asarray(mask).astype(bool)


def iou(pred, gt) -> float:
    p, g = _binary(pred), _binary(gt)
    union = np.logical_or(p, g).sum()
    return float(np.logical_and(p, g).sum() / union) if union else 1.0


def precision_recall_f1(pred, gt) -> tuple[float, float, float]:
    p, g = _binary(pred), _binary(gt)
    tp = np.logical_and(p, g).sum()
    precision = float(tp / p.sum()) if p.sum() else (1.0 if not g.sum() else 0.0)
    recall = float(tp / g.sum()) if g.sum() else (1.0 if not p.sum() else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, float(f1)


def tolerance_f1(pred, gt, tolerance_px: int = 3) -> float:
    p, g = _binary(pred), _binary(gt)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance_px + 1, 2 * tolerance_px + 1))
    near_gt = cv2.dilate(g.astype(np.uint8), kernel).astype(bool)
    near_pred = cv2.dilate(p.astype(np.uint8), kernel).astype(bool)
    p_score = float(np.logical_and(p, near_gt).sum() / p.sum()) if p.sum() else (1.0 if not g.sum() else 0.0)
    r_score = float(np.logical_and(g, near_pred).sum() / g.sum()) if g.sum() else (1.0 if not p.sum() else 0.0)
    return 2 * p_score * r_score / (p_score + r_score) if p_score + r_score else 0.0


def centerline_chamfer(pred, gt) -> float:
    p = skeletonize(_binary(pred))
    g = skeletonize(_binary(gt))
    if not p.any() and not g.any():
        return 0.0
    if not p.any() or not g.any():
        return float(max(pred.shape))
    d_to_g = cv2.distanceTransform((~g).astype(np.uint8), cv2.DIST_L2, 3)
    d_to_p = cv2.distanceTransform((~p).astype(np.uint8), cv2.DIST_L2, 3)
    return float((d_to_g[p].mean() + d_to_p[g].mean()) / 2)


def score_prediction(pred, gt, processing_time_ms: float = 0.0, sample_id: str = "", mode: str = "raster") -> dict:
    precision, recall, f1 = precision_recall_f1(pred, gt)
    return {"sample_id": sample_id, "mode": mode, "wall_iou": iou(pred, gt), "pixel_precision": precision, "pixel_recall": recall, "pixel_f1": f1, "f1_at_3px": tolerance_f1(pred, gt, 3), "centerline_chamfer_px": centerline_chamfer(pred, gt), "processing_time_ms": processing_time_ms}


def aggregate_results(per_sample: list[dict]) -> dict:
    keys = ["wall_iou", "pixel_precision", "pixel_recall", "pixel_f1", "f1_at_3px", "centerline_chamfer_px", "processing_time_ms"]
    return {"n_samples": len(per_sample), "mean": {key: float(np.mean([row[key] for row in per_sample])) if per_sample else 0.0 for key in keys}, "per_sample": per_sample}


def save_results(results: dict, json_path: str | Path, csv_path: str | Path) -> None:
    Path(json_path).write_text(json.dumps(results, indent=2) + "\n")
    rows = results["per_sample"]
    with Path(csv_path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(rows)

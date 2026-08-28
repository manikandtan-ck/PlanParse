"""Evaluate Vector, Raster, and Hybrid on five fixed FloorPlanCAD SVG drawings."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import tarfile
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import fitz
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from planparse.evaluation import score_prediction
from planparse.pdf import analyze_pdf
from planparse.raster_fallback import raster_wall_detection
from planparse.visualization import MAGENTA, GREEN, draw_wall_overlay, save_image


ARCHIVE_DEFAULT = ROOT.parent / "test-00.tar.xz"
SEED = 42
SELECTED_IDS = ["0036-0119", "0165-0007", "0493-0070", "0557-2134", "1261-0140"]


def archive_svg_ids(archive: Path) -> list[str]:
    with tarfile.open(archive, "r:xz") as tar:
        return sorted(member.name[:-4] for member in tar if member.name.endswith(".svg") and "/" not in member.name)


def selected_ids(archive: Path) -> list[str]:
    return sorted(random.Random(SEED).sample(archive_svg_ids(archive), 5))


def manifest_for(archive: Path, ids: list[str]) -> dict:
    payload = {"source_archive": str(archive), "sample_ids": ids, "seed": SEED, "selection_method": "eligible top-level SVG test IDs sorted, random.Random(42).sample(..., 5), then sorted"}
    payload["manifest_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def _numbers(text: str) -> list[float]:
    return [float(value) for value in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", text)]


def path_segments(d: str) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    tokens = re.findall(r"[A-Za-z]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", d)
    i = 0
    command = None
    current = (0.0, 0.0)
    start = current
    result = []
    while i < len(tokens):
        if tokens[i].isalpha():
            command = tokens[i]
            i += 1
        if command in ("M", "L"):
            needed = 2
            if i + needed > len(tokens) or any(t.isalpha() for t in tokens[i:i + needed]):
                command = None
                continue
            point = (float(tokens[i]), float(tokens[i + 1])); i += 2
            if command == "M":
                current = point; start = point; command = "L"
            else:
                result.append((current, point)); current = point
        elif command in ("m", "l"):
            needed = 2
            if i + needed > len(tokens) or any(t.isalpha() for t in tokens[i:i + needed]):
                command = None
                continue
            point = (current[0] + float(tokens[i]), current[1] + float(tokens[i + 1])); i += 2
            if command == "m":
                current = point; start = point; command = "l"
            else:
                result.append((current, point)); current = point
        elif command in ("H", "h", "V", "v"):
            if i >= len(tokens) or tokens[i].isalpha(): command = None; continue
            value = float(tokens[i]); i += 1
            point = (value, current[1]) if command == "H" else (current[0] + value, current[1]) if command == "h" else (current[0], value) if command == "V" else (current[0], current[1] + value)
            result.append((current, point)); current = point
        elif command in ("A", "a"):
            if i + 7 > len(tokens) or any(t.isalpha() for t in tokens[i:i + 7]): command = None; continue
            values = [float(t) for t in tokens[i:i + 7]]; i += 7
            point = (values[5], values[6]) if command == "A" else (current[0] + values[5], current[1] + values[6])
            result.append((current, point)); current = point
        elif command in ("Z", "z"):
            if current != start: result.append((current, start))
            current = start; command = None
        else:
            command = None
    return result


def _label(element: ET.Element) -> str:
    return " ".join(value for key, value in element.attrib.items() if key.endswith("label"))


def svg_records(svg_path: Path) -> tuple[tuple[float, float, float, float], list[dict], list[dict]]:
    root = ET.parse(svg_path).getroot()
    viewbox = [float(v) for v in root.attrib.get("viewBox", "0 0 100 100").split()]
    min_x, min_y, width, height = viewbox
    all_paths = []
    wall_paths = []

    def visit(element: ET.Element, wall_context: bool = False):
        context = wall_context or bool(re.search(r"wall|墙", _label(element), re.I))
        for child in list(element):
            if child.tag.endswith("path") and child.get("d"):
                record = {"segments": path_segments(child.get("d")), "stroke_width": float(child.get("stroke-width", "0.15").replace("px", "")) if re.match(r"^[\d.]+", child.get("stroke-width", "0.15")) else 0.15}
                all_paths.append(record)
                if context: wall_paths.append(record)
            visit(child, context)
    visit(root)
    return (min_x, min_y, width, height), all_paths, wall_paths


def svg_to_pdf(svg_path: Path, pdf_path: Path) -> tuple[float, float, list[dict]]:
    (min_x, min_y, width, height), paths, wall_paths = svg_records(svg_path)
    # The archive PNGs are 1000px square; at 150 DPI, 1 SVG unit maps to
    # 4.8 PDF points and therefore to 10 rendered pixels.
    scale = 4.8
    doc = fitz.open()
    page = doc.new_page(width=width * scale, height=height * scale)
    for record in paths:
        for p1, p2 in record["segments"]:
            page.draw_line(((p1[0] - min_x) * scale, (p1[1] - min_y) * scale), ((p2[0] - min_x) * scale, (p2[1] - min_y) * scale), color=(0.15, 0.15, 0.15), width=max(0.5, record["stroke_width"] * scale * 0.12))
    doc.save(pdf_path)
    doc.close()
    return width, height, wall_paths


def mask_from_paths(paths: list[dict], shape: tuple[int, int], width: float, height: float) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    sx, sy = shape[1] / width, shape[0] / height
    for record in paths:
        for p1, p2 in record["segments"]:
            a = (round(p1[0] * sx), round(p1[1] * sy)); b = (round(p2[0] * sx), round(p2[1] * sy))
            cv2.line(mask, a, b, 1, max(1, round(record["stroke_width"] * sx * 0.8)), cv2.LINE_AA)
    return mask > 0


def pred_mask(result) -> np.ndarray:
    mask = np.zeros(result.image.shape[:2], np.uint8)
    for wall in result.walls:
        cv2.line(mask, tuple(round(v) for v in wall.centerline.p1), tuple(round(v) for v in wall.centerline.p2), 1, max(1, round(wall.thickness_px)))
    return mask > 0


def resize_panel(image: np.ndarray, size=(360, 360)) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ARCHIVE_DEFAULT)
    parser.add_argument("--staging", type=Path, help="Optional directory containing the selected SVG/PNG pairs.")
    args = parser.parse_args()
    ids = SELECTED_IDS if args.staging else selected_ids(args.archive)
    manifest = manifest_for(args.archive, ids)
    (ROOT / "vector_benchmark").mkdir(exist_ok=True)
    (ROOT / "vector_benchmark" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    montage_rows = []
    with tempfile.TemporaryDirectory(prefix="planparse_vector_benchmark_") as temp:
        temp_dir = Path(temp)
        extracted = {}
        if args.staging:
            extracted = {f"{sample}.{ext}": args.staging / f"{sample}.{ext}" for sample in ids for ext in ("svg", "png")}
        else:
            with tarfile.open(args.archive, "r:xz") as tar:
                wanted = {f"{sample}.{ext}" for sample in ids for ext in ("svg", "png")}
                for member in tar:
                    if member.name in wanted:
                        target = temp_dir / Path(member.name).name
                        target.write_bytes(tar.extractfile(member).read())
                        extracted[member.name] = target
        for sample in ids:
            svg = extracted[f"{sample}.svg"]
            pdf = temp_dir / f"{sample}.pdf"
            width, height, wall_paths = svg_to_pdf(svg, pdf)
            doc = fitz.open(pdf)
            assert len(doc[0].get_drawings()) > 0
            doc.close()
            vector_check = analyze_pdf(pdf, 0, mode="hybrid")
            assert vector_check.diagnostics["filtered_line_segment_count"] > 0
            vector = {}
            panels = [cv2.imread(str(extracted[f"{sample}.png"]), cv2.IMREAD_COLOR)]
            gt = mask_from_paths(wall_paths, panels[0].shape[:2], width, height)
            for mode in ("vector", "raster", "hybrid"):
                start = time.perf_counter()
                if mode == "raster":
                    doc = fitz.open(pdf); page = doc[0]; pix = page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), alpha=False); image = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n); image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR); doc.close(); result = raster_wall_detection(image)
                else:
                    result = analyze_pdf(pdf, 0, mode=mode)
                elapsed = (time.perf_counter() - start) * 1000
                metrics = score_prediction(pred_mask(result), gt, elapsed, sample, mode)
                results.append(metrics)
                if mode == "vector": vector["result"] = result
                panels.append(draw_wall_overlay(result.image, result.walls, 0.85))
            gt_view = panels[0].copy(); gt_view[gt] = (0, 0, 180); panels.append(gt_view)
            row = cv2.hconcat([resize_panel(panel) for panel in panels])
            montage_rows.append(row)
    all_rows = results
    summary = {"manifest": manifest, "modes": {mode: {"n_samples": 5, "mean": {key: float(np.mean([row[key] for row in all_rows if row["mode"] == mode])) for key in ("wall_iou", "pixel_precision", "pixel_recall", "pixel_f1", "f1_at_3px", "centerline_chamfer_px")}} for mode in ("vector", "raster", "hybrid")}, "per_sample": all_rows, "vector_preservation_verified": True}
    out = ROOT / "vector_benchmark"
    (out / "results.csv").write_text("sample_id,mode,wall_iou,pixel_precision,pixel_recall,pixel_f1,f1_at_3px,centerline_chamfer_px,processing_time_ms\n" + "\n".join(",".join(str(row[key]) for key in ("sample_id", "mode", "wall_iou", "pixel_precision", "pixel_recall", "pixel_f1", "f1_at_3px", "centerline_chamfer_px", "processing_time_ms")) for row in all_rows) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_image(ROOT / "assets" / "vector_benchmark_montage.png", cv2.vconcat(montage_rows))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

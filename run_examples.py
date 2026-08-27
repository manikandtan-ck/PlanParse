"""Run the qualitative real-PDF demo pipeline and save staged visualizations."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

from inspect_pdf import profile_pdf
from planparse.pdf import analyze_pdf
from planparse.visualization import draw_candidate_overlay, draw_vector_overlay, draw_wall_overlay, save_image


ROOT = Path(__file__).parent


def main() -> None:
    manifest = json.loads((ROOT / "examples/demo_manifest.json").read_text())
    for example in manifest["examples"]:
        source = ROOT / example["pdf"]
        if not source.exists():
            raise SystemExit(f"Missing {source}; run python download_examples.py")
        start = time.perf_counter()
        result = analyze_pdf(source, example["page"])
        elapsed_ms = (time.perf_counter() - start) * 1000
        out = ROOT / "outputs/examples" / example["id"]
        out.mkdir(parents=True, exist_ok=True)
        original = result.image
        save_image(out / "00_original.png", original)
        save_image(out / "01_raw_vectors.png", draw_vector_overlay(original, result.raw_vectors, (0, 0, 255), 1))
        save_image(out / "02_wall_pairs.png", draw_candidate_overlay(original, result.candidates, show_pairs=True, show_centerlines=True))
        save_image(out / "03_merged_candidates.png", draw_candidate_overlay(original, result.candidates, show_pairs=False, show_centerlines=True, merged_segments=result.merged_vectors))
        save_image(out / "04_hybrid_result.png", draw_wall_overlay(draw_vector_overlay(original, result.vectors, (100, 100, 100), 1), result.walls, 0.85))
        payload = {"example": example, "diagnostics": {**result.diagnostics, "processing_time_ms": elapsed_ms}, "walls": [wall.as_dict(i) for i, wall in enumerate(result.walls)], "weak_walls": [wall.as_dict(i) for i, wall in enumerate(result.weak_walls)]}
        (out / "geometry.json").write_text(json.dumps(payload, indent=2) + "\n")
        profile = profile_pdf(source)
        page_profile = profile["pages"][example["page"]]
        (ROOT / "outputs/vector_debug").mkdir(parents=True, exist_ok=True)
        (ROOT / "outputs/vector_debug" / f"{example['id']}_stats.json").write_text(json.dumps({"source": str(source), "page": example["page"], "profile": page_profile, "pipeline": result.diagnostics, "processing_time_ms": elapsed_ms}, indent=2) + "\n")
        print(example["id"], json.dumps({"raw_vectors": len(result.vectors), "wall_candidates": result.diagnostics["wall_candidate_count"], "deduplicated": result.diagnostics["deduplicated_candidate_count"], "accepted": len(result.walls), "weak": len(result.weak_walls), "processing_time_ms": round(elapsed_ms, 1)}))
        if example["id"].endswith("p09"):
            save_image(ROOT / "outputs/examples/failure_case.png", draw_candidate_overlay(original, result.candidates, show_pairs=False, show_centerlines=True, merged_segments=result.merged_vectors))


if __name__ == "__main__":
    main()

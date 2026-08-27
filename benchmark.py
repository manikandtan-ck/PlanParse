import json
import time
from pathlib import Path

import cv2

from planparse.evaluation import aggregate_results, save_results, score_prediction
from planparse.raster import raster_baseline
from planparse.visualization import benchmark_comparison, save_image


ROOT = Path(__file__).parent


def main() -> None:
    manifest = json.loads((ROOT / "benchmark/manifest.json").read_text())
    samples = manifest.get("samples", [])
    if not samples:
        raise SystemExit("No benchmark samples found. Run `python download_benchmark.py` first.")
    rows = []
    for sample in samples:
        image_path = ROOT / sample["local_image"]
        gt_path = ROOT / sample["local_gt_mask"]
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if image is None or gt is None:
            raise SystemExit(f"Missing image or mask for {sample['sample_id']}")
        assert gt.shape == image.shape[:2], (sample["sample_id"], image.shape, gt.shape)
        start = time.perf_counter()
        result = raster_baseline(image)
        assert result.wall_mask.shape == gt.shape, (sample["sample_id"], result.wall_mask.shape, gt.shape)
        elapsed = (time.perf_counter() - start) * 1000
        row = score_prediction(result.wall_mask, gt > 0, elapsed, sample["sample_id"], "raster")
        rows.append(row)
        save_image(ROOT / "outputs/benchmark_visualizations" / f"{sample['sample_id']}.png", benchmark_comparison(image, gt > 0, result.wall_mask))
    results = aggregate_results(rows)
    save_results(results, ROOT / "outputs/benchmark_results.json", ROOT / "outputs/benchmark_results.csv")
    print("PlanParse FloorPlanCAD Microbenchmark")
    print("=" * 38)
    print(f"{'Sample':<18} {'IoU':>7} {'F1@3px':>8} {'Chamfer(px)':>13}")
    print("-" * 50)
    for row in rows:
        print(f"{row['sample_id']:<18} {row['wall_iou']:>7.3f} {row['f1_at_3px']:>8.3f} {row['centerline_chamfer_px']:>13.2f}")
    mean = results["mean"]
    print("-" * 50)
    print(f"{'MEAN':<18} {mean['wall_iou']:>7.3f} {mean['f1_at_3px']:>8.3f} {mean['centerline_chamfer_px']:>13.2f}")
    print(f"\nSaved outputs/benchmark_results.json and outputs/benchmark_results.csv")


if __name__ == "__main__":
    main()

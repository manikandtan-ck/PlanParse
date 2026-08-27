"""Download the fixed five-sample FloorPlanCAD benchmark."""

from __future__ import annotations

import json
import base64
import io
import shutil
import urllib.request
import zlib
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).parent


def _wall_mask(sample) -> np.ndarray:
    height, width = int(sample.metadata.height), int(sample.metadata.width)
    full = np.zeros((height, width), np.uint8)
    for detection in sample.ground_truth.detections:
        if detection.label != "wall" or detection.mask is None:
            continue
        x, y, w, h = detection.bounding_box
        x0, y0 = round(x * width), round(y * height)
        x1, y1 = min(width, round((x + w) * width)), min(height, round((y + h) * height))
        mask = np.asarray(detection.mask)
        if mask.size == 0 or x1 <= x0 or y1 <= y0:
            continue
        mask = cv2.resize(mask.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        full[y0:y1, x0:x1] = np.maximum(full[y0:y1, x0:x1], mask)
    return full


def _wall_mask_dict(sample: dict, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape[:2]
    full = np.zeros((height, width), np.uint8)
    for detection in sample.get("ground_truth", {}).get("detections", []):
        if detection.get("label") != "wall" or not detection.get("mask"):
            continue
        x, y, w, h = detection["bounding_box"]
        x0, y0 = round(x * width), round(y * height)
        x1, y1 = min(width, round((x + w) * width)), min(height, round((y + h) * height))
        encoded = detection["mask"]["$binary"]["base64"]
        mask = np.lib.format.read_array(io.BytesIO(zlib.decompress(base64.b64decode(encoded))), allow_pickle=False)
        if x1 > x0 and y1 > y0:
            mask = cv2.resize(mask.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
            full[y0:y1, x0:x1] = np.maximum(full[y0:y1, x0:x1], mask)
    return full


def _mask_shape_dict(detection: dict) -> list[int] | None:
    encoded = detection.get("mask", {}).get("$binary", {}).get("base64")
    if not encoded:
        return None
    mask = np.lib.format.read_array(io.BytesIO(zlib.decompress(base64.b64decode(encoded))), allow_pickle=False)
    return list(mask.shape)


def _write_sample(index: int, sample_id: str, image: np.ndarray, mask: np.ndarray, source_filepath: str, wall_count: int, annotations: list[dict] | None = None) -> dict:
    out_dir = ROOT / "benchmark/samples" / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path, mask_path = out_dir / "image.png", out_dir / "gt_wall_mask.png"
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)
    metadata = {"sample_id": sample_id, "source_filepath": source_filepath, "width": image.shape[1], "height": image.shape[0], "wall_annotations": wall_count, "annotations": annotations or []}
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"index": index, "sample_id": sample_id, "source_filepath": source_filepath, "local_image": str(image_path.relative_to(ROOT)), "local_gt_mask": str(mask_path.relative_to(ROOT))}


def _fallback_from_public_metadata() -> list[dict]:
    """Small dependency-free fallback for environments without FiftyOne."""
    metadata_url = "https://huggingface.co/datasets/Voxel51/FloorPlanCAD/resolve/main/samples.json"
    with urllib.request.urlopen(metadata_url) as response:
        data = json.load(response)
    samples = sorted(data["samples"], key=lambda sample: sample["filepath"])[:5]
    rows = []
    for index, sample in enumerate(samples):
        filepath = sample["filepath"]
        image_url = f"https://huggingface.co/datasets/Voxel51/FloorPlanCAD/resolve/main/{filepath}"
        with urllib.request.urlopen(image_url) as response:
            image = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        sample_id = sample["_id"]["$oid"]
        annotations = [{"label": d.get("label"), "bounding_box": d.get("bounding_box"), "mask_shape": _mask_shape_dict(d)} for d in sample.get("ground_truth", {}).get("detections", [])]
        rows.append(_write_sample(index, sample_id, image, _wall_mask_dict(sample, image.shape), image_url, sum(d.get("label") == "wall" for d in sample.get("ground_truth", {}).get("detections", [])), annotations))
    return rows


def main() -> None:
    try:
        from fiftyone.utils.huggingface import load_from_hub
    except ImportError:
        manifest_samples = _fallback_from_public_metadata()
        selection_method = "first five samples after stable filepath sort from the public mirror metadata fallback"
    else:
        dataset = load_from_hub("Voxel51/FloorPlanCAD", max_samples=5)
        samples = sorted(list(dataset), key=lambda s: (str(s.id), str(s.filepath)))[:5]
        manifest_samples = []
        for index, sample in enumerate(samples):
            sample_id = str(sample.id)
            out_dir = ROOT / "benchmark/samples" / sample_id
            out_dir.mkdir(parents=True, exist_ok=True)
            image_path = out_dir / "image.png"
            mask_path = out_dir / "gt_wall_mask.png"
            shutil.copyfile(sample.filepath, image_path)
            cv2.imwrite(str(mask_path), _wall_mask(sample))
            metadata = {"sample_id": sample_id, "source_filepath": str(sample.filepath), "width": sample.metadata.width, "height": sample.metadata.height, "wall_annotations": sum(d.label == "wall" for d in sample.ground_truth.detections), "annotations": [{"label": d.label, "bounding_box": list(d.bounding_box), "mask_shape": list(np.asarray(d.mask).shape) if d.mask is not None else None} for d in sample.ground_truth.detections]}
            (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            manifest_samples.append({"index": index, "sample_id": sample_id, "source_filepath": str(sample.filepath), "local_image": str(image_path.relative_to(ROOT)), "local_gt_mask": str(mask_path.relative_to(ROOT))})
        selection_method = "first five samples returned by fixed FiftyOne loader configuration, sorted by stable sample id"
    manifest = {"dataset": "Voxel51/FloorPlanCAD", "selection_method": selection_method, "samples": manifest_samples}
    (ROOT / "benchmark/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Downloaded {len(manifest_samples)} samples and wrote benchmark/manifest.json")


if __name__ == "__main__":
    main()

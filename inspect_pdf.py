"""Profile PDF pages for native vector content."""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf


def profile_pdf(path: str | Path) -> dict:
    doc = pymupdf.open(str(path))
    pages = []
    for index, page in enumerate(doc):
        drawings = page.get_drawings()
        counts = {"line": 0, "curve": 0, "rect": 0, "quad": 0, "other": 0, "filled_paths": 0, "stroke_only_paths": 0}
        for drawing in drawings:
            if drawing.get("fill") is not None:
                counts["filled_paths"] += 1
            if drawing.get("color") is not None and drawing.get("fill") is None:
                counts["stroke_only_paths"] += 1
            for item in drawing.get("items", []):
                kind = item[0] if item else "other"
                if kind == "l":
                    counts["line"] += 1
                elif kind == "c":
                    counts["curve"] += 1
                elif kind == "re":
                    counts["rect"] += 1
                elif kind == "qu":
                    counts["quad"] += 1
                else:
                    counts["other"] += 1
        line_like = counts["line"] + counts["curve"] + counts["rect"] + counts["quad"]
        image_count = len(page.get_images(full=True))
        page_area = max(1.0, page.rect.width * page.rect.height)
        image_rects = []
        for image in page.get_images(full=True):
            image_rects.extend(page.get_image_rects(image[0]))
        image_coverage = min(1.0, sum(max(0.0, r.width) * max(0.0, r.height) for r in image_rects) / page_area)
        if image_coverage >= 0.35:
            classification = "HYBRID"
        elif line_like == 0 and image_count:
            classification = "RASTER_DOMINANT"
        elif line_like < 25 and image_count:
            classification = "HYBRID"
        elif line_like:
            classification = "VECTOR_RICH"
        else:
            classification = "EMPTY_OR_UNSUPPORTED"
        pages.append({"page_index": index, "width_pt": page.rect.width, "height_pt": page.rect.height, "vector_path_count": len(drawings), "line_segment_count": counts["line"], "image_count": image_count, "image_coverage": round(image_coverage, 3), "text_block_count": len(page.get_text("blocks")), "counts": counts, "classification": classification, "line_like_primitives": line_like})
    doc.close()
    return {"file": str(path), "page_count": len(pages), "pages": pages}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python inspect_pdf.py examples/pdfs/rinker_a102.pdf")
    report = profile_pdf(sys.argv[1])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

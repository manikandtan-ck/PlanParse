import cv2
import numpy as np

from .models import LineSegment, RasterResult


def _ink_mask(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 4:
        # FloorPlanCAD uses transparent RGBA line art; RGB alone loses black wall strokes.
        return image[:, :, 3] > 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if float(gray.mean()) < 128:
        return gray > 0
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return ink > 0


def raster_stages(image: np.ndarray, line_kernel: int = 25, closing_kernel: int = 3, min_component_area: int = 1500) -> dict[str, np.ndarray]:
    ink = (_ink_mask(image).astype(np.uint8) * 255)
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (line_kernel, 1)))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_kernel)))
    structural = cv2.morphologyEx(horizontal | vertical, cv2.MORPH_CLOSE, np.ones((closing_kernel, closing_kernel), np.uint8))
    filtered = np.zeros_like(structural)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(structural)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= min_component_area:
            filtered[labels == index] = 255
    return {
        "grayscale": cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy(),
        "binary": ink,
        "horizontal_response": horizontal,
        "vertical_response": vertical,
        "structural": structural,
        "wall_mask": filtered,
    }


def wall_mask_from_image(image: np.ndarray, line_kernel: int = 25, closing_kernel: int = 3, min_component_area: int = 1500) -> np.ndarray:
    return raster_stages(image, line_kernel, closing_kernel, min_component_area)["wall_mask"] > 0


def raster_baseline(image: np.ndarray) -> RasterResult:
    mask = wall_mask_from_image(image)
    edges = cv2.Canny((mask.astype(np.uint8) * 255), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(12, min(image.shape[:2]) // 30), minLineLength=max(20, min(image.shape[:2]) // 20), maxLineGap=12)
    centerlines = []
    for row in np.asarray(lines).reshape(-1, 4) if lines is not None else []:
        x1, y1, x2, y2 = map(float, row)
        centerlines.append(LineSegment((x1, y1), (x2, y2)))
    return RasterResult(mask, centerlines, {"ink_pixel_count": int(_ink_mask(image).sum()), "wall_pixel_count": int(mask.sum()), "hough_line_count": len(centerlines)})

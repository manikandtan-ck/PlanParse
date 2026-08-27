# FloorPlanCAD benchmark

The benchmark uses five fixed samples from the Voxel51 FloorPlanCAD mirror. Run `python download_benchmark.py` to reconstruct the local images and masks, then run `python benchmark.py`.

Each sample contains wall detections with bbox-relative masks. Wall masks are reconstructed at the original image resolution using nearest-neighbor resizing and union. RGBA alpha compositing is preserved for the raster baseline.

Metrics are wall IoU, pixel precision/recall/F1, F1 at 3 px tolerance, and symmetric centerline Chamfer distance.

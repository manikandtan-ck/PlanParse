from planparse.pdf import create_synthetic_pdf
from planparse.raster_fallback import render_raster_page


def test_rendered_pixel_coordinate_mapping():
    image, metadata = render_raster_page(create_synthetic_pdf(), 0)
    assert image.shape[1] == round(metadata["width_pt"] * metadata["scale_px_per_pt"])
    assert image.shape[0] == round(metadata["height_pt"] * metadata["scale_px_per_pt"])
    assert metadata["width_px"] == image.shape[1]
    assert metadata["height_px"] == image.shape[0]

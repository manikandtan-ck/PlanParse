from planparse.pdf import pdf_to_pixel


def test_pdf_to_pixel_transform():
    assert pdf_to_pixel((72.0, 36.0), 150 / 72) == (150.0, 75.0)


from streamlit.testing.v1 import AppTest


APP = "app.py"


def test_app_loads_with_demo_controls():
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert app.selectbox[0].value == 1
    assert app.selectbox[0].options == ["1"]
    assert app.selectbox[1].value == "Auto (recommended)"


def test_detection_mode_options_and_tooltips_are_present():
    app = AppTest.from_file(APP).run()
    mode = app.selectbox[1]
    assert mode.options == ["Auto (recommended)", "Hybrid", "Vector", "Raster"]
    assert "drawing lines" in mode.help.lower()
    assert app.checkbox[0].help


def test_raster_fallback_returns_raster_source():
    from planparse.pdf import create_synthetic_pdf
    from planparse.raster_fallback import analyze_experimental

    result, mode = analyze_experimental(create_synthetic_pdf(), 0, "Raster")
    assert mode == "Raster"
    assert all(wall.source == "raster" for wall in result.walls)


def test_vector_and_hybrid_sources_are_distinct():
    from planparse.pdf import create_synthetic_pdf
    from planparse.raster_fallback import analyze_experimental

    vector, vector_mode = analyze_experimental(create_synthetic_pdf(), 0, "Vector")
    hybrid, hybrid_mode = analyze_experimental(create_synthetic_pdf(), 0, "Hybrid")
    assert vector_mode == "Vector"
    assert hybrid_mode == "Hybrid"
    assert all(wall.source == "vector" for wall in vector.walls)
    assert all(wall.source == "hybrid" for wall in hybrid.walls)

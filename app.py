"""Streamlit interface for PlanParse."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import streamlit as st

from planparse.pdf import create_synthetic_pdf, pdf_page_count
from planparse.geometry import line_to_points
from planparse.visualization import GREEN, LIGHT_GRAY, YELLOW, draw_candidate_overlay, draw_vector_overlay, draw_wall_overlay
from planparse.raster_fallback import analyze_experimental, draw_raster_debug


ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

st.set_page_config(page_title="PlanParse", page_icon="🧱", layout="wide")


def load_pdf(label: str, data: bytes, page: int = 0) -> None:
    st.session_state["pdf_bytes"] = data
    st.session_state["example_name"] = label
    st.session_state["example_page"] = page


def diagnostics_payload(result, elapsed_ms: float) -> dict:
    return {
        "page": {key: result.diagnostics[key] for key in ("page_index", "width_pt", "height_pt", "width_px", "height_px")},
        "diagnostics": {key: value for key, value in result.diagnostics.items() if key != "debug_stages"} | {"processing_time_ms": round(elapsed_ms, 1)},
        "walls": [wall.as_dict(index) for index, wall in enumerate(result.walls)],
    }


def source_key(data: bytes, label: str) -> str:
    return f"{label}:{hashlib.sha256(data).hexdigest()}"


def fit_display(image, max_width: int = 760, max_height: int = 500):
    height, width = image.shape[:2]
    scale = min(1.0, max_width / max(1, width), max_height / max(1, height))
    if scale == 1.0:
        return image
    return cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)


def draw_colored_candidate_overlay(image, candidates, color):
    out = image.copy()
    for wall in candidates:
        if wall.line_a and wall.line_b:
            cv2.line(out, *line_to_points(wall.line_a), color, 1, cv2.LINE_AA)
            cv2.line(out, *line_to_points(wall.line_b), color, 1, cv2.LINE_AA)
        cv2.line(out, *line_to_points(wall.centerline), color, max(2, round(wall.thickness_px / 2)), cv2.LINE_AA)
    return out


def diagnostics_caption(result, elapsed_ms):
    diagnostics = result.diagnostics
    mode = diagnostics.get("document_mode", "unknown").upper()
    if mode == "RASTER-ONLY":
        return f"{mode} · {diagnostics.get('raster_raw_line_count', 0):,} raster lines · {diagnostics.get('wall_candidate_count', 0):,} candidates · {len(result.walls):,} accepted walls · {elapsed_ms:.0f} ms"
    return f"{mode} · {diagnostics.get('vector_path_count', 0):,} PDF paths · {diagnostics.get('wall_candidate_count', 0):,} candidates · {len(result.walls):,} accepted walls · {elapsed_ms:.0f} ms"


if "pdf_bytes" not in st.session_state:
    load_pdf("Synthetic vector example", create_synthetic_pdf())
if "source_key" not in st.session_state:
    st.session_state["source_key"] = source_key(st.session_state["pdf_bytes"], st.session_state["example_name"])
if "page_number" not in st.session_state:
    st.session_state["page_number"] = 1
if "analysis_key" not in st.session_state:
    st.session_state["analysis_key"] = None
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None


demo, benchmark, how = st.tabs(["Demo", "Benchmark", "How it works"])

with demo:
    st.title("PlanParse")
    st.subheader("Find likely walls in architectural PDF drawings.")
    st.caption("PlanParse reads the drawing and highlights wall-like structures so you can inspect and export the result.")

    controls, viewer = st.columns([1, 2], gap="medium")
    with controls:
        st.caption("Input")
        source_mode = st.radio("Source", ["Built-in example", "Upload PDF"], horizontal=True, label_visibility="collapsed", key="source_mode")
        uploaded = None
        if source_mode == "Upload PDF":
            uploaded = st.file_uploader("PDF file", type=["pdf"], help="Upload a PDF drawing to inspect one page at a time.", label_visibility="collapsed")
            if uploaded is not None and uploaded.size > MAX_UPLOAD_BYTES:
                st.error("This demo accepts PDFs up to 20 MB.")
                uploaded = None

        if source_mode == "Built-in example":
            current_bytes = create_synthetic_pdf()
            current_label = "Synthetic vector example"
        elif uploaded is not None:
            current_bytes = uploaded.getvalue()
            current_label = uploaded.name
        else:
            current_bytes = None
            current_label = ""

        current_key = source_key(current_bytes, current_label) if current_bytes is not None else "no-upload"
        if current_key != st.session_state.get("source_key"):
            st.session_state["source_key"] = current_key
            st.session_state["page_number"] = 1
            st.session_state["analysis_key"] = None
            st.session_state["analysis_result"] = None
            if current_bytes is not None:
                load_pdf(current_label, current_bytes, 0)

        page_count = 0
        if current_bytes is not None:
            try:
                page_count = pdf_page_count(current_bytes)
            except Exception:
                st.error("This file could not be opened as a PDF.")

        st.caption("Page")
        if page_count:
            if st.session_state["page_number"] > page_count:
                st.session_state["page_number"] = 1
            selected_page_number = st.selectbox("Page", options=list(range(1, page_count + 1)), key="page_number", label_visibility="collapsed", help="Choose the page to process. Page numbers start at 1.")
            page_index = selected_page_number - 1
        else:
            selected_page_number = None
            page_index = 0

        st.caption("View")
        opacity = st.slider("Overlay opacity", 0.2, 1.0, 0.75, help="Adjusts how strongly detected walls are shown over the drawing.")

        detection_mode = st.selectbox(
            "Detection mode",
            ["Auto (recommended)", "Hybrid", "Vector", "Raster"],
            help="Auto: Uses Hybrid detection when the PDF contains drawing lines, otherwise uses Raster detection. Hybrid: Uses PDF drawing lines together with the rendered page image. Vector: Uses drawing lines stored in the PDF only. Raster: Uses the rendered page image only.",
        )

        with st.expander("Developer options"):
            show_raw = st.checkbox("Show PDF drawing lines", help="Displays the lines read directly from the PDF before wall detection; display only.")
            show_candidates = st.checkbox("Show wall candidates", help="Displays line pairs that the detector considers possible walls; display only.")
            show_weak = st.checkbox("Show weak candidates", help="Displays lower-confidence wall candidates; display only.")
            show_diagnostics = st.checkbox("Diagnostics", help="Shows intermediate counts and processing details useful for debugging; display only.")
            show_raster_mask = st.checkbox("Show image mask", help="Displays the foreground mask used by raster fallback; display only.")
            show_raster_response = st.checkbox("Show horizontal/vertical response", help="Displays directional morphology responses used to find raster lines; display only.")
            show_raster_lines = st.checkbox("Show raster line segments", help="Displays raw horizontal and vertical raster line segments; display only.")
            show_raster_pairs = st.checkbox("Show raster wall candidates", help="Displays raster-derived parallel wall pairs before acceptance; display only.")

        st.caption("Export")
        analyze_clicked = st.button("Analyze page", use_container_width=True, disabled=current_bytes is None)
        analysis_key = (current_key, page_index, detection_mode)
        if current_bytes is not None and (analyze_clicked or st.session_state.get("analysis_key") != analysis_key):
            try:
                start = time.perf_counter()
                result, mode_used = analyze_experimental(current_bytes, page_index, detection_mode)
                elapsed_ms = (time.perf_counter() - start) * 1000
                st.session_state["analysis_result"] = (result, elapsed_ms)
                st.session_state["analysis_mode"] = mode_used
                st.session_state["analysis_key"] = analysis_key
            except Exception as exc:
                st.session_state["analysis_result"] = None
                st.error(f"Could not analyze this page: {exc}")

        stored = st.session_state.get("analysis_result")
        if stored is not None and st.session_state.get("analysis_key") == (current_key, page_index, detection_mode):
            result, elapsed_ms = stored
            st.download_button("Download JSON", json.dumps(diagnostics_payload(result, elapsed_ms), indent=2), file_name="planparse_geometry.json", mime="application/json", use_container_width=True)

    with viewer:
        stored = st.session_state.get("analysis_result")
        if current_bytes is None:
            st.info("Upload a PDF to preview wall detection.")
        elif stored is None or st.session_state.get("analysis_key") != (current_key, page_index, detection_mode):
            st.info("Choose a page and click Analyze page to preview wall detection.")
        else:
            result, elapsed_ms = stored
            overlay = result.image.copy()
            mode_used = st.session_state.get("analysis_mode", "Hybrid")
            if show_raw and mode_used != "Raster":
                overlay = draw_vector_overlay(overlay, result.raw_vectors, LIGHT_GRAY, 1)
            if show_candidates and mode_used != "Raster":
                overlay = draw_colored_candidate_overlay(overlay, result.candidates, (0, 165, 255))
            if show_weak and mode_used != "Raster":
                overlay = draw_candidate_overlay(overlay, result.weak_walls, show_pairs=True, show_centerlines=True)
            overlay = draw_wall_overlay(overlay, result.walls, opacity)
            original_display = fit_display(cv2.cvtColor(result.image, cv2.COLOR_BGR2RGB))
            result_display = fit_display(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), max_width=original_display.shape[1], max_height=original_display.shape[0])
            left, right = st.columns(2, gap="small")
            with left:
                st.caption("Original")
                st.image(original_display, use_container_width=True)
            with right:
                st.caption("Detected walls")
                st.image(result_display, use_container_width=True)
            status = "Image-based fallback — works best on clean floor plans." if mode_used == "Raster" else ("Uses native PDF drawing lines with raster support." if mode_used == "Hybrid" else "Uses native PDF drawing lines only.")
            st.caption(f"Mode used: {mode_used} · detected walls: {len(result.walls)} · {status}")
            if not result.walls:
                st.info("No wall-like structures detected.")
            if show_diagnostics:
                st.caption(diagnostics_caption(result, elapsed_ms))
            if mode_used == "Raster" and any((show_raster_mask, show_raster_response, show_raster_lines, show_raster_pairs)):
                debug = draw_raster_debug(result)
                if show_raster_mask:
                    st.image(debug["foreground_mask"], caption="Foreground mask", clamp=True, use_container_width=True)
                if show_raster_response:
                    response_left, response_right = st.columns(2, gap="small")
                    with response_left:
                        st.image(debug["horizontal_response"], caption="Horizontal response", clamp=True, use_container_width=True)
                    with response_right:
                        st.image(debug["vertical_response"], caption="Vertical response", clamp=True, use_container_width=True)
                if show_raster_lines:
                    st.image(cv2.cvtColor(debug["raw_raster_lines"], cv2.COLOR_BGR2RGB), caption="Raw raster line segments", use_container_width=True)
                if show_raster_pairs:
                    st.image(cv2.cvtColor(debug["raster_wall_pairs"], cv2.COLOR_BGR2RGB), caption="Raster wall candidates", use_container_width=True)
            legend_items = [("Detected wall", GREEN)]
            if show_candidates and mode_used != "Raster":
                legend_items.append(("Wall candidate", (0, 165, 255)))
            if show_weak and mode_used != "Raster":
                legend_items.append(("Weak candidate", YELLOW))
            if show_raw and mode_used != "Raster":
                legend_items.append(("PDF drawing line", LIGHT_GRAY))
            if mode_used == "Raster" and show_raster_lines:
                legend_items.append(("Raster line segment", (0, 165, 255)))
            if mode_used == "Raster" and show_raster_pairs:
                legend_items.append(("Raster wall candidate", (0, 165, 255)))
            legend_html = "".join(
                f'<span style="white-space:nowrap"><span style="color:rgb({color[2]},{color[1]},{color[0]})">■</span> {label}</span>'
                for label, color in legend_items
            )
            st.markdown(
                f'<div style="display:flex;flex-wrap:wrap;gap:14px;align-items:center;font-size:0.82rem">{legend_html}</div>',
                unsafe_allow_html=True,
            )

with benchmark:
    st.title("FloorPlanCAD microbenchmark")
    st.write("A fixed five-sample raster semantic sanity check. It is separate from the PDF-vector demo because semantic wall masks and inferred PDF wall centerlines are different representations.")
    st.metric("Winning baseline · F1@3px", "0.477")
    st.table([
        {"Method": "Classical structural CV", "IoU": 0.182, "F1@3px": 0.477, "Chamfer ↓": 99.28},
        {"Method": "Pretrained CubiCasa", "IoU": 0.113, "F1@3px": 0.391, "Chamfer ↓": 88.83},
        {"Method": "CubiCasa domain fine-tune", "IoU": 0.117, "F1@3px": 0.408, "Chamfer ↓": 82.20},
        {"Method": "Binary-512 adaptation", "IoU": 0.000, "F1@3px": 0.000, "Chamfer ↓": 1000.00},
    ])
    montage = ROOT / "assets/benchmark_montage.png"
    if montage.exists():
        st.image(str(montage), caption="Five FloorPlanCAD samples with green PlanParse wall predictions and per-sample F1@3px")
    with st.expander("What the experiment means"):
        st.write("The pretrained semantic model showed domain shift from residential floor-plan imagery to sparse CAD-style drawings. Small same-domain fine-tuning improved transfer modestly but did not beat the simpler structural baseline, so semantic learning remains a documented experiment rather than an active dependency.")

with how:
    st.title("How it works")
    st.image(str(ROOT / "assets/pipeline.svg"))
    st.markdown("""
1. Inspect document composition and embedded-image coverage.
2. Extract native vector geometry with PyMuPDF.
3. Normalize line primitives and merge fragmented collinear segments.
4. Generate approximately parallel wall candidates.
5. Score candidates with vector geometry and raster support.
6. Suppress duplicates and export grounded wall geometry as JSON.
""")
    st.subheader("Boundaries")
    st.write("Raster-only drawings have reduced geometric precision; dimensions and diagram geometry can resemble walls; filled wall polygons, curved walls, and page-level layout isolation remain heuristic. The system does not infer doors, windows, rooms, OCR, BIM, or quantities.")

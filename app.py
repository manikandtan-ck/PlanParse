"""Streamlit interface for PlanParse."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import streamlit as st

from planparse.pdf import analyze_pdf, create_synthetic_pdf, pdf_page_count
from planparse.visualization import LIGHT_GRAY, YELLOW, draw_candidate_overlay, draw_vector_overlay, draw_wall_overlay


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
        "diagnostics": {**result.diagnostics, "processing_time_ms": round(elapsed_ms, 1)},
        "walls": [wall.as_dict(index) for index, wall in enumerate(result.walls)],
    }


if "pdf_bytes" not in st.session_state:
    load_pdf("Synthetic vector example", create_synthetic_pdf())

demo, benchmark, how = st.tabs(["Demo", "Benchmark", "How it works"])

with demo:
    st.title("PlanParse")
    st.subheader("Hybrid raster/vector geometry extraction for construction drawings.")
    st.write("Native PDF geometry provides precise primitives while lightweight raster processing provides an independent view of visible structure.")

    examples = [("Synthetic vector example", None, 0)]
    public_manifest = ROOT / "examples/public_examples.json"
    if public_manifest.exists():
        for source in json.loads(public_manifest.read_text())["sources"]:
            path = ROOT / "examples/pdfs" / source["local_filename"]
            if path.exists():
                pages = source.get("pages_tested") or [0]
                examples.append((source["name"], path, pages[0]))
    labels = [item[0] for item in examples]
    current = st.session_state.get("example_name", labels[0])
    selected_label = st.selectbox("Try a built-in example", labels, index=labels.index(current) if current in labels else 0)
    if st.button("Load example"):
        label, path, page = examples[labels.index(selected_label)]
        load_pdf(label, create_synthetic_pdf() if path is None else path.read_bytes(), page)
        if hasattr(st, "rerun"):
            st.rerun()
        st.experimental_rerun()

    uploaded = st.file_uploader("Or upload a PDF", type=["pdf"], help="One page is processed at a time; maximum size is 20 MB.")
    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_BYTES:
            st.error("This demo accepts PDFs up to 20 MB.")
        else:
            load_pdf(uploaded.name, uploaded.getvalue())

    pdf_bytes = st.session_state["pdf_bytes"]
    try:
        page_count = pdf_page_count(pdf_bytes)
    except Exception:
        st.error("This file could not be opened as a PDF. Try an unencrypted, non-corrupt document.")
        st.stop()
    st.caption(f"Source: {st.session_state.get('example_name', 'PDF')} · {page_count} page(s) detected")
    page_idx = st.number_input("Page", min_value=0, max_value=max(0, page_count - 1), value=min(int(st.session_state.get("example_page", 0)), max(0, page_count - 1)), step=1)
    st.caption("Only the selected page is processed. Pages render at approximately 150 DPI and are capped at 2,500 px.")
    opacity = st.slider("Overlay opacity", 0.2, 1.0, 0.75)
    controls = st.columns(3)
    show_raw = controls[0].checkbox("Raw PDF vectors")
    show_candidates = controls[1].checkbox("Wall candidate pairs")
    show_weak = controls[2].checkbox("Weak candidates")

    if st.button("Analyze selected page"):
        try:
            start = time.perf_counter()
            result = analyze_pdf(pdf_bytes, int(page_idx))
            elapsed_ms = (time.perf_counter() - start) * 1000
        except Exception as exc:
            st.error(f"Could not analyze this page: {exc}")
            st.stop()
        if result.diagnostics["document_mode"] == "raster-only":
            st.warning("This page appears raster-dominant; native vector extraction is limited.")
        overlay = result.image.copy()
        if show_raw:
            overlay = draw_vector_overlay(overlay, result.raw_vectors, LIGHT_GRAY, 1)
        if show_candidates:
            overlay = draw_candidate_overlay(overlay, result.candidates, show_pairs=True, show_centerlines=True)
        if show_weak:
            overlay = draw_candidate_overlay(overlay, result.weak_walls, show_pairs=True, show_centerlines=True)
        overlay = draw_wall_overlay(overlay, result.walls, opacity)
        left, right = st.columns(2)
        with left:
            st.subheader("Original")
            st.image(cv2.cvtColor(result.image, cv2.COLOR_BGR2RGB))
        with right:
            st.subheader("PlanParse result")
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        st.subheader("Page diagnostics")
        cols = st.columns(5)
        cols[0].metric("Document mode", result.diagnostics["document_mode"].upper())
        cols[1].metric("Vector paths", f"{result.diagnostics['vector_path_count']:,}")
        cols[2].metric("Line segments", f"{result.diagnostics['filtered_line_segment_count']:,}")
        cols[3].metric("Wall candidates", f"{result.diagnostics['wall_candidate_count']:,}")
        cols[4].metric("Accepted walls", f"{len(result.walls):,}")
        st.caption(f"Processing time: {elapsed_ms:.0f} ms · image coverage estimate: {result.diagnostics['image_coverage']:.0%}")
        st.download_button("Download structured geometry JSON", json.dumps(diagnostics_payload(result, elapsed_ms), indent=2), file_name="planparse_geometry.json", mime="application/json")
        st.markdown(
            '<div style="font-size:0.9rem"><b>Overlay</b>&nbsp;&nbsp;'
            '<span style="color:#00a800">■</span> Detected wall&nbsp;&nbsp;'
            '<span style="color:#d0aa00">■</span> Candidate wall&nbsp;&nbsp;'
            '<span style="color:#888888">■</span> Raw PDF vector</div>',
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

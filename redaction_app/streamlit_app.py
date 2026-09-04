from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw

from aws_transfer import AWSConfig, log_review_metric_to_ddb, upload_approved_files_to_s3
from site_catalog import load_site_ids
from utils import (
    collect_approved_files,
    compile_redacted_document,
    get_page_redaction_boxes,
    load_tracker,
    override_page_redactions,
    process_reports_local,
    save_tracker,
    set_review_metadata,
    set_review_decision,
)

def _patch_streamlit_image_to_url() -> None:
    """
    streamlit-drawable-canvas expects `streamlit.elements.image.image_to_url`.
    Newer Streamlit moved this helper; add a shim when possible.
    """
    try:
        from streamlit.elements import image as st_image_module
    except Exception:
        return

    if hasattr(st_image_module, "image_to_url"):
        return

    try:
        from streamlit.elements.lib.image_utils import image_to_url as new_image_to_url
    except Exception:
        return

    try:
        setattr(st_image_module, "image_to_url", new_image_to_url)
    except Exception:
        return


def _ensure_drawable_canvas():
    """
    Quietly attempts to import/install streamlit-drawable-canvas for rectangle drawing.
    Returns st_canvas callable when available, else None.
    """
    _patch_streamlit_image_to_url()
    try:
        module = importlib.import_module("streamlit_drawable_canvas")
        return getattr(module, "st_canvas", None)
    except Exception:
        pass

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "streamlit-drawable-canvas",
                "--quiet",
                "--disable-pip-version-check",
                "--no-input",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=180,
        )
        _patch_streamlit_image_to_url()
        module = importlib.import_module("streamlit_drawable_canvas")
        return getattr(module, "st_canvas", None)
    except Exception:
        return None


st_canvas = _ensure_drawable_canvas()


st.set_page_config(layout="wide")
st.title("Safe Harbor PHI Redaction Workflow")

if "tracker_path" not in st.session_state:
    st.session_state.tracker_path = ""
if "df" not in st.session_state:
    st.session_state.df = None
if "jump_to_tab" not in st.session_state:
    st.session_state.jump_to_tab = ""
if "post_load_message" not in st.session_state:
    st.session_state.post_load_message = ""
if "drag_rect_supported" not in st.session_state:
    st.session_state.drag_rect_supported = True

SITE_ID_OPTIONS = load_site_ids()
if not SITE_ID_OPTIONS:
    SITE_ID_OPTIONS = ["BCH"]
if "selected_site_id" not in st.session_state:
    st.session_state.selected_site_id = SITE_ID_OPTIONS[0]


def _load_df() -> pd.DataFrame:
    if st.session_state.df is None:
        raise RuntimeError("No tracker loaded")
    return st.session_state.df


def _apply_site_id_to_tracker(site_id: str) -> None:
    site = str(site_id or "").strip().upper()
    tracker_path = str(st.session_state.get("tracker_path", "") or "").strip()
    if not site or not tracker_path:
        return
    try:
        df = load_tracker(tracker_path, create_if_missing=True)
    except Exception:
        return
    if "site_id" not in df.columns or df.empty:
        st.session_state.df = df
        return
    needs_update = df["site_id"].fillna("").astype(str).str.strip().str.upper().ne(site)
    if bool(needs_update.any()):
        df.loc[:, "site_id"] = site
        save_tracker(df, tracker_path)
    st.session_state.df = df


def _render_pdf_preview(pdf_path: Path, max_pages: int = 6, dpi: int = 130) -> None:
    if not pdf_path.exists():
        st.warning(f"File not found: {pdf_path}")
        return
    try:
        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
        shown = min(page_count, max_pages)
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        for page_idx in range(shown):
            page = doc.load_page(page_idx)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            _show_image_compat(image, caption=f"Page {page_idx + 1} of {page_count}")
        doc.close()
        if page_count > shown:
            st.caption(f"Showing first {shown} pages. Use download for full PDF.")
    except Exception as exc:
        st.warning(f"Could not render PDF preview: {exc}")


def _show_image_compat(image_obj: object, caption: str | None = None) -> None:
    try:
        st.image(image_obj, use_container_width=True, caption=caption)
    except TypeError:
        # Older Streamlit versions
        st.image(image_obj, use_column_width=True, caption=caption)


def _jump_to_tab(tab_label: str) -> None:
    # Best-effort frontend click on the requested Streamlit tab.
    script = f"""
    <script>
    const target = {json.dumps(tab_label)};
    setTimeout(() => {{
      const doc = window.parent.document;
      const tabs = Array.from(doc.querySelectorAll('button[role="tab"]'));
      const match = tabs.find((tab) => (tab.innerText || '').trim() === target);
      if (match) {{
        match.click();
      }}
    }}, 150);
    </script>
    """
    components.html(script, height=0)


def _clean_cell_str(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "<na>"}:
        return ""
    return text


def _infer_study_date_default(row: pd.Series) -> pd.Timestamp:
    existing = pd.to_datetime(_clean_cell_str(row.get("study_date", "")), errors="coerce")
    if not pd.isna(existing):
        return existing

    manifest_path = Path(_clean_cell_str(row.get("manifest_json", "")))
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for page in manifest.get("pages", []):
                for span in page.get("pii_spans", []):
                    tag = _clean_cell_str(span.get("tag", "")).lower()
                    if tag not in {"date", "dob"}:
                        continue
                    value = _clean_cell_str(span.get("text", ""))
                    ts = pd.to_datetime(value, errors="coerce")
                    if not pd.isna(ts):
                        return ts
        except Exception:
            pass

    # Avoid using "today" directly; default to first of current month.
    return pd.Timestamp.today().replace(day=1)


def _annotate_manual_image(
    image: Image.Image,
    current_points: List[tuple[int, int]],
    pending_boxes: List[tuple[int, int, int, int]],
) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for x1, y1, x2, y2 in pending_boxes:
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        r = 4
        draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=(255, 0, 0))
        draw.ellipse([x2 - r, y2 - r, x2 + r, y2 + r], fill=(255, 0, 0))
    for idx, (x, y) in enumerate(current_points):
        r = 7 if idx == 0 else 6
        color = (0, 200, 0) if idx == 0 else (255, 180, 0)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    return canvas


tab_ingest, tab_run, tab_review, tab_manual, tab_send = st.tabs(
    ["1) Link Reports", "2) Run Redaction", "3) Review", "4) Manual Edits", "5) Send to AWS"]
)

pending_jump = str(st.session_state.get("jump_to_tab", "") or "").strip()
if pending_jump:
    _jump_to_tab(pending_jump)
    st.session_state.jump_to_tab = ""


with tab_ingest:
    st.subheader("Load a report folder or existing tracker")
    selected_site = st.selectbox(
        "Site ID",
        options=SITE_ID_OPTIONS,
        index=(SITE_ID_OPTIONS.index(st.session_state.get("selected_site_id", SITE_ID_OPTIONS[0])) if st.session_state.get("selected_site_id", SITE_ID_OPTIONS[0]) in SITE_ID_OPTIONS else 0),
        key="selected_site_id",
        help="Used across Review and Send-to-AWS, and written to tracker rows for consistency.",
    )
    _apply_site_id_to_tracker(selected_site)
    if st.session_state.post_load_message:
        st.success(st.session_state.post_load_message)
        st.session_state.post_load_message = ""
    input_dir = st.text_input("Report folder path", value=st.session_state.get("input_dir", ""))
    st.session_state["input_dir"] = input_dir
    output_dir = st.text_input(
        "Output folder path (optional)",
        value=st.session_state.get("output_dir", ""),
        help="If blank, defaults to <report folder>/redaction_output",
    )
    st.session_state["output_dir"] = output_dir

    existing_tracker = st.text_input("Existing tracker CSV path (optional)")
    if st.button("Load Existing Tracker"):
        normalized = (existing_tracker or "").strip()
        if normalized in {"", "."}:
            if not input_dir:
                st.warning("Enter report folder path first, or provide tracker CSV path.")
            else:
                tracker_root = output_dir.strip() if output_dir.strip() else str(
                    Path(input_dir).resolve() / "redaction_output"
                )
                tracker_target = str(Path(tracker_root).resolve())
                df = load_tracker(tracker_target, create_if_missing=True)
                st.session_state.df = df
                st.session_state.tracker_path = str(
                    Path(tracker_target).resolve() / "redaction_tracker.csv"
                )
                st.session_state.post_load_message = (
                    f"Tracker ready at {st.session_state.tracker_path} ({len(df)} rows)."
                )
                _apply_site_id_to_tracker(st.session_state.get("selected_site_id", ""))
                st.session_state.jump_to_tab = "2) Run Redaction"
                st.rerun()
        else:
            tracker_target = Path(normalized).expanduser().resolve()
            if tracker_target.is_dir():
                tracker_target = tracker_target / "redaction_tracker.csv"
            df = load_tracker(str(tracker_target), create_if_missing=True)
            st.session_state.df = df
            st.session_state.tracker_path = str(tracker_target)
            st.session_state.post_load_message = f"Tracker ready at {tracker_target} ({len(df)} rows)."
            _apply_site_id_to_tracker(st.session_state.get("selected_site_id", ""))
            st.session_state.jump_to_tab = "2) Run Redaction"
            st.rerun()


with tab_run:
    st.subheader("Run PHI redaction")
    st.caption(
        "One-click processing mode: runs deterministic NLP detection (no Llama), CPU device, and optimized defaults."
    )
    detector_backend = "nlp"
    device = "cpu"
    dpi = 200
    ai_model_path = None
    ai_prompt_template = None
    overwrite = st.checkbox("Overwrite OCR/PDF extraction cache", value=False)
    ocr_backend = "glmocr"
    pdf_text_mode = "hybrid_pdf_text"
    resolved_output_dir = output_dir.strip() if output_dir.strip() else None
    if resolved_output_dir:
        tracker_path_for_run = str(Path(resolved_output_dir).resolve() / "redaction_tracker.csv")
    else:
        tracker_path_for_run = str(Path(input_dir).resolve() / "redaction_output" / "redaction_tracker.csv") if input_dir else ""
    tracker_exists_for_run = bool(tracker_path_for_run and Path(tracker_path_for_run).exists())
    skip_files_in_tracker = st.checkbox(
        "Skip files already listed in tracker",
        value=tracker_exists_for_run,
        help="When enabled, files whose source paths already exist in the tracker are not reprocessed.",
    )

    if st.button("Run Redaction Pipeline"):
        if not input_dir:
            st.error("Enter a report folder path in stage 1")
        else:
            with st.spinner("Running pipeline..."):
                df = process_reports_local(
                    input_dir=input_dir,
                    output_dir=resolved_output_dir,
                    device=device,
                    dpi=dpi,
                    overwrite=overwrite,
                    detector_backend=detector_backend,
                    ocr_backend=ocr_backend,
                    pdf_text_mode=pdf_text_mode,
                    ai_model_path=ai_model_path,
                    ai_prompt_template=ai_prompt_template,
                    append_to_tracker=True,
                    tracker_csv_path=tracker_path_for_run,
                    skip_files_in_tracker=skip_files_in_tracker,
                )
            tracker_path = tracker_path_for_run
            st.session_state.tracker_path = tracker_path
            st.session_state.df = df
            _apply_site_id_to_tracker(st.session_state.get("selected_site_id", ""))
            st.success(f"Pipeline complete. Tracker: {tracker_path}")
            st.dataframe(df)


with tab_review:
    st.subheader("Review")
    reviewer_name = st.text_input("Reviewer name", value=st.session_state.get("reviewer_name", ""))
    st.session_state["reviewer_name"] = reviewer_name

    if st.session_state.df is None:
        st.info("Run stage 2 or load a tracker first.")
    else:
        df = _load_df()
        if df.empty:
            st.info("Tracker is empty. Run stage 2 (Run Redaction) to populate documents.")
            st.stop()
        if "doc_id" not in df.columns:
            st.error("Tracker is missing required column: doc_id")
            st.stop()
        doc_ids = df["doc_id"].dropna().astype(str).str.strip()
        doc_ids = [value for value in doc_ids.tolist() if value]
        if not doc_ids:
            st.info("No document IDs available yet. Run redaction first.")
            st.stop()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Docs", len(df))
        col2.metric("PHI Flagged", int(pd.to_numeric(df["phi_found"], errors="coerce").fillna(0).sum()))
        col3.metric("Approved", int((pd.to_numeric(df["approved_to_send"], errors="coerce").fillna(0) == 1).sum()))
        col4.metric("Errors", int((df["review_status"] == "error").sum()))

        doc_id = st.selectbox("Select document", doc_ids)
        st.session_state["selected_doc_id"] = doc_id
        matches = df[df["doc_id"].astype(str) == doc_id]
        if matches.empty:
            st.warning("Selected document is not present in tracker rows.")
            st.stop()
        row = matches.iloc[-1]
        st.caption(f"Detector: {row.get('detector_backend', 'unknown')} | Source: {row['source_filename']}")

        phi_found_val = pd.to_numeric(row.get("phi_found", 0), errors="coerce")
        phi_found_int = 0 if pd.isna(phi_found_val) else int(phi_found_val)
        if phi_found_int == 1:
            st.error("PHI detected. Review carefully before approval.")
        else:
            st.success("No PHI detected by model(s).")

        source_file = Path(str(row["source_file"]))
        redacted_file = Path(str(row["redacted_file"])) if str(row["redacted_file"]).strip() else None
        st.markdown("### Tracker")
        st.dataframe(st.session_state.df)

        st.markdown("### Required metadata")
        st.caption(
            "Patient ID format: XXX-FFFLLL-i (site code + first 3 first-name letters + last 3 last-name letters + index)."
        )
        meta_col1, meta_col2, meta_col3 = st.columns(3)
        pid_default = _clean_cell_str(row.get("patient_id", "")) or "XXX-FFFLLL-1"
        patient_id_val = meta_col1.text_input(
            "Patient ID",
            value=pid_default,
            key=f"pid_{doc_id}",
        )
        existing_file_id = _clean_cell_str(row.get("file_id", ""))
        default_file_id = existing_file_id
        if not default_file_id:
            ymd = pd.to_datetime(_infer_study_date_default(row), errors="coerce")
            ymd_str = "19000101" if pd.isna(ymd) else ymd.strftime("%Y%m%d")
            default_file_id = f"{pid_default}_{ymd_str}_1"
        file_id_val = st.text_input(
            "File ID",
            value=default_file_id,
            key=f"rid_{doc_id}",
        )
        modality_value = _clean_cell_str(row.get("modality_type", ""))
        modality_options = ["CMR", "CT", "Echo", "StressTest", "Cath", "Other"]
        modality_val = meta_col2.selectbox(
            "Modality type",
            options=modality_options,
            index=(modality_options.index(modality_value) if modality_value in modality_options else 5),
            key=f"mod_{doc_id}",
        )
        default_date = _infer_study_date_default(row)
        study_date_val = meta_col3.date_input(
            "Study date (day forced to 01)",
            value=default_date.date(),
            key=f"study_{doc_id}",
        )
        if st.button("Save Metadata", key=f"save_meta_{doc_id}"):
            try:
                updated_df = set_review_metadata(
                    tracker_csv=st.session_state.tracker_path,
                    doc_id=doc_id,
                    site_id=st.session_state.get("selected_site_id", ""),
                    force_id=patient_id_val,
                    file_id=file_id_val,
                    first_name=_clean_cell_str(row.get("first_name", "")),
                    last_name=_clean_cell_str(row.get("last_name", "")),
                    mrn=_clean_cell_str(row.get("mrn", "")),
                    dob=_clean_cell_str(row.get("dob", "")),
                    gender=_clean_cell_str(row.get("gender", "")),
                    modality_type=modality_val,
                    study_date=str(study_date_val),
                )
                st.session_state.df = updated_df
                saved_row = updated_df[updated_df["doc_id"] == doc_id].iloc[-1]
                st.success(f"Saved. Normalized study date: {saved_row.get('study_date', '')}")
            except Exception as exc:
                st.error(str(exc))

        st.markdown("### Why Things Were Redacted")
        manifest_path = Path(str(row.get("manifest_json", "") or ""))
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                reason_rows = []
                for page in manifest.get("pages", []):
                    page_number = int(page.get("page_index", 0)) + 1
                    for span in page.get("effective_redaction_spans", page.get("pii_spans", [])):
                        reason_rows.append(
                            {
                                "page": page_number,
                                "tag": span.get("tag", ""),
                                "value": str(span.get("text", ""))[:120],
                                "source": span.get("source", ""),
                                "reason": span.get("reason", ""),
                            }
                        )
                if reason_rows:
                    st.dataframe(pd.DataFrame(reason_rows), use_container_width=True)
                else:
                    st.info("No explicit PII spans recorded in manifest.")
            except Exception as exc:
                st.warning(f"Could not parse manifest for redaction reasons: {exc}")
        else:
            st.info("Manifest not found for this document.")

        st.markdown("### Review decision")
        notes = st.text_area("Notes")
        approve = st.button("Approve")
        reject = st.button("Reject")
        if approve or reject:
            if not reviewer_name.strip():
                st.warning("Reviewer name is required")
            else:
                approved = bool(approve)
                updated_df = set_review_decision(
                    tracker_csv=st.session_state.tracker_path,
                    doc_id=doc_id,
                    approved_to_send=approved,
                    reviewer=reviewer_name,
                    review_notes=notes,
                )
                st.session_state.df = updated_df
                if approved:
                    st.session_state.jump_to_tab = "5) Send to AWS"
                    st.rerun()
                st.success("Decision saved")

        st.markdown("### Downloads")
        down_col1, down_col2 = st.columns(2)
        with down_col1:
            if source_file.exists():
                with source_file.open("rb") as file_handle:
                    st.download_button(
                        "Download Original",
                        file_handle,
                        file_name=source_file.name,
                        key=f"download_orig_{doc_id}",
                    )
        with down_col2:
            if redacted_file and redacted_file.exists():
                with redacted_file.open("rb") as file_handle:
                    st.download_button(
                        "Download Redacted",
                        file_handle,
                        file_name=redacted_file.name,
                        key=f"download_red_{doc_id}",
                    )

        st.markdown("### Documents")
        view_original, view_redacted = st.columns(2)
        with view_original:
            st.markdown("#### Original")
            if source_file.suffix.lower() == ".pdf":
                if source_file.exists():
                    _render_pdf_preview(source_file)
            elif source_file.exists():
                _show_image_compat(str(source_file))

        with view_redacted:
            st.markdown("#### Redacted")
            if redacted_file and redacted_file.exists():
                if redacted_file.suffix.lower() == ".pdf":
                    _render_pdf_preview(redacted_file)
                else:
                    _show_image_compat(str(redacted_file))
            else:
                st.warning("No redacted output found")


with tab_manual:
    st.subheader("Manual Edits")
    if st.session_state.df is None:
        st.info("Run stage 2 or load a tracker first.")
    else:
        df = _load_df()
        if df.empty or "doc_id" not in df.columns:
            st.info("No documents available for manual edits.")
        else:
            doc_ids = [value for value in df["doc_id"].dropna().astype(str).str.strip().tolist() if value]
            if not doc_ids:
                st.info("No document IDs available yet.")
            else:
                default_doc = st.session_state.get("selected_doc_id", doc_ids[0])
                if default_doc not in doc_ids:
                    default_doc = doc_ids[0]
                manual_doc_id = st.selectbox(
                    "Select document for manual edits",
                    doc_ids,
                    index=doc_ids.index(default_doc),
                    key="manual_doc_select",
                )
                st.session_state["selected_doc_id"] = manual_doc_id
                matches = df[df["doc_id"].astype(str) == manual_doc_id]
                if matches.empty:
                    st.warning("Selected document not found.")
                else:
                    row = matches.iloc[-1]
                    review_dir = Path(str(row["review_pages_dir"]))

                    if review_dir.exists():
                        page_images = sorted(review_dir.glob("*.png"))
                        if page_images:
                            st.markdown("### Page Selection")
                            page_count = len(page_images)
                            page_state_key = f"manual_page_idx_{manual_doc_id}"
                            current_page = int(st.session_state.get(page_state_key, 1))
                            if current_page < 1 or current_page > page_count:
                                current_page = 1
                            top_col1, top_col2, top_col3 = st.columns([2, 1, 1])
                            with top_col1:
                                page_num = st.selectbox(
                                    "Select page",
                                    options=list(range(1, page_count + 1)),
                                    index=current_page - 1,
                                    key=f"manual_page_select_{manual_doc_id}",
                                )
                                st.session_state[page_state_key] = int(page_num)
                            with top_col2:
                                st.metric("Total Pages", page_count)
                            with top_col3:
                                if st.button("Next Page", key=f"next_page_{manual_doc_id}"):
                                    next_page = int(page_num) + 1
                                    if next_page > page_count:
                                        next_page = page_count
                                    st.session_state[page_state_key] = next_page
                                    st.rerun()

                            selected_path = page_images[int(page_num) - 1]

                            st.markdown("### Click-To-Box Viewer")
                            selected_img = Image.open(selected_path)
                            img_w, img_h = selected_img.size

                            pending_boxes_key = f"pending_boxes_{manual_doc_id}_{page_num}"
                            active_base_boxes_key = f"active_base_boxes_{manual_doc_id}_{page_num}"
                            if pending_boxes_key not in st.session_state:
                                st.session_state[pending_boxes_key] = []
                            if active_base_boxes_key not in st.session_state:
                                st.session_state[active_base_boxes_key] = get_page_redaction_boxes(
                                    tracker_csv=st.session_state.tracker_path,
                                    doc_id=manual_doc_id,
                                    page_number=int(page_num),
                                )
                            pending_boxes = list(st.session_state.get(pending_boxes_key, []))
                            active_base_boxes = list(st.session_state.get(active_base_boxes_key, []))
                            annotated_img = _annotate_manual_image(
                                selected_img,
                                current_points=[],
                                pending_boxes=[*active_base_boxes, *pending_boxes],
                            )

                            st.markdown(
                                "<div style='font-size:1.05rem; font-weight:700; line-height:1.45;'>"
                                "Instructions: Select <b>Click-Drag Rectangle</b>, drag and release on the image "
                                "to draw one or more boxes, click <b>Queue Drawn Box(es)</b>, then click "
                                "<b>Run Box Redaction</b> to apply."
                                "</div>",
                                unsafe_allow_html=True,
                            )

                            viewer_col, panel_col = st.columns([3, 1])
                            with viewer_col:
                                st.caption("Viewer")
                                _show_image_compat(annotated_img)
                                if st_canvas is not None and bool(st.session_state.get("drag_rect_supported", True)):
                                    try:
                                        canvas_result = st_canvas(
                                            fill_color="rgba(0, 0, 0, 0.2)",
                                            stroke_width=2,
                                            stroke_color="#FF0000",
                                            background_image=annotated_img,
                                            update_streamlit=True,
                                            height=img_h,
                                            width=img_w,
                                            drawing_mode="rect",
                                            key=f"drag_canvas_{manual_doc_id}_{page_num}",
                                        )
                                        if st.button("Queue Drawn Box(es)", key=f"queue_drag_boxes_{manual_doc_id}_{page_num}"):
                                            objects = []
                                            if canvas_result and canvas_result.json_data:
                                                objects = canvas_result.json_data.get("objects", []) or []
                                            new_boxes = []
                                            for obj in objects:
                                                if obj.get("type") != "rect":
                                                    continue
                                                left = float(obj.get("left", 0))
                                                top = float(obj.get("top", 0))
                                                width = float(obj.get("width", 0))
                                                height = float(obj.get("height", 0))
                                                scale_x = float(obj.get("scaleX", 1))
                                                scale_y = float(obj.get("scaleY", 1))
                                                x1 = int(round(left))
                                                y1 = int(round(top))
                                                x2 = int(round(left + (width * scale_x)))
                                                y2 = int(round(top + (height * scale_y)))
                                                xa, xb = sorted([x1, x2])
                                                ya, yb = sorted([y1, y2])
                                                if xa != xb and ya != yb:
                                                    new_boxes.append((xa, ya, xb, yb))
                                            if not new_boxes:
                                                st.warning("No valid rectangles detected.")
                                            else:
                                                pending = list(st.session_state[pending_boxes_key])
                                                pending.extend(new_boxes)
                                                st.session_state[pending_boxes_key] = pending
                                                st.success(f"Queued {len(new_boxes)} box(es).")
                                                st.rerun()

                                        st.caption("Move/resize existing boxes")
                                        editable_boxes = [*active_base_boxes, *pending_boxes]
                                        initial_objects = [
                                            {
                                                "type": "rect",
                                                "left": float(b[0]),
                                                "top": float(b[1]),
                                                "width": float(max(1, b[2] - b[0])),
                                                "height": float(max(1, b[3] - b[1])),
                                                "fill": "rgba(255, 0, 0, 0.08)",
                                                "stroke": "#FF0000",
                                                "strokeWidth": 2,
                                            }
                                            for b in editable_boxes
                                        ]
                                        transform_result = st_canvas(
                                            fill_color="rgba(255, 0, 0, 0.08)",
                                            stroke_width=2,
                                            stroke_color="#FF0000",
                                            background_image=annotated_img,
                                            update_streamlit=True,
                                            height=img_h,
                                            width=img_w,
                                            drawing_mode="transform",
                                            initial_drawing={"version": "4.4.0", "objects": initial_objects},
                                            key=f"transform_canvas_{manual_doc_id}_{page_num}",
                                        )
                                        if st.button("Apply Resize/Move Edits", key=f"apply_transform_boxes_{manual_doc_id}_{page_num}"):
                                            objects = []
                                            if transform_result and transform_result.json_data:
                                                objects = transform_result.json_data.get("objects", []) or []
                                            edited_boxes = []
                                            for obj in objects:
                                                if obj.get("type") != "rect":
                                                    continue
                                                left = float(obj.get("left", 0))
                                                top = float(obj.get("top", 0))
                                                width = float(obj.get("width", 0))
                                                height = float(obj.get("height", 0))
                                                scale_x = float(obj.get("scaleX", 1))
                                                scale_y = float(obj.get("scaleY", 1))
                                                x1 = int(round(left))
                                                y1 = int(round(top))
                                                x2 = int(round(left + (width * scale_x)))
                                                y2 = int(round(top + (height * scale_y)))
                                                xa, xb = sorted([x1, x2])
                                                ya, yb = sorted([y1, y2])
                                                if xa != xb and ya != yb:
                                                    edited_boxes.append((xa, ya, xb, yb))
                                            st.session_state[active_base_boxes_key] = edited_boxes
                                            st.session_state[pending_boxes_key] = []
                                            st.success(f"Applied {len(edited_boxes)} edited box(es) to current page state.")
                                            st.rerun()
                                    except Exception:
                                        st.session_state.drag_rect_supported = False
                                        st.warning(
                                            "Click-drag mode is incompatible with this Streamlit runtime."
                                        )
                                        st.rerun()
                                else:
                                    st.warning("Drag-and-release box tool is unavailable in this environment.")

                            with panel_col:
                                st.markdown(
                                    "<div style='position: sticky; top: 1rem;'>",
                                    unsafe_allow_html=True,
                                )
                                st.markdown("### Manual Redaction Panel")
                                st.write(f"Document: `{manual_doc_id}`")
                                st.write(f"Page: `{page_num}/{page_count}`")
                                if active_base_boxes:
                                    st.caption("AI/NLP boxes (can be removed):")
                                    ai_df = pd.DataFrame(
                                        [
                                            {"#": i + 1, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}
                                            for i, b in enumerate(active_base_boxes)
                                        ]
                                    )
                                    st.dataframe(ai_df, use_container_width=True, hide_index=True)
                                    delete_idx = st.number_input(
                                        "AI/NLP box # to delete",
                                        min_value=1,
                                        max_value=max(1, len(active_base_boxes)),
                                        value=1,
                                        step=1,
                                        key=f"delete_ai_idx_{manual_doc_id}_{page_num}",
                                    )
                                    if st.button("Delete Selected AI/NLP Box", key=f"delete_ai_box_{manual_doc_id}_{page_num}"):
                                        idx = int(delete_idx) - 1
                                        if 0 <= idx < len(active_base_boxes):
                                            active_base_boxes.pop(idx)
                                            st.session_state[active_base_boxes_key] = active_base_boxes
                                            st.rerun()
                                if pending_boxes:
                                    st.caption("Queued manual boxes:")
                                    queue_df = pd.DataFrame(
                                        [
                                            {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}
                                            for b in pending_boxes
                                        ]
                                    )
                                    st.dataframe(queue_df, use_container_width=True, hide_index=True)
                                else:
                                    st.info("No queued boxes yet.")

                                c1, c2 = st.columns(2)
                                with c1:
                                    if st.button("Undo Last Move", key=f"undo_last_{manual_doc_id}_{page_num}"):
                                        if pending_boxes:
                                            pending_boxes.pop()
                                            st.session_state[pending_boxes_key] = pending_boxes
                                with c2:
                                    if st.button("Remove All Moves", key=f"clear_all_{manual_doc_id}_{page_num}"):
                                        st.session_state[pending_boxes_key] = []

                                if st.button("Run Box Redaction", key=f"run_boxes_{manual_doc_id}_{page_num}"):
                                    final_boxes = [*active_base_boxes, *pending_boxes]
                                    if not final_boxes:
                                        st.warning("No boxes to apply.")
                                    else:
                                        updated_df = override_page_redactions(
                                            tracker_csv=st.session_state.tracker_path,
                                            doc_id=manual_doc_id,
                                            page_number=int(page_num),
                                            boxes_xyxy=final_boxes,
                                        )
                                        st.session_state.df = updated_df
                                        st.session_state[pending_boxes_key] = []
                                        st.session_state[active_base_boxes_key] = final_boxes
                                        st.success(f"Applied {len(final_boxes)} total box(es).")
                                        st.rerun()

                                if st.button("Compile Edited PDF", key=f"compile_pdf_{manual_doc_id}"):
                                    updated_df = compile_redacted_document(
                                        tracker_csv=st.session_state.tracker_path,
                                        doc_id=manual_doc_id,
                                    )
                                    st.session_state.df = updated_df
                                    st.success("Compiled updated redacted document from edited pages.")
                                st.markdown("</div>", unsafe_allow_html=True)

                        else:
                            st.info("No redacted preview pages found for this document.")
                    else:
                        st.info("No review page directory found for this document.")


with tab_send:
    st.subheader("Send approved reports to AWS")
    if st.session_state.df is None:
        st.info("Run stage 2 or load a tracker first.")
    else:
        site_id = str(st.session_state.get("selected_site_id", SITE_ID_OPTIONS[0]) or "").strip()
        st.caption(f"Using Site ID from Home tab: {site_id}")
        cfg = AWSConfig()
        tracker_df = _load_df()
        approved_mask = pd.to_numeric(tracker_df["approved_to_send"], errors="coerce").fillna(0) == 1
        approved_df = tracker_df[approved_mask].copy()
        approved_count = len(approved_df)
        if approved_count and "review_status" in approved_df.columns:
            manual_count = int((approved_df["review_status"].astype(str) == "manual_redaction_applied").sum())
        else:
            manual_count = 0
        auto_count = int(approved_count - manual_count)
        reviewers = sorted(
            {
                str(value).strip()
                for value in tracker_df.get("reviewer", pd.Series(dtype=str)).tolist()
                if str(value).strip()
            }
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Documents Approved To Send", approved_count)
        c2.metric("Auto-Redacted (No Manual Edit)", auto_count)
        c3.metric("Required Manual Redaction", manual_count)
        st.write(f"Reviewers: {', '.join(reviewers) if reviewers else 'None yet'}")

        if st.button("Collect Approved Locally"):
            files = collect_approved_files(tracker_csv=st.session_state.tracker_path)
            st.success(f"Collected {len(files)} approved file(s)")
            for path in files:
                st.write(path)

        if st.button("Send Approved Reports"):
            uploaded = upload_approved_files_to_s3(
                tracker_csv=st.session_state.tracker_path,
                config=cfg,
                site_id=site_id,
            )
            st.success(f"Sent {len(uploaded)} file object(s).")
            for key in uploaded:
                st.write(key)

        if st.button("Sync Review Metrics"):
            tracker_df = _load_df()
            synced = 0
            for _, row in tracker_df.iterrows():
                approved_val = pd.to_numeric(row.get("approved_to_send", 0), errors="coerce")
                approved = (not pd.isna(approved_val)) and int(approved_val) == 1
                reviewer = str(row.get("reviewer", "") or "")
                if not reviewer:
                    continue
                log_review_metric_to_ddb(
                    row=row.to_dict(),
                    reviewer=reviewer,
                    approved=approved,
                    review_notes=str(row.get("review_notes", "") or ""),
                    config=cfg,
                    site_id=site_id,
                )
                synced += 1
            st.success(f"Synced {synced} review metrics to DynamoDB")




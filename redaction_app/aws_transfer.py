from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import boto3
import pandas as pd
import requests

from utils import load_tracker, run_ocr, save_tracker

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class AWSConfig:
    region: str = "us-east-1"
    s3_bucket: str = "safe-harbor-data-lake"
    s3_prefix: str = "phi-redaction-uploads"
    ddb_table: str = "safe-harbor-redaction-review"


def _normalize_modality_for_upload(value: object) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "unknown"
    clean = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    aliases = {
        "cmr": "cmr",
        "cardiac_mri": "cmr",
        "cardiac_mr": "cmr",
        "mri": "cmr",
        "ct": "ct",
        "echo": "echo",
        "echocardiogram": "echo",
        "stress": "stress_test",
        "stress_test": "stress_test",
        "stresstest": "stress_test",
        "cath": "cath",
        "catheterization": "cath",
    }
    return aliases.get(clean, clean or "unknown")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_review_metric_to_ddb(
    row: Dict[str, object],
    reviewer: str,
    approved: bool,
    review_notes: str,
    config: AWSConfig,
    site_id: str = "external-site",
) -> None:
    ddb = boto3.resource("dynamodb", region_name=config.region)
    table = ddb.Table(config.ddb_table)

    item = {
        "pk": f"SITE#{site_id}",
        "sk": f"DOC#{row.get('file_id', row.get('report_id', row.get('doc_id', 'unknown')))}#{_utc_now_iso()}",
        "file_id": str(row.get("file_id", row.get("report_id", row.get("doc_id", "")))),
        "reviewer": reviewer,
        "approved_to_send": int(bool(approved)),
        "review_status": str(row.get("review_status", "")),
        "review_notes": review_notes,
        "phi_found": int(float(row.get("phi_found", 0)) if row.get("phi_found") is not None else 0),
        "total_pii_spans": int(float(row.get("total_pii_spans", 0)) if row.get("total_pii_spans") is not None else 0),
        "detector_backend": str(row.get("detector_backend", "")),
        "recorded_at_utc": _utc_now_iso(),
    }
    table.put_item(Item=item)


def upload_approved_files_to_s3(
    tracker_csv: str,
    config: AWSConfig,
    site_id: str = "external-site",
    only_unsent: bool = True,
) -> List[str]:
    tracker_df = load_tracker(tracker_csv)
    approved = tracker_df[tracker_df["approved_to_send"] == 1].copy()
    if only_unsent and "sent_to_aws" in approved.columns:
        approved = approved[pd.to_numeric(approved["sent_to_aws"], errors="coerce").fillna(0).ne(1)].copy()

    if approved.empty:
        return []

    client = boto3.client("s3", region_name=config.region)
    uploaded_keys: List[str] = []

    for _, row in approved.iterrows():
        redacted_file = Path(str(row["redacted_file"]))
        if not redacted_file.exists():
            continue
        safe_file_id = _normalize_file_id(
            str(row.get("force_id", "") or row.get("patient_id", "") or ""),
            str(row.get("study_date", "") or ""),
            str(row.get("file_id", "") or row.get("report_id", "") or ""),
        )
        safe_filename = f"{safe_file_id}{redacted_file.suffix.lower() or '.pdf'}"

        key = (
            f"{config.s3_prefix}/site={site_id}/file_id={safe_file_id}/"
            f"{safe_filename}"
        )
        client.upload_file(str(redacted_file), config.s3_bucket, key)
        uploaded_keys.append(key)
        tracker_df.loc[row.name, "sent_to_aws"] = 1
        tracker_df.loc[row.name, "sent_to_aws_at_utc"] = _utc_now_iso()

    save_tracker(tracker_df, tracker_csv)
    return uploaded_keys


def generate_presigned_put_url(
    object_key: str,
    config: AWSConfig,
    expires_sec: int = 3600,
) -> str:
    client = boto3.client("s3", region_name=config.region)
    return client.generate_presigned_url(
        ClientMethod="put_object",
        Params={"Bucket": config.s3_bucket, "Key": object_key},
        ExpiresIn=expires_sec,
    )


def request_presigned_upload_via_api(
    api_url: str,
    payload: Dict[str, object],
    api_key: Optional[str] = None,
    timeout_sec: int = 20,
) -> Dict[str, object]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        token_value = str(api_key)
        if token_value.lower().startswith("bearer "):
            headers["Authorization"] = token_value
        elif token_value.count(".") == 2:
            headers["Authorization"] = token_value
        else:
            headers["x-api-key"] = api_key

    response = requests.post(
        api_url,
        headers=headers,
        data=json.dumps(payload),
        timeout=timeout_sec,
    )
    if response.status_code >= 400:
        body_preview = (response.text or "").strip().replace("\n", " ")
        if len(body_preview) > 500:
            body_preview = body_preview[:500] + "..."
        raise requests.HTTPError(
            f"{response.status_code} {response.reason} for {api_url} :: {body_preview}",
            response=response,
        )
    return response.json()


def upload_file_with_presigned_url(
    presigned_url: str,
    file_path: str,
    required_headers: Optional[Dict[str, object]] = None,
) -> None:
    file_obj = Path(file_path)
    if not file_obj.exists():
        raise FileNotFoundError(f"File not found: {file_obj}")

    headers: Dict[str, str] = {}
    if required_headers:
        for k, v in required_headers.items():
            if v is None:
                continue
            headers[str(k)] = str(v)

    with file_obj.open("rb") as handle:
        last_exc: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                handle.seek(0)
                response = requests.put(
                    presigned_url,
                    data=handle,
                    headers=headers or None,
                    timeout=120,
                )
                response.raise_for_status()
                return
            except Exception as exc:
                last_exc = exc
        if last_exc is not None:
            raise last_exc


def upload_bytes_with_presigned_url(
    presigned_url: str,
    payload_bytes: bytes,
    required_headers: Optional[Dict[str, object]] = None,
) -> None:
    headers: Dict[str, str] = {}
    if required_headers:
        for k, v in required_headers.items():
            if v is None:
                continue
            headers[str(k)] = str(v)
    last_exc: Optional[Exception] = None
    for _attempt in range(1, 4):
        try:
            response = requests.put(
                presigned_url,
                data=payload_bytes,
                headers=headers or None,
                timeout=120,
            )
            response.raise_for_status()
            return
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def _derive_input_kind_from_row(row: pd.Series) -> str:
    raw_kind = str(row.get("input_kind", "") or "").strip().lower()
    if raw_kind:
        return raw_kind
    ext = str(row.get("source_ext", "") or "").strip().lower()
    if ext == ".pdf":
        return "pdf_report"
    if ext in _IMAGE_EXTS:
        return "screenshot_report"
    return "report"


def _normalize_file_id(force_id: str, study_date: str, file_id: str) -> str:
    candidate = re.sub(r"\s+", "", str(file_id or "").strip().upper())
    # Preserve caller-provided file IDs whenever they already encode date/instance.
    # Older strict format checks could collapse distinct rows into the same fallback ID.
    if candidate:
        if re.match(r"^[A-Z0-9\-]+_\d{8}_\d+$", candidate):
            return candidate
        if re.match(r"^[A-Z0-9\-]+_\d{8}$", candidate):
            return f"{candidate}_1"
        m_any = re.search(r"^(?P<prefix>[A-Z0-9\-]+)_(?P<ymd>\d{8})(?:_(?P<inst>\d+))?$", candidate)
        if m_any:
            prefix = str(m_any.group("prefix") or "").strip()
            ymd = str(m_any.group("ymd") or "").strip()
            inst = str(m_any.group("inst") or "1").strip() or "1"
            if prefix and ymd:
                return f"{prefix}_{ymd}_{inst}"
    fid = (force_id or "XXX-FFFLLL-1").strip().upper()
    dt = pd.to_datetime(study_date, errors="coerce")
    ymd = "19000101" if pd.isna(dt) else dt.strftime("%Y%m%d")
    # Try to retain an explicit instance suffix from file_id if present.
    inst = "1"
    m_inst = re.search(r"_(\d+)\s*$", candidate) if candidate else None
    if m_inst:
        inst = str(m_inst.group(1) or "1")
    return f"{fid}_{ymd}_{inst}"


def _extract_manifest_user_metrics(manifest_path: Path) -> Dict[str, int]:
    if not manifest_path.exists():
        return {"user_deleted_box_count": 0, "user_added_box_count": 0}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manual = manifest.get("manual_redaction", {}) or {}
        return {
            "user_deleted_box_count": _safe_int(manual.get("user_deleted_box_count", 0)),
            "user_added_box_count": _safe_int(manual.get("user_added_box_count", 0)),
        }
    except Exception:
        return {"user_deleted_box_count": 0, "user_added_box_count": 0}


def _safe_redact_text_from_spans(
    source_text: str,
    spans: List[Dict[str, object]],
) -> str:
    text = str(source_text or "")
    if not text:
        return ""
    chars = list(text)
    intervals: List[tuple[int, int, str]] = []
    for span in spans:
        try:
            start = int(span.get("start", -1))
            end = int(span.get("end", -1))
        except Exception:
            continue
        if start < 0 or end <= start:
            continue
        tag = str(span.get("tag", "") or "").lower()
        if tag in {"patient_name", "mrn"}:
            fill = "#"
        else:
            fill = "x"
        intervals.append((start, end, fill))
    intervals.sort(key=lambda t: (t[0], t[1]))
    for start, end, fill in intervals:
        s = max(0, min(len(chars), start))
        e = max(0, min(len(chars), end))
        for i in range(s, e):
            if chars[i].isspace():
                continue
            chars[i] = fill
    return "".join(chars)


def _safe_redact_text_from_terms(
    source_text: str,
    terms_with_tags: List[tuple[str, str]],
) -> str:
    text = str(source_text or "")
    if not text:
        return ""
    chars = list(text)
    for raw_term, raw_tag in terms_with_tags:
        term = str(raw_term or "").strip()
        if not term:
            continue
        tag = str(raw_tag or "").strip().lower()
        fill = "#" if tag in {"patient_name", "mrn"} else "x"
        # Prevent short numeric PHI terms (e.g., DOB day "17") from redacting
        # unrelated measurements such as "179.0".
        if re.fullmatch(r"\d+", term):
            # Extremely short numeric tokens (common for day-only/date fragments)
            # are too collision-prone and can corrupt measurements like 171.9.
            if len(term) <= 2 and tag in {"date", "dob", "date_day", "time"}:
                continue
            # Match numeric term as a standalone token, but never when attached
            # to decimal points (e.g., avoid redacting the "9" in "171.9").
            pat = re.compile(rf"(?<![\d.]){re.escape(term)}(?![\d.])")
            for m in pat.finditer(text):
                idx, end = m.start(), m.end()
                for i in range(max(0, idx), min(len(chars), end)):
                    if chars[i].isspace():
                        continue
                    chars[i] = fill
            continue

        lower_text = text.lower()
        lower_term = term.lower()
        start = 0
        while True:
            idx = lower_text.find(lower_term, start)
            if idx < 0:
                break
            end = idx + len(lower_term)
            for i in range(max(0, idx), min(len(chars), end)):
                if chars[i].isspace():
                    continue
                chars[i] = fill
            start = max(idx + 1, end)
    return "".join(chars)


def _normalize_notes_lines(text: str) -> List[str]:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in normalized.split("\n")]
    out: List[str] = []
    blank = False
    for ln in lines:
        if ln == "":
            if blank:
                continue
            out.append("")
            blank = True
            continue
        out.append(ln)
        blank = False
    return out


def _split_table_row(line: str) -> Optional[List[str]]:
    if "|" in line:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2 and any(p for p in parts):
            return parts
    tab_parts = [p.strip() for p in line.split("\t")]
    if len(tab_parts) >= 2 and any(p for p in tab_parts):
        return tab_parts
    space_parts = [p.strip() for p in re.split(r"\s{2,}", line.strip())]
    if len(space_parts) >= 2 and any(p for p in space_parts):
        return space_parts
    return None


def _format_table_block(rows: List[List[str]]) -> List[str]:
    if not rows:
        return []
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]
    col_widths = [max(len(r[c]) for r in padded) for c in range(width)]
    out: List[str] = []
    for r in padded:
        cells: List[str] = []
        for c, val in enumerate(r):
            if c == width - 1:
                cells.append(val)
            else:
                cells.append(val.ljust(col_widths[c]))
        out.append(" | ".join(cells).rstrip())
    return out


def _format_readable_notes(text: str) -> str:
    lines = _normalize_notes_lines(text)
    if not lines:
        return ""
    out: List[str] = []
    table_buf: List[List[str]] = []

    def flush_table() -> None:
        nonlocal table_buf
        if not table_buf:
            return
        if len(table_buf) >= 2:
            out.extend(_format_table_block(table_buf))
        else:
            out.append(" ".join(table_buf[0]).strip())
        table_buf = []

    for ln in lines:
        row = _split_table_row(ln)
        if row is not None:
            table_buf.append(row)
            continue
        flush_table()
        out.append(ln)
    flush_table()
    return "\n".join(out).strip()


def _extract_pdf_text_by_page(source_pdf: Path) -> Dict[int, str]:
    try:
        import fitz  # type: ignore
    except Exception:
        return {}
    try:
        doc = fitz.open(source_pdf)
    except Exception:
        return {}
    out: Dict[int, str] = {}
    try:
        for idx in range(len(doc)):
            try:
                page = doc.load_page(idx)
                text = str(page.get_text("text") or "")
            except Exception:
                continue
            if text.strip():
                out[idx] = text
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return out


def _select_post_redaction_ocr_backend(ocr_backend: str) -> str:
    backend = str(ocr_backend or "paddle").strip().lower()
    if backend in {"paddle", "paddleocr"}:
        return "paddle"
    if backend in {"tesseract"}:
        return "tesseract"
    if backend in {"glmocr", "glm_ocr", "glm-ocr"}:
        return "glmocr"
    # Avoid forced network dependency when the primary pipeline used Textract.
    return "paddle"


def _extract_post_redaction_text_by_page(
    manifest_path: Path,
    manifest: Dict[str, object],
) -> Dict[int, str]:
    pages = list(manifest.get("pages", []) or [])
    if not pages:
        return {}
    backend = _select_post_redaction_ocr_backend(str(manifest.get("ocr_backend", "") or ""))
    out: Dict[int, str] = {}
    for page in pages:
        if not isinstance(page, dict):
            continue
        try:
            page_index = int(page.get("page_index", 0))
        except Exception:
            continue
        review_png_raw = str(page.get("review_png_path", "") or "").strip()
        if not review_png_raw:
            continue
        review_png = Path(review_png_raw)
        if not review_png.is_absolute():
            review_png = manifest_path.parent / review_png
        if not review_png.exists() or review_png.suffix.lower() not in _IMAGE_EXTS:
            continue
        try:
            ocr_text, _ = run_ocr(str(review_png), ocr_backend=backend)
        except Exception:
            continue
        if str(ocr_text or "").strip():
            out[page_index] = str(ocr_text)
    if len(out) >= len(pages):
        return out

    redacted_file_raw = str(manifest.get("redacted_file", "") or "").strip()
    if not redacted_file_raw:
        return out
    redacted_file = Path(redacted_file_raw)
    if not redacted_file.is_absolute():
        redacted_file = manifest_path.parent / redacted_file
    if not redacted_file.exists() or redacted_file.suffix.lower() != ".pdf":
        return out

    try:
        import fitz  # type: ignore
    except Exception:
        return out
    try:
        doc = fitz.open(redacted_file)
    except Exception:
        return out
    try:
        for page in pages:
            if not isinstance(page, dict):
                continue
            try:
                page_index = int(page.get("page_index", 0))
            except Exception:
                continue
            if page_index in out:
                continue
            if page_index < 0 or page_index >= len(doc):
                continue
            try:
                pdf_page = doc.load_page(page_index)
                pix = pdf_page.get_pixmap(dpi=200, alpha=False)
            except Exception:
                continue
            temp_path: Optional[Path] = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    temp_path = Path(tmp.name)
                pix.save(str(temp_path))
                ocr_text, _ = run_ocr(str(temp_path), ocr_backend=backend)
                if str(ocr_text or "").strip():
                    out[page_index] = str(ocr_text)
            except Exception:
                continue
            finally:
                if temp_path is not None:
                    try:
                        temp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return out


def _normalize_line_for_measurements(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _is_plausible_height(value: str) -> bool:
    v = str(value or "").strip(" :;,.")
    if not v or len(v) > 24 or not re.search(r"\d", v):
        return False
    if re.search(r"[A-Za-z]", v) and not re.search(r"\b(cm|mm|m|ft|in)\b", v, flags=re.IGNORECASE):
        return False
    return bool(re.fullmatch(r"[0-9A-Za-z\.\s'\"/-]+", v))


def _is_plausible_weight(value: str) -> bool:
    v = str(value or "").strip(" :;,.")
    if not v or len(v) > 20:
        return False
    m = re.fullmatch(r"(?i)(\d{2,3}(?:\.\d{1,2})?)\s*(lb|lbs|kg)?", v)
    if not m:
        return False
    try:
        num = float(m.group(1))
    except Exception:
        return False
    return 10.0 <= num <= 700.0


def _is_plausible_pre_ex_hr(value: str) -> bool:
    v = str(value or "").strip(" :;,.")
    if not v or len(v) > 12:
        return False
    m = re.fullmatch(r"(?i)(\d{2,3})(?:\s*bpm)?", v)
    if not m:
        return False
    try:
        num = int(m.group(1))
    except Exception:
        return False
    return 20 <= num <= 260


def _is_plausible_baseline_o2_sat(value: str) -> bool:
    v = str(value or "").strip(" :;,.")
    if not v or len(v) > 12:
        return False
    m = re.fullmatch(r"(?i)%?\s*(\d{2,3})(?:\s*%)?", v)
    if not m:
        return False
    try:
        num = int(m.group(1))
    except Exception:
        return False
    return 50 <= num <= 100


def _extract_safe_measurements(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw_line in str(text or "").splitlines():
        line = _normalize_line_for_measurements(raw_line)
        if not line:
            continue
        m_height = re.search(r"(?i)\bheight\b\s*[:=|]?\s*(?P<value>[0-9A-Za-z\.\s'\"/-]{1,24})", line)
        if m_height and "height" not in out:
            value = str(m_height.group("value") or "").strip()
            if _is_plausible_height(value):
                out["height"] = value

        m_weight = re.search(r"(?i)\bweight\b\s*[:=|]?\s*(?P<value>[0-9A-Za-z\.\s]{1,20})", line)
        if m_weight and "weight" not in out:
            value = str(m_weight.group("value") or "").strip()
            if _is_plausible_weight(value):
                out["weight"] = value

        m_pre_hr = re.search(
            r"(?i)\b(?:pre[\s-]*exercise(?:\s+heart\s*rate|\s+hr)?|baseline\s+hr)\b\s*[:=|]?\s*(?P<value>[0-9A-Za-z\.\s]{1,12})",
            line,
        )
        if m_pre_hr and "pre_exercise_hr" not in out:
            value = str(m_pre_hr.group("value") or "").strip()
            if _is_plausible_pre_ex_hr(value):
                out["pre_exercise_hr"] = value

        m_o2 = re.search(
            r"(?i)\b(?:baseline\s*o\s*2\s*sat|baseline\s*oxygen\s*saturation|o\s*2\s*sat)\b\s*[:=|]?\s*(?P<value>%?\s*\d{2,3}\s*%?)",
            line,
        )
        if m_o2 and "baseline_o2_sat" not in out:
            value = str(m_o2.group("value") or "").strip()
            if _is_plausible_baseline_o2_sat(value):
                out["baseline_o2_sat"] = value
    return out


def _normalize_o2_sat_value(value: str) -> str:
    m = re.search(r"(\d{2,3})", str(value or ""))
    return str(int(m.group(1))) if m else ""


def _replace_baseline_o2_sat_value(text: str, desired_value: str) -> str:
    desired_num = _normalize_o2_sat_value(desired_value)
    if not desired_num:
        return text
    desired_token = f"{desired_num} %"
    pat = re.compile(
        r"(?i)(\b(?:baseline\s*o\s*2\s*sat|baseline\s*oxygen\s*saturation|o\s*2\s*sat)\b\s*[:=|]?\s*)(%?\s*\d{2,3}\s*%?)"
    )

    def _repl(m: re.Match[str]) -> str:
        prefix = str(m.group(1) or "")
        return f"{prefix}{desired_token}"

    out, n = pat.subn(_repl, str(text or ""), count=1)
    return out if n > 0 else text


def _merge_post_ground_truth_with_safe_prefill(post_text: str, pre_redacted_text: str) -> str:
    post = str(post_text or "").strip()
    pre = str(pre_redacted_text or "").strip()
    if not post:
        return pre
    post_values = _extract_safe_measurements(post)
    pre_values = _extract_safe_measurements(pre)
    # Baseline O2 Sat can drift by OCR (e.g., 95 -> 96). Since this field is not PHI,
    # prefer the pre-redaction extracted value when both exist but disagree.
    if "baseline_o2_sat" in pre_values and "baseline_o2_sat" in post_values:
        pre_o2 = _normalize_o2_sat_value(pre_values["baseline_o2_sat"])
        post_o2 = _normalize_o2_sat_value(post_values["baseline_o2_sat"])
        if pre_o2 and post_o2 and pre_o2 != post_o2:
            post = _replace_baseline_o2_sat_value(post, pre_values["baseline_o2_sat"])
            post_values["baseline_o2_sat"] = pre_values["baseline_o2_sat"]

    missing: Dict[str, str] = {}
    for key in ("height", "weight", "pre_exercise_hr", "baseline_o2_sat"):
        if key in pre_values and key not in post_values:
            missing[key] = pre_values[key]
    if not missing:
        return post

    additions: List[str] = []
    if "height" in missing:
        additions.append(f"Height: {missing['height']}")
    if "weight" in missing:
        additions.append(f"Weight: {missing['weight']}")
    if "pre_exercise_hr" in missing:
        additions.append(f"Pre Exercise HR: {missing['pre_exercise_hr']}")
    if "baseline_o2_sat" in missing:
        additions.append(f"Baseline O2 Sat: {missing['baseline_o2_sat']}")
    if not additions:
        return post

    suffix = "Supplemental Measurements (Recovered)\n" + "\n".join(additions)
    return f"{post}\n\n{suffix}"


def _build_redacted_ocr_payload(manifest_path: Path, safe_file_id: str) -> Optional[Dict[str, object]]:
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    manual_enabled = bool((manifest.get("manual_redaction", {}) or {}).get("enabled", False))
    source_pdf_page_text: Dict[int, str] = {}
    source_file_raw = str(manifest.get("source_file", "") or "").strip()
    if source_file_raw:
        source_file_path = Path(source_file_raw)
        if source_file_path.exists() and source_file_path.suffix.lower() == ".pdf":
            source_pdf_page_text = _extract_pdf_text_by_page(source_file_path)
    post_redaction_page_text = _extract_post_redaction_text_by_page(manifest_path, manifest)

    pages_out: List[Dict[str, object]] = []
    for page in manifest.get("pages", []) or []:
        page_index = int(page.get("page_index", 0))
        # Pre-redaction source text is still used as a constrained fallback source.
        pre_redaction_page_text = str(
            page.get("source_text_for_export", "")
            or page.get("source_text_for_dates", "")
            or page.get("text", "")
            or source_pdf_page_text.get(page_index, "")
            or page.get("normalized_text_for_export", "")
            or ""
        )
        effective_spans = list(page.get("effective_redaction_spans", []) or page.get("pii_spans", []) or [])
        redaction_boxes_src = list(page.get("redaction_boxes", []) or [])
        terms_from_final_boxes: List[tuple[str, str]] = []
        for box in redaction_boxes_src:
            box_text = str(box.get("text", "") or "").strip()
            if not box_text:
                continue
            box_tag = str(box.get("tag", "") or "").strip().lower()
            terms_from_final_boxes.append((box_text, box_tag))
        # Apply automatic/effective spans first, then apply final box terms as an
        # additive pass. This keeps output aligned with the reviewed redaction state
        # while preserving as much non-PHI extracted text as possible.
        span_max_end = -1
        for s in effective_spans:
            try:
                span_max_end = max(span_max_end, int(s.get("end", -1)))
            except Exception:
                continue
        spans_compatible = bool(pre_redaction_page_text) and (span_max_end <= len(pre_redaction_page_text))
        pre_redacted_text = (
            _safe_redact_text_from_spans(pre_redaction_page_text, effective_spans)
            if spans_compatible
            else str(pre_redaction_page_text or "")
        )
        if terms_from_final_boxes:
            pre_redacted_text = _safe_redact_text_from_terms(pre_redacted_text, terms_from_final_boxes)

        post_redacted_text = str(post_redaction_page_text.get(page_index, "") or "")
        if post_redacted_text:
            # Post-redaction OCR is the ground truth; only backfill strictly-safe values.
            if terms_from_final_boxes:
                post_redacted_text = _safe_redact_text_from_terms(post_redacted_text, terms_from_final_boxes)
            redacted_text = _merge_post_ground_truth_with_safe_prefill(post_redacted_text, pre_redacted_text)
        else:
            redacted_text = pre_redacted_text

        readable_notes = _format_readable_notes(redacted_text)
        pages_out.append(
            {
                "page_index": page_index,
                "notes": readable_notes,
                "manual_redaction_applied": bool(manual_enabled),
            }
        )

    combined_notes_parts: List[str] = []
    for p in pages_out:
        page_num = int(p.get("page_index", 0)) + 1
        page_notes = str(p.get("notes", "") or "").strip()
        if not page_notes:
            continue
        combined_notes_parts.append(f"Page {page_num}\n{page_notes}")

    return {
        "schema_version": "redacted_ocr_notes_v2",
        "file_id": safe_file_id,
        "total_pages": len(pages_out),
        "created_at_utc": _utc_now_iso(),
        "notes": "\n\n".join(combined_notes_parts).strip(),
        "pages": pages_out,
    }


def _upload_redacted_ocr_payload_via_api(
    *,
    site_id: str,
    reviewer: str,
    safe_file_id: str,
    modality_type: str,
    redacted_ocr_payload: Dict[str, object],
    presign_api_url: str,
    complete_api_url: str,
    api_key: Optional[str] = None,
) -> str:
    modality_norm = _normalize_modality_for_upload(modality_type)
    ocr_presign_payload: Dict[str, object] = {
        "site_id": site_id,
        "file_id": safe_file_id,
        "report_id": safe_file_id,
        "modality_type": modality_norm,
        "modality": modality_norm,
        "filename": f"{safe_file_id}.redacted_ocr.json",
        "content_type": "application/json",
        "expires_seconds": 3600,
    }
    ocr_presign_resp = request_presigned_upload_via_api(
        presign_api_url,
        ocr_presign_payload,
        api_key=api_key,
    )
    ocr_upload_url = str(ocr_presign_resp.get("upload_url", ""))
    ocr_object_key = str(ocr_presign_resp.get("object_key", ""))
    ocr_required_headers = ocr_presign_resp.get("required_headers", {}) or {}
    if not ocr_upload_url or not ocr_object_key:
        raise RuntimeError(f"Invalid OCR presign response for file_id={safe_file_id}: {ocr_presign_resp}")

    upload_bytes_with_presigned_url(
        ocr_upload_url,
        json.dumps(redacted_ocr_payload, ensure_ascii=False).encode("utf-8"),
        required_headers=ocr_required_headers if isinstance(ocr_required_headers, dict) else {},
    )

    # The completion lambda de-duplicates at file_id level, so OCR and PDF can collide.
    ocr_complete_payload: Dict[str, object] = {
        "site_id": site_id,
        "object_key": ocr_object_key,
        "file_type": "json",
        "reviewer": reviewer,
        "file_id": safe_file_id,
        "report_id": safe_file_id,
        "doc_id": safe_file_id,
    }
    try:
        request_presigned_upload_via_api(
            complete_api_url,
            ocr_complete_payload,
            api_key=api_key,
        )
    except requests.HTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status != 409:
            raise
    return ocr_object_key


def backfill_redacted_ocr_via_api(
    tracker_csv: str,
    *,
    site_id: str,
    reviewer: str,
    presign_api_url: str,
    complete_api_url: str,
    api_key: Optional[str] = None,
    approved_only: bool = True,
    only_missing_uploads: bool = True,
    write_local_payload_files: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, int]:
    tracker_df = load_tracker(tracker_csv)
    work_df = tracker_df.copy()

    if approved_only:
        approved_num = pd.to_numeric(work_df.get("approved_to_send", 0), errors="coerce").fillna(0)
        work_df = work_df[approved_num.eq(1)].copy()
    if only_missing_uploads:
        uploaded_col = work_df.get("redacted_ocr_uploaded_at_utc", pd.Series("", index=work_df.index))
        work_df = work_df[uploaded_col.fillna("").astype(str).str.strip().eq("")].copy()

    if work_df.empty:
        return {
            "eligible_rows": 0,
            "generated_payloads": 0,
            "uploaded_payloads": 0,
            "skipped_missing_manifest": 0,
            "skipped_invalid_manifest": 0,
            "skipped_already_uploaded": 0,
            "errors": 0,
        }

    generated_payloads = 0
    uploaded_payloads = 0
    skipped_missing_manifest = 0
    skipped_invalid_manifest = 0
    skipped_already_uploaded = 0
    errors = 0
    total_rows = len(work_df)
    tracker_dir = Path(tracker_csv).expanduser().resolve().parent
    local_payload_dir = tracker_dir / "approved_for_transfer" / "ocr"

    for done_rows, (idx, row) in enumerate(work_df.iterrows(), start=1):
        safe_file_id = _normalize_file_id(
            str(row.get("force_id", "") or row.get("patient_id", "") or ""),
            str(row.get("study_date", "") or ""),
            str(row.get("file_id", "") or row.get("report_id", "") or ""),
        )
        modality_norm = _normalize_modality_for_upload(row.get("modality_type", ""))
        manifest_path = Path(str(row.get("manifest_json", "") or ""))
        if not manifest_path.exists():
            skipped_missing_manifest += 1
            if progress_callback:
                progress_callback(f"[{done_rows}/{total_rows}] Skipping {safe_file_id}: missing manifest.")
            continue
        redacted_ocr_payload = _build_redacted_ocr_payload(manifest_path, safe_file_id)
        if redacted_ocr_payload is None:
            skipped_invalid_manifest += 1
            if progress_callback:
                progress_callback(f"[{done_rows}/{total_rows}] Skipping {safe_file_id}: invalid manifest.")
            continue

        generated_payloads += 1
        tracker_df.loc[idx, "redacted_ocr_generated_at_utc"] = _utc_now_iso()

        if write_local_payload_files:
            local_payload_dir.mkdir(parents=True, exist_ok=True)
            payload_path = local_payload_dir / f"{safe_file_id}.redacted_ocr.json"
            payload_path.write_text(
                json.dumps(redacted_ocr_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tracker_df.loc[idx, "redacted_ocr_json"] = str(payload_path)

        uploaded_raw = tracker_df.loc[idx, "redacted_ocr_uploaded_at_utc"]
        uploaded_text = "" if pd.isna(uploaded_raw) else str(uploaded_raw).strip()
        already_uploaded = uploaded_text != ""
        if only_missing_uploads and already_uploaded:
            skipped_already_uploaded += 1
            if progress_callback:
                progress_callback(f"[{done_rows}/{total_rows}] Skipping upload for {safe_file_id}: already marked uploaded.")
            continue

        try:
            if progress_callback:
                progress_callback(f"[{done_rows}/{total_rows}] Uploading redacted OCR JSON for {safe_file_id}...")
            ocr_object_key = _upload_redacted_ocr_payload_via_api(
                site_id=site_id,
                reviewer=reviewer,
                safe_file_id=safe_file_id,
                modality_type=modality_norm,
                redacted_ocr_payload=redacted_ocr_payload,
                presign_api_url=presign_api_url,
                complete_api_url=complete_api_url,
                api_key=api_key,
            )
            uploaded_payloads += 1
            tracker_df.loc[idx, "redacted_ocr_uploaded_at_utc"] = _utc_now_iso()
            tracker_df.loc[idx, "redacted_ocr_s3_key"] = ocr_object_key
        except Exception as exc:
            errors += 1
            if progress_callback:
                progress_callback(f"[{done_rows}/{total_rows}] Failed {safe_file_id}: {exc}")

    save_tracker(tracker_df, tracker_csv)
    return {
        "eligible_rows": total_rows,
        "generated_payloads": generated_payloads,
        "uploaded_payloads": uploaded_payloads,
        "skipped_missing_manifest": skipped_missing_manifest,
        "skipped_invalid_manifest": skipped_invalid_manifest,
        "skipped_already_uploaded": skipped_already_uploaded,
        "errors": errors,
    }


def backfill_redacted_ocr_local(
    tracker_csv: str,
    *,
    output_dir: str,
    approved_only: bool = True,
    only_missing_local_files: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, int]:
    tracker_df = load_tracker(tracker_csv)
    work_df = tracker_df.copy()

    if approved_only:
        approved_num = pd.to_numeric(work_df.get("approved_to_send", 0), errors="coerce").fillna(0)
        work_df = work_df[approved_num.eq(1)].copy()

    out_root = Path(output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if only_missing_local_files:
        keep_rows = []
        for idx, row in work_df.iterrows():
            existing_path_raw = str(row.get("redacted_ocr_json", "") or "").strip()
            if not existing_path_raw:
                keep_rows.append(idx)
                continue
            existing_path = Path(existing_path_raw)
            if not existing_path.exists():
                keep_rows.append(idx)
        work_df = work_df.loc[keep_rows].copy() if keep_rows else work_df.iloc[0:0].copy()

    if work_df.empty:
        return {
            "eligible_rows": 0,
            "generated_payloads": 0,
            "stored_payloads": 0,
            "skipped_missing_manifest": 0,
            "skipped_invalid_manifest": 0,
            "skipped_existing_local": 0,
            "errors": 0,
        }

    generated_payloads = 0
    stored_payloads = 0
    skipped_missing_manifest = 0
    skipped_invalid_manifest = 0
    skipped_existing_local = 0
    errors = 0
    total_rows = len(work_df)

    for done_rows, (idx, row) in enumerate(work_df.iterrows(), start=1):
        safe_file_id = _normalize_file_id(
            str(row.get("force_id", "") or row.get("patient_id", "") or ""),
            str(row.get("study_date", "") or ""),
            str(row.get("file_id", "") or row.get("report_id", "") or ""),
        )
        output_path = out_root / f"{safe_file_id}.redacted_ocr.json"

        if only_missing_local_files and output_path.exists():
            skipped_existing_local += 1
            if progress_callback:
                progress_callback(f"[{done_rows}/{total_rows}] Skipping {safe_file_id}: local payload already exists.")
            continue

        manifest_path = Path(str(row.get("manifest_json", "") or ""))
        if not manifest_path.exists():
            skipped_missing_manifest += 1
            if progress_callback:
                progress_callback(f"[{done_rows}/{total_rows}] Skipping {safe_file_id}: missing manifest.")
            continue
        redacted_ocr_payload = _build_redacted_ocr_payload(manifest_path, safe_file_id)
        if redacted_ocr_payload is None:
            skipped_invalid_manifest += 1
            if progress_callback:
                progress_callback(f"[{done_rows}/{total_rows}] Skipping {safe_file_id}: invalid manifest.")
            continue

        generated_payloads += 1
        tracker_df.loc[idx, "redacted_ocr_generated_at_utc"] = _utc_now_iso()
        try:
            output_path.write_text(
                json.dumps(redacted_ocr_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tracker_df.loc[idx, "redacted_ocr_json"] = str(output_path)
            stored_payloads += 1
            if progress_callback:
                progress_callback(f"[{done_rows}/{total_rows}] Stored redacted OCR JSON for {safe_file_id}.")
        except Exception as exc:
            errors += 1
            if progress_callback:
                progress_callback(f"[{done_rows}/{total_rows}] Failed {safe_file_id}: {exc}")

    save_tracker(tracker_df, tracker_csv)
    return {
        "eligible_rows": total_rows,
        "generated_payloads": generated_payloads,
        "stored_payloads": stored_payloads,
        "skipped_missing_manifest": skipped_missing_manifest,
        "skipped_invalid_manifest": skipped_invalid_manifest,
        "skipped_existing_local": skipped_existing_local,
        "errors": errors,
    }


def generate_local_redacted_ocr_for_doc(
    tracker_csv: str,
    doc_id: str,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    tracker_df = load_tracker(tracker_csv)
    mask = tracker_df.get("doc_id", pd.Series("", index=tracker_df.index)).astype(str).eq(str(doc_id))
    if not bool(mask.any()):
        return None

    idx = tracker_df[mask].index[-1]
    row = tracker_df.loc[idx]
    safe_file_id = _normalize_file_id(
        str(row.get("force_id", "") or row.get("patient_id", "") or ""),
        str(row.get("study_date", "") or ""),
        str(row.get("file_id", "") or row.get("report_id", "") or ""),
    )
    manifest_path = Path(str(row.get("manifest_json", "") or ""))
    if not manifest_path.exists():
        return None
    redacted_ocr_payload = _build_redacted_ocr_payload(manifest_path, safe_file_id)
    if redacted_ocr_payload is None:
        return None

    if str(output_dir or "").strip():
        out_root = Path(output_dir).expanduser().resolve()
    else:
        tracker_dir = Path(tracker_csv).expanduser().resolve().parent
        out_root = tracker_dir / "approved_for_transfer" / "ocr"
    out_root.mkdir(parents=True, exist_ok=True)
    output_path = out_root / f"{safe_file_id}.redacted_ocr.json"
    output_path.write_text(
        json.dumps(redacted_ocr_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    tracker_df.loc[idx, "redacted_ocr_generated_at_utc"] = _utc_now_iso()
    tracker_df.loc[idx, "redacted_ocr_json"] = str(output_path)
    save_tracker(tracker_df, tracker_csv)
    return str(output_path)


def upload_approved_files_via_api(
    tracker_csv: str,
    *,
    site_id: str,
    reviewer: str,
    presign_api_url: str,
    complete_api_url: str,
    review_event_api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    extraction_session_id: Optional[str] = None,
    approved_session_id: Optional[str] = None,
    only_unsent: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[str]:
    tracker_df = load_tracker(tracker_csv)
    approved = tracker_df[tracker_df["approved_to_send"] == 1].copy()
    if only_unsent and "sent_to_aws" in approved.columns:
        approved = approved[pd.to_numeric(approved["sent_to_aws"], errors="coerce").fillna(0).ne(1)].copy()
    if approved_session_id:
        if "approved_session_id" not in approved.columns:
            approved = approved.iloc[0:0].copy()
        else:
            approved = approved[
                approved["approved_session_id"]
                .fillna("")
                .astype(str)
                .eq(str(approved_session_id))
            ].copy()
    if approved.empty:
        return []

    uploaded_keys: List[str] = []
    total_docs = len(approved)
    done_docs = 0
    for _, row in approved.iterrows():
        done_docs += 1
        redacted_file = Path(str(row.get("redacted_file", "")))
        if not redacted_file.exists():
            if progress_callback:
                progress_callback(f"[{done_docs}/{total_docs}] Skipping missing redacted file for doc.")
            continue
        safe_file_id = _normalize_file_id(
            str(row.get("force_id", "") or row.get("patient_id", "") or ""),
            str(row.get("study_date", "") or ""),
            str(row.get("file_id", "") or row.get("report_id", "") or ""),
        )
        modality_norm = _normalize_modality_for_upload(row.get("modality_type", ""))
        redacted_ocr_payload = None
        redacted_ocr_json_path = Path(str(row.get("redacted_ocr_json", "") or "").strip())
        if redacted_ocr_json_path.exists():
            try:
                loaded_payload = json.loads(redacted_ocr_json_path.read_text(encoding="utf-8"))
                if isinstance(loaded_payload, dict):
                    redacted_ocr_payload = loaded_payload
            except Exception:
                redacted_ocr_payload = None

        if redacted_ocr_payload is None:
            if progress_callback:
                progress_callback(
                    f"[{done_docs}/{total_docs}] Skipping {safe_file_id}: OCR payload missing/invalid. Generate at approval step (OK to Send)."
                )
            continue

        if progress_callback:
            progress_callback(f"[{done_docs}/{total_docs}] Requesting presigned URLs for PDF/OCR pair: {safe_file_id}...")

        pdf_presign_payload: Dict[str, object] = {
            "site_id": site_id,
            "file_id": safe_file_id,
            "report_id": safe_file_id,  # backward compatibility
            "modality_type": modality_norm,
            "modality": modality_norm,
            "filename": f"{safe_file_id}.pdf",
            "content_type": "application/pdf",
            "expires_seconds": 3600,
        }
        pdf_presign_resp = request_presigned_upload_via_api(
            presign_api_url,
            pdf_presign_payload,
            api_key=api_key,
        )
        pdf_upload_url = str(pdf_presign_resp.get("upload_url", ""))
        pdf_object_key = str(pdf_presign_resp.get("object_key", ""))
        pdf_required_headers = pdf_presign_resp.get("required_headers", {}) or {}
        if not pdf_upload_url or not pdf_object_key:
            raise RuntimeError(f"Invalid PDF presign response for doc_id={row.get('doc_id')}: {pdf_presign_resp}")

        ocr_presign_payload: Dict[str, object] = {
            "site_id": site_id,
            "file_id": safe_file_id,
            "report_id": safe_file_id,
            "modality_type": modality_norm,
            "modality": modality_norm,
            "filename": f"{safe_file_id}.redacted_ocr.json",
            "content_type": "application/json",
            "expires_seconds": 3600,
        }
        ocr_presign_resp = request_presigned_upload_via_api(
            presign_api_url,
            ocr_presign_payload,
            api_key=api_key,
        )
        ocr_upload_url = str(ocr_presign_resp.get("upload_url", ""))
        ocr_object_key = str(ocr_presign_resp.get("object_key", ""))
        ocr_required_headers = ocr_presign_resp.get("required_headers", {}) or {}
        if not ocr_upload_url or not ocr_object_key:
            raise RuntimeError(f"Invalid OCR presign response for doc_id={row.get('doc_id')}: {ocr_presign_resp}")

        if progress_callback:
            progress_callback(f"[{done_docs}/{total_docs}] Uploading PDF for {safe_file_id}...")
        try:
            upload_file_with_presigned_url(
                pdf_upload_url,
                str(redacted_file),
                required_headers=pdf_required_headers if isinstance(pdf_required_headers, dict) else {},
            )
        except Exception as exc:
            raise RuntimeError(f"Failed PUT upload for redacted PDF file_id={safe_file_id}: {exc}") from exc
        if progress_callback:
            progress_callback(f"[{done_docs}/{total_docs}] Uploading sanitized OCR JSON for {safe_file_id}...")
        try:
            upload_bytes_with_presigned_url(
                ocr_upload_url,
                json.dumps(redacted_ocr_payload, ensure_ascii=False).encode("utf-8"),
                required_headers=ocr_required_headers if isinstance(ocr_required_headers, dict) else {},
            )
        except Exception as exc:
            if progress_callback:
                progress_callback(
                    f"[{done_docs}/{total_docs}] Pair upload failed for {safe_file_id}: OCR upload error ({exc})."
                )
                progress_callback(
                    f"[{done_docs}/{total_docs}] Skipped completion for {safe_file_id}: enforcing PDF/OCR pairing."
                )
            continue

        pdf_complete_payload: Dict[str, object] = {
            "site_id": site_id,
            "object_key": pdf_object_key,
            "reviewer": reviewer,
            "file_id": safe_file_id,
            "report_id": safe_file_id,  # backward compatibility
            "doc_id": safe_file_id,  # backward compatibility
            "session_id": extraction_session_id or "",
            "modality_type": modality_norm,
            "file_type": "pdf",
            "force_id": str(row.get("force_id", "") or row.get("patient_id", "") or ""),
            "study_date": str(row.get("study_date", "") or ""),
            "raw_study_date": str(row.get("raw_study_date", "") or row.get("study_date", "") or ""),
            "dup": _safe_int(row.get("dup", 0)),
            "modality_instance": _safe_int(row.get("modality_instance", 1), 1),
        }
        ocr_complete_payload: Dict[str, object] = {
            "site_id": site_id,
            "object_key": ocr_object_key,
            "reviewer": reviewer,
            "file_id": safe_file_id,
            "report_id": safe_file_id,
            "doc_id": safe_file_id,
            "session_id": extraction_session_id or "",
            "modality_type": modality_norm,
            "file_type": "json",
            "force_id": str(row.get("force_id", "") or row.get("patient_id", "") or ""),
            "study_date": str(row.get("study_date", "") or ""),
            "raw_study_date": str(row.get("raw_study_date", "") or row.get("study_date", "") or ""),
            "dup": _safe_int(row.get("dup", 0)),
            "modality_instance": _safe_int(row.get("modality_instance", 1), 1),
        }
        if progress_callback:
            progress_callback(f"[{done_docs}/{total_docs}] Finalizing PDF/OCR pair for {safe_file_id}...")
        pair_complete_failed = False
        try:
            request_presigned_upload_via_api(
                complete_api_url,
                pdf_complete_payload,
                api_key=api_key,
            )
        except requests.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 409:
                if progress_callback:
                    progress_callback(
                        f"[{done_docs}/{total_docs}] PDF completion already finalized for {safe_file_id} (409 duplicate)."
                    )
            else:
                if progress_callback:
                    progress_callback(
                        f"[{done_docs}/{total_docs}] Pair completion failed for {safe_file_id}: PDF completion error ({exc})."
                    )
                pair_complete_failed = True
        try:
            request_presigned_upload_via_api(
                complete_api_url,
                ocr_complete_payload,
                api_key=api_key,
            )
        except requests.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 409:
                if progress_callback:
                    progress_callback(
                        f"[{done_docs}/{total_docs}] OCR completion already finalized for {safe_file_id} (409 duplicate)."
                    )
            else:
                if progress_callback:
                    progress_callback(
                        f"[{done_docs}/{total_docs}] Pair completion failed for {safe_file_id}: OCR completion error ({exc})."
                    )
                pair_complete_failed = True
        if pair_complete_failed:
            if progress_callback:
                progress_callback(
                    f"[{done_docs}/{total_docs}] Pair not marked successful for {safe_file_id}."
                )
            continue

        uploaded_keys.append(pdf_object_key)
        tracker_df.loc[row.name, "sent_to_aws"] = 1
        tracker_df.loc[row.name, "sent_to_aws_at_utc"] = _utc_now_iso()
        tracker_df.loc[row.name, "redacted_ocr_uploaded_at_utc"] = _utc_now_iso()
        tracker_df.loc[row.name, "redacted_ocr_s3_key"] = ocr_object_key

        if review_event_api_url:
            manifest_metrics = _extract_manifest_user_metrics(Path(str(row.get("manifest_json", ""))))
            is_eu_mode = _safe_int(row.get("eu_mode", 0), 0) == 1
            age_at_event = str(row.get("age_at_event", "") or "").strip()
            input_kind = _derive_input_kind_from_row(row)
            review_payload: Dict[str, object] = {
                "site_id": site_id,
                "file_id": safe_file_id,
                "report_id": safe_file_id,  # backward compatibility
                "doc_id": safe_file_id,  # backward compatibility
                "reviewer": reviewer,
                "review_status": str(row.get("review_status", "")),
                "review_notes": "",
                "approved_to_send": int(float(row.get("approved_to_send", 0) or 0)),
                "phi_found": int(float(row.get("phi_found", 0) or 0)),
                "total_pii_spans": int(float(row.get("total_pii_spans", 0) or 0)),
                "force_id": str(row.get("force_id", "") or row.get("patient_id", "") or ""),
                "modality_type": str(row.get("modality_type", "") or ""),
                # EU mode sends age only (no dates in outbound metadata).
                "study_date": "" if is_eu_mode else str(row.get("study_date", "") or ""),
                "raw_study_date": "" if is_eu_mode else str(row.get("raw_study_date", "") or row.get("study_date", "") or ""),
                "dup": _safe_int(row.get("dup", 0)),
                "modality_instance": _safe_int(row.get("modality_instance", 1), 1),
                "age_at_event": age_at_event if is_eu_mode else (age_at_event or ""),
                "eu_mode": int(is_eu_mode),
                "file_type": str(row.get("source_ext", "") or ""),
                "input_type": input_kind,
                "documents_in_file": _safe_int(row.get("total_pages", 0)),
                "auto_redaction_elements": _safe_int(row.get("total_redaction_boxes", 0)),
                "user_deleted_elements": manifest_metrics["user_deleted_box_count"],
                "user_added_elements": manifest_metrics["user_added_box_count"],
            }
            request_presigned_upload_via_api(
                review_event_api_url,
                review_payload,
                api_key=api_key,
            )

        if progress_callback:
            progress_callback(f"[{done_docs}/{total_docs}] Completed {safe_file_id}.")

    save_tracker(tracker_df, tracker_csv)
    return uploaded_keys

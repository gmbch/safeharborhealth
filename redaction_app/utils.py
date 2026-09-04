from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import time
import types
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import io
import importlib

import cv2
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    from pii_redaction import PIIHandlingMode, tag_pii_in_documents
except Exception:  # pragma: no cover
    PIIHandlingMode = None
    tag_pii_in_documents = None


logger = logging.getLogger("phi_redaction_pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_DOC_EXTS = {".pdf", *SUPPORTED_IMAGE_EXTS}

DEFAULT_DPI = 200
DEFAULT_PAD_RATIO_X = 0.001
DEFAULT_PAD_RATIO_Y = 0.001
DEFAULT_DEVICE = "cpu"
DEFAULT_DETECTOR_BACKEND = "nlp"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_FILENAME = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
DEFAULT_MODEL_URL = (
    "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/"
    "resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf?download=true"
)
TRACKER_COLUMNS = [
    "site_id",
    "source_file",
    "source_filename",
    "source_ext",
    "input_kind",
    "doc_id",
    "detector_backend",
    "ocr_backend",
    "pdf_text_mode",
    "redacted_file",
    "review_pages_dir",
    "manifest_json",
    "redacted_ocr_json",
    "redacted_ocr_generated_at_utc",
    "redacted_ocr_uploaded_at_utc",
    "redacted_ocr_s3_key",
    "total_pages",
    "total_pii_spans",
    "auto_redaction_elements",
    "total_redaction_boxes",
    "user_deleted_elements",
    "user_added_elements",
    "duration_sec",
    "phi_found",
    "review_status",
    "approved_to_send",
    "approved_session_id",
    "approved_at_utc",
    "sent_to_aws",
    "sent_to_aws_at_utc",
    "reviewer",
    "review_notes",
    "force_id",
    "file_id",
    "modality_instance",
    "first_name",
    "last_name",
    "mrn",
    "raw_dob",
    "dob",
    "gender",
    "dup",
    # Legacy compatibility column (do not rely on this for new writes).
    "patient_id",
    "modality_type",
    "raw_study_date",
    "study_date",
    "age_at_event",
    "eu_mode",
    "full_date_overlay_mode",
]


def classify_input_kind(file_path: Path) -> str:
    """
    Classify report source for downstream tracker/AWS metadata.
    """
    ext = file_path.suffix.lower()
    name = file_path.name.lower()
    parts_lower = [p.lower() for p in file_path.parts]
    if ext == ".pdf":
        if (
            any(p == ".event_group_inputs" for p in parts_lower)
            or name.endswith("__screenshots.pdf")
            or name.endswith(".screenshots.pdf")
        ):
            return "screenshot_report"
        return "pdf_report"
    if ext in SUPPORTED_IMAGE_EXTS:
        return "screenshot_report"
    if ext == ".txt":
        return "free_text"
    return "report"


def get_default_llama_prompt() -> str:
    return (
        "You are a compliance assistant.\n"
        "Your task is to check OCR text from radiology reports for PHI.\n"
        "PHI includes patient names (physician name is OK), MRNs, Study IDs, Event IDs, DOBs,\n"
        "and all dates/times (even if only a day value).\n"
        "You must EXHAUSTIVELY identify every PHI value present. Do not skip any values.\n\n"
        "Respond ONLY in one of these formats:\n"
        "SAFE\n"
        "or\n"
        "PHI DETECTED:\n"
        "- Name: <value or None>\n"
        "- MRN: <value or None>\n"
        "- Study ID: <value or None>\n"
        "- Event ID: <value or None>\n"
        "- DOB: <value or None>\n"
        "- Date: <value or None>\n"
        "- Time: <value or None>\n\n"
        "OCR text:\n{document_text}\n"
    )


def ensure_local_llama_model(model_path: Optional[str] = None) -> str:
    if model_path and str(model_path).strip():
        resolved = Path(model_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Configured local model not found: {resolved}")
        return str(resolved)

    ensure_dir(DEFAULT_MODEL_DIR)
    local_model = DEFAULT_MODEL_DIR / DEFAULT_MODEL_FILENAME
    if local_model.exists():
        return str(local_model)

    logger.info("Downloading local Llama model to %s", local_model)
    try:
        urllib.request.urlretrieve(DEFAULT_MODEL_URL, str(local_model))
    except Exception as exc:
        raise RuntimeError(
            f"Could not auto-download local model from {DEFAULT_MODEL_URL}. "
            "Please place a GGUF model under src/models and retry."
        ) from exc

    return str(local_model)


@dataclass
class OCRLine:
    text: str
    confidence: float
    polygon: List[List[float]]
    bbox_xyxy: Tuple[float, float, float, float]
    char_start: int
    char_end: int


@dataclass
class OCRPage:
    page_index: int
    width: int
    height: int
    text: str
    lines: List[OCRLine]
    extraction_method: str


@dataclass
class PIISpan:
    start: int
    end: int
    tag: str
    text: str
    source: str = ""
    reason: str = ""


@dataclass
class RedactionBox:
    page_index: int
    tag: str
    text: str
    bbox_xyxy: Tuple[float, float, float, float]


class BasePHIDetector:
    detector_name = "base"

    def detect(self, text: str, device: Optional[str] = None) -> List[PIISpan]:
        raise NotImplementedError


class LlamaPHIDetector(BasePHIDetector):
    detector_name = "openpipe_llama"

    def __init__(self) -> None:
        if tag_pii_in_documents is None or PIIHandlingMode is None:
            raise RuntimeError(
                "pii-redaction is unavailable. Install dependency 'pii-redaction'."
            )

    def detect(self, text: str, device: Optional[str] = None) -> List[PIISpan]:
        if not text.strip():
            return []
        tagged = tag_pii_in_documents([text], device=device, mode=PIIHandlingMode.TAG)[0]
        aligned = align_pii_spans_to_ocr_text(text, tagged)
        for span in aligned:
            if not span.text:
                span.text = text[span.start:span.end]
        return aligned


class LocalPromptLlamaDetector(BasePHIDetector):
    """
    Local prompt-driven detector using llama.cpp.
    Requires `llama-cpp-python` and a local GGUF model file.
    """

    detector_name = "local_llama_prompt"

    DEFAULT_PROMPT = get_default_llama_prompt()

    def __init__(self, model_path: Optional[str] = None, prompt_template: Optional[str] = None) -> None:
        configured = (model_path or os.getenv("LLAMA_MODEL_PATH", "")).strip()
        self.model_path = ensure_local_llama_model(configured if configured else None)
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Missing dependency 'llama-cpp-python'. Install it to use AI-based local prompt model."
            ) from exc

        self._llm = Llama(model_path=self.model_path, n_ctx=4096, verbose=False)
        self.prompt_template = (prompt_template or self.DEFAULT_PROMPT).strip()

    def _parse_model_output(self, raw: str) -> List[Tuple[str, str]]:
        entities: List[Tuple[str, str]] = []
        text = raw.strip()
        try:
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                payload = json.loads(text[start : end + 1])
                if isinstance(payload, list):
                    for item in payload:
                        if not isinstance(item, dict):
                            continue
                        tag = str(item.get("tag", "")).strip().lower()
                        value = str(item.get("text", "")).strip()
                        if tag and value:
                            entities.append((tag, value))
        except Exception:
            pass

        if entities:
            return entities

        # Fallback for compliance-style bullet output:
        # PHI DETECTED:
        # - Name: ...
        label_to_tag = {
            "name": "patient_name",
            "mrn": "mrn",
            "study id": "study_id",
            "event id": "event_id",
            "dob": "dob",
            "date": "date",
            "time": "time",
        }
        for line in text.splitlines():
            clean = line.strip().lstrip("-").strip()
            if ":" not in clean:
                continue
            label, value = clean.split(":", 1)
            label_norm = label.strip().lower()
            value = value.strip()
            if value.lower() in {"none", "n/a", ""}:
                continue
            if label_norm in label_to_tag:
                entities.append((label_to_tag[label_norm], value))

        if entities:
            return entities

        # Fallback: parse "tag|value" lines if model did not output strict JSON.
        for line in text.splitlines():
            if "|" not in line:
                continue
            tag, value = line.split("|", 1)
            tag = tag.strip().lower()
            value = value.strip()
            if tag and value:
                entities.append((tag, value))
        return entities

    def detect(self, text: str, device: Optional[str] = None) -> List[PIISpan]:
        if not text.strip():
            return []

        prompt = self.prompt_template.replace("{document_text}", text)
        response = self._llm.create_completion(
            prompt=prompt,
            max_tokens=768,
            temperature=0.0,
            stop=["</s>"],
        )
        raw = str(response["choices"][0]["text"])
        entities = self._parse_model_output(raw)

        spans: List[PIISpan] = []
        for tag, value in entities:
            pattern = re.compile(re.escape(value), flags=re.IGNORECASE)
            for match in pattern.finditer(text):
                spans.append(
                    PIISpan(
                        start=match.start(),
                        end=match.end(),
                        tag=tag,
                        text=match.group(0),
                    )
                )
        return merge_overlapping_spans(spans)


class NameOnlyLlamaDetector(LocalPromptLlamaDetector):
    """
    Llama detector restricted to patient names only.
    """

    detector_name = "local_llama_name_only"
    NAME_ONLY_PROMPT = (
        "You are a compliance assistant.\n"
        "Extract ONLY patient names from radiology report OCR text.\n"
        "Do NOT include clinician/doctor/specialist names.\n"
        "Return ONLY JSON list of {\"tag\":\"patient_name\",\"text\":\"...\"}.\n"
        "If none found, return []\n"
        "OCR text:\n{document_text}\n"
    )
    CLINICIAN_CONTEXT = re.compile(
        r"\b(?:dr\.?|doctor|physician|cardiologist|surgeon|interpreter|interpreting|reading)\b",
        flags=re.IGNORECASE,
    )
    NON_PATIENT_CONTEXT = re.compile(
        r"\b(?:medication|comments|indications|stage|duration|speed|workload|hr|bp|exercise|test|abnormalit(?:y|ies)|chest\s+pain|arrhythmia)\b",
        flags=re.IGNORECASE,
    )
    NAME_SHAPE = re.compile(r"^[A-Za-z][A-Za-z'`-]+(?:[ ,]+[A-Za-z][A-Za-z'`-]+){1,3}$")
    NON_NAME_TOKENS = {
        "digoxin",
        "carvedilol",
        "lisinopril",
        "aspirin",
        "amitriptyline",
        "stage",
        "duration",
        "speed",
        "workload",
        "exercise",
        "test",
        "medication",
        "comments",
        "indications",
        "cardiology",
        "department",
        "hospital",
        "patient",
        "name",
        "demographics",
    }
    PATIENT_NEARBY_CONTEXT = re.compile(
        r"\b(?:patient\s*name|mrn|medical\s+record|record\s*(?:#|no\.?|number)|dob|date\s*of\s*birth|born)\b",
        flags=re.IGNORECASE,
    )

    def __init__(self, model_path: Optional[str] = None, prompt_template: Optional[str] = None) -> None:
        super().__init__(
            model_path=model_path,
            prompt_template=prompt_template or self.NAME_ONLY_PROMPT,
        )

    def detect(self, text: str, device: Optional[str] = None) -> List[PIISpan]:
        spans = super().detect(text, device=device)
        filtered: List[PIISpan] = []
        for span in spans:
            if span.tag != "patient_name":
                continue
            value = (span.text or text[span.start:span.end]).strip()
            if value.lower() in {"patient name", "name", "demographics"}:
                continue
            if not self.NAME_SHAPE.match(value):
                continue
            low = value.lower()
            if any(tok in low for tok in self.NON_NAME_TOKENS):
                continue
            snippet = text[max(0, span.start - 40) : min(len(text), span.end + 40)]
            if self.CLINICIAN_CONTEXT.search(snippet):
                continue
            if self.NON_PATIENT_CONTEXT.search(snippet):
                continue
            wide = text[max(0, span.start - 180) : min(len(text), span.end + 180)]
            if not self.PATIENT_NEARBY_CONTEXT.search(wide):
                continue
            # Reject candidates that are mostly all-caps shorthand/symbol-like fragments.
            parts = [p for p in re.split(r"[,\s]+", value) if p]
            if any(len(p) <= 2 for p in parts):
                continue
            filtered.append(
                PIISpan(
                    start=span.start,
                    end=span.end,
                    tag="patient_name",
                    text=value,
                    source=self.detector_name,
                    reason="llama_name_only",
                )
            )
        return merge_overlapping_spans(filtered)


class DeterministicCorePHIDetector(BasePHIDetector):
    """
    Focused deterministic detector:
    - patient name (strict formats only)
    - MRN
    - date/DOB (later reduced to day-only spans by effective span builder)
    """

    detector_name = "deterministic_core"

    NAME_PATTERNS: List[re.Pattern[str]] = [
        # "Patient Name: John Smith" or "Name: DOE, JOHN"
        re.compile(
            r"\b(?:Patient\s*Name|Name)\s*[:\-]\s*"
            r"([A-Z][A-Za-z'`-]+(?:,\s*[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+)*)?"
            r"|[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+){1,3}"
            r"|[A-Z]{2,}(?:\s+[A-Z]{2,}){1,3})\b"
        ),
        # Table-like label format with no colon: "Patient Name    John Smith"
        # Keep this strict to "Patient Name" only (not generic "Name") to avoid false positives.
        re.compile(
            r"\b(?:Patient\s*Name)\s+"
            r"([A-Z][A-Za-z'`-]+(?:,\s*[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+)*)?"
            r"|[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+){1,3}"
            r"|[A-Z]{2,}(?:\s+[A-Z]{2,}){1,3})\b"
        ),
        # Footer-like "Lastname, Firstname (MRN)"
        re.compile(
            r"\b([A-Z][A-Za-z'`-]+,\s*[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+)*)\s*\(\s*[A-Za-z0-9-]{3,}\s*\)"
        ),
        # "Lastname, Firstname" immediately followed by MRN label in nearby text
        re.compile(
            r"\b([A-Z][A-Za-z'`-]+,\s*[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+)*)\b"
        ),
    ]
    NAME_EXCLUDE_CONTEXT = re.compile(
        r"\b(?:dr\.?|doctor|physician|cardiologist|surgeon|interpreter|interpreting|reading|referring|ordering)\b",
        flags=re.IGNORECASE,
    )

    MRN_PATTERNS: List[re.Pattern[str]] = [
        # "MRN: 12345"
        re.compile(
            r"\b(?:MRN|Medical\s+Record\s+Number)\s*[:#\-]?\s*([A-Za-z0-9\-]{4,})\b",
            flags=re.IGNORECASE,
        ),
        # "MR # 12345" / "MR No 12345" / "M.R. # 12345"
        re.compile(
            r"\b(?:MR|M\.?\s*R\.?)\s*(?:#|No\.?|Number)\s*[:#\-]?\s*([A-Za-z0-9\-]{3,})\b",
            flags=re.IGNORECASE,
        ),
        # Explicit compact label variant: "MR#: 12345" / "MR#12345"
        re.compile(
            r"\b(?:MR|M\.?\s*R\.?)\s*#\s*[:\-]?\s*([A-Za-z0-9\-]{2,})\b",
            flags=re.IGNORECASE,
        ),
        # "Record #: 12345" / "Record No: 12345" / "Medical Record #: 12345"
        re.compile(
            r"\b(?:Rec(?:o|0|O)rd|Medical\s+Rec(?:o|0|O)rd)\s*(?:Number|No\.?|#)?\s*[:#\-]?\s*([A-Za-z0-9\-]{4,})\b",
            flags=re.IGNORECASE,
        ),
        # "(12345)" in patient footer context
        re.compile(r"\(\s*([A-Za-z0-9-]{4,})\s*\)"),
    ]

    DATE_PATTERNS: List[re.Pattern[str]] = [
        # Common numeric formats
        re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"),
        # "Apr 23, 2024"
        re.compile(r"\b[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}\b"),
        # OCR-noisy month-name forms, including punctuation/newline-fragmented day tokens.
        re.compile(
            r"\b[A-Za-z]{3,9}[\s\.,:/-]*-?[0-9Il]{1,2}[\s\.,:/-]+\d{4}\b",
            flags=re.IGNORECASE,
        ),
        # "23-Apr-2024"
        re.compile(r"\b\d{1,2}-[A-Za-z]{3}-\d{4}\b"),
    ]
    PATIENT_NAME_LINE_PATTERN = re.compile(
        r"(?im)^\s*patient\s*name\s*[:\-]?\s*(?P<val>[^\n\r]{2,80})$"
    )
    PATIENT_NAME_INLINE_PATTERN = re.compile(
        r"\bpatient\s*name\s*[:\-]?\s*(?P<val>[A-Za-z][A-Za-z ,.'`-]{2,80})",
        flags=re.IGNORECASE,
    )
    DEMOGRAPHIC_SUMMARY_NAME_PATTERN = re.compile(
        r"(?im)^\s*(?P<name>[A-Z][A-Za-z'`.-]+(?:[ ,]+[A-Z][A-Za-z'`.-]+){0,2})\s*(?:,|-)?\s*"
        r"\d{1,3}\s*y\.?\s*o\.?\s*(?:male|female)\b[^\n\r]{0,48}\bborn\b"
    )
    MRN_LABEL = re.compile(r"\b(?:MRN|Medical\s+Record\s+Number)\b", flags=re.IGNORECASE)
    RECORD_LABEL = re.compile(
        r"\b(?:Rec(?:o|0|O)rd|Medical\s+Rec(?:o|0|O)rd)\s*(?:Number|No\.?|#)?\b",
        flags=re.IGNORECASE,
    )
    FOOTER_STAMP_PATTERN = re.compile(
        r"\b(?P<name>[A-Z][A-Za-z'`-]+(?:,\s*[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+)*|\s+[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+)*))\s*"
        r"\(\s*(?P<mrn>[A-Za-z0-9\- ]{3,})\s*\)\b",
        flags=re.IGNORECASE,
    )
    FOOTER_STAMP_WITH_TRAILER_PATTERN = re.compile(
        r"\b(?P<name>[A-Z][A-Za-z'`-]+(?:,\s*[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+)*|\s+[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+)*))\s*"
        r"\(\s*(?P<mrn>[A-Za-z0-9\- ]{3,})\s*\)\s*(?:study\s*date|page)\b",
        flags=re.IGNORECASE,
    )
    PHONE_PATTERN = re.compile(
        r"^(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})$"
    )
    PHONE_IN_CONTEXT_PATTERN = re.compile(
        r"\(\s*\d{3}\s*\)\s*\d{3}[-.\s]?\d{4}"
    )
    PHONE_CONTEXT = re.compile(
        r"\b(?:phone|tel|telephone|fax|contact|call)\b",
        flags=re.IGNORECASE,
    )
    DIAGNOSIS_CONTEXT = re.compile(
        r"\b(?:diagnos(?:is|es)|reason\s+for\s+study|impression|assessment|conclusion|indication|findings)\b",
        flags=re.IGNORECASE,
    )
    NON_PATIENT_NAME_TOKENS = {
        "cardiology",
        "cardiac",
        "mri",
        "cmr",
        "ct",
        "echo",
        "cath",
        "stresstest",
        "stress",
        "test",
        "exam",
        "study",
        "report",
        "impression",
        "findings",
        "procedure",
        "department",
        "clinic",
        "hospital",
        "medicine",
        "medical",
        "center",
        "centre",
        "radiology",
        "service",
        "program",
        "team",
    }
    US_STATE_ABBR = {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN",
        "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ",
        "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA",
        "WI", "WV", "WY",
    }

    def _looks_like_phone(self, value: str) -> bool:
        candidate = value.strip()
        if not candidate:
            return False
        if self.PHONE_PATTERN.match(candidate):
            return True
        digits = re.sub(r"\D", "", candidate)
        return len(digits) == 10

    def _valid_mrn(self, value: str, context_snippet: str) -> bool:
        raw = (value or "").strip()
        # Guardrail: reject date-like or symbol-heavy tokens as MRN.
        if re.search(r"[^\w\- ]", raw):
            return False
        if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", raw):
            return False
        compact = re.sub(r"\s+", "", raw)
        if len(compact) < 4:
            return False
        # Avoid false-positive MRN from phone numbers.
        if self._looks_like_phone(compact):
            return False
        if self.PHONE_IN_CONTEXT_PATTERN.search(context_snippet or ""):
            return False
        if self.PHONE_CONTEXT.search(context_snippet):
            return False
        has_mrn_label = bool(
            self.MRN_LABEL.search(context_snippet or "") or self.RECORD_LABEL.search(context_snippet or "")
        )
        if self.DIAGNOSIS_CONTEXT.search(context_snippet or "") and not has_mrn_label:
            # Avoid redacting diagnosis/problem-list codes that happen to be numeric.
            return False
        # MRN should include at least one digit and usually be non-trivial.
        if not re.search(r"\d", compact):
            return False
        # Prevent short numeric tokens (for example phone area code) from being treated as MRN.
        alnum = re.sub(r"[^A-Za-z0-9]", "", compact)
        if alnum.isdigit() and len(alnum) < 5:
            return False
        return len(re.sub(r"[^A-Za-z0-9]", "", compact)) >= 3

    def _looks_like_non_patient_name(self, value: str) -> bool:
        candidate = (value or "").strip()
        if not candidate:
            return True
        low = candidate.lower()
        if any(tok in low for tok in self.NON_PATIENT_NAME_TOKENS):
            return True
        tokens = [t for t in re.split(r"[\s,]+", candidate) if t]
        if len(tokens) < 2:
            return True
        # Common false positive: location tokens like "BOSTON, MA".
        if "," in candidate and len(tokens) >= 2:
            tail = re.sub(r"[^A-Za-z]", "", tokens[-1]).upper()
            if len(tail) == 2 and tail in self.US_STATE_ABBR:
                return True
        return False

    def _propagate_repeated_literals(self, text: str, spans: List[PIISpan]) -> List[PIISpan]:
        propagated: List[PIISpan] = list(spans)
        seen = {(s.tag, s.start, s.end) for s in spans}
        literals: List[Tuple[str, str]] = []
        for s in spans:
            lit = (s.text or text[s.start:s.end]).strip()
            if not lit:
                continue
            if s.tag == "mrn":
                literals.append(("mrn", lit))
            elif s.tag == "patient_name" and ("," in lit or len(lit.split()) >= 2):
                literals.append(("patient_name", lit))

        for tag, lit in literals:
            pattern = re.compile(re.escape(lit), flags=re.IGNORECASE)
            for match in pattern.finditer(text):
                key = (tag, match.start(), match.end())
                if key in seen:
                    continue
                seen.add(key)
                propagated.append(
                    PIISpan(
                        start=match.start(),
                        end=match.end(),
                        tag=tag,
                        text=text[match.start():match.end()],
                        source=self.detector_name,
                        reason=f"deterministic_repeat_{tag}",
                    )
                )
        return propagated

    def detect(self, text: str, device: Optional[str] = None) -> List[PIISpan]:
        if not text:
            return []

        spans: List[PIISpan] = []

        # Highest-confidence table/header pattern for many stress-test style reports.
        for m in self.PATIENT_NAME_LINE_PATTERN.finditer(text):
            vs, ve = m.start("val"), m.end("val")
            raw = text[vs:ve]
            # Trim trailing neighboring fields on the same OCR line if present.
            raw = re.split(
                r"\b(?:study\s*time|signed\s*on|date\s*of\s*birth|dob|medical\s*record|mrn)\b",
                raw,
                flags=re.IGNORECASE,
            )[0].strip(" ,;:-")
            if not raw:
                continue
            # Keep person-like names only.
            if self._looks_like_non_patient_name(raw):
                continue
            # Map back to the original text offsets for redaction.
            sub = text[m.start("val"):m.end("val")]
            rel = sub.lower().find(raw.lower())
            if rel < 0:
                start = m.start("val")
                end = m.end("val")
            else:
                start = m.start("val") + rel
                end = start + len(raw)
            spans.append(
                PIISpan(
                    start=start,
                    end=end,
                    tag="patient_name",
                    text=text[start:end],
                    source=self.detector_name,
                    reason="deterministic_patient_name_line",
                )
            )

        # Inline/table OCR fallback where rows may be merged into long lines.
        for m in self.PATIENT_NAME_INLINE_PATTERN.finditer(text):
            vs, ve = m.start("val"), m.end("val")
            raw = text[vs:ve]
            raw = re.split(
                r"\b(?:study\s*time|signed\s*on|date\s*of\s*birth|dob|medical\s*record|mrn|age\s*at\s*study|gender)\b",
                raw,
                flags=re.IGNORECASE,
            )[0].strip(" ,;:-")
            if not raw:
                continue
            if raw.lower() in {"patient name", "name", "demographics"}:
                continue
            if self._looks_like_non_patient_name(raw):
                continue
            sub = text[m.start("val"):m.end("val")]
            rel = sub.lower().find(raw.lower())
            if rel < 0:
                start = m.start("val")
                end = m.end("val")
            else:
                start = m.start("val") + rel
                end = start + len(raw)
            spans.append(
                PIISpan(
                    start=start,
                    end=end,
                    tag="patient_name",
                    text=text[start:end],
                    source=self.detector_name,
                    reason="deterministic_patient_name_inline",
                )
            )

        # Demographic-summary fallback, e.g. "Bridget 26 y.o. Female born Dec. 1999".
        # This intentionally supports one-token first names because the surrounding
        # age/sex/born context is strongly patient-specific.
        for m in self.DEMOGRAPHIC_SUMMARY_NAME_PATTERN.finditer(text):
            raw = str(m.group("name") or "").strip(" ,;:-")
            if not raw:
                continue
            if any(ch.isdigit() for ch in raw):
                continue
            low = raw.lower()
            if low in {"patient", "name", "female", "male"}:
                continue
            if any(tok in low for tok in self.NON_PATIENT_NAME_TOKENS):
                continue
            start = int(m.start("name"))
            end = int(m.end("name"))
            if end <= start:
                continue
            spans.append(
                PIISpan(
                    start=start,
                    end=end,
                    tag="patient_name",
                    text=text[start:end],
                    source=self.detector_name,
                    reason="deterministic_demographic_summary_name",
                )
            )

        # Explicit stamped-footer/header pattern: "Lastname, Firstname (MRN)" (or "Lastname Firstname (MRN)")
        for pat in (self.FOOTER_STAMP_PATTERN, self.FOOTER_STAMP_WITH_TRAILER_PATTERN):
            for m in pat.finditer(text):
                ns, ne = m.start("name"), m.end("name")
                ms, me = m.start("mrn"), m.end("mrn")
                name_val = text[ns:ne]
                mrn_val = text[ms:me]
                if not self._looks_like_non_patient_name(name_val):
                    spans.append(
                        PIISpan(
                            start=ns,
                            end=ne,
                            tag="patient_name",
                            text=name_val,
                            source=self.detector_name,
                            reason="deterministic_footer_stamp_name",
                        )
                    )
                ctx = text[max(0, ms - 40) : min(len(text), me + 40)]
                if self._valid_mrn(mrn_val, ctx):
                    spans.append(
                        PIISpan(
                            start=ms,
                            end=me,
                            tag="mrn",
                            text=mrn_val,
                            source=self.detector_name,
                            reason="deterministic_footer_stamp_mrn",
                        )
                    )

        for pattern in self.NAME_PATTERNS:
            for match in pattern.finditer(text):
                if match.lastindex:
                    start, end = match.start(1), match.end(1)
                else:
                    start, end = match.start(), match.end()
                # For generic "Last, First" hits, require nearby MRN context to reduce false positives.
                if pattern.pattern == self.NAME_PATTERNS[-1].pattern:
                    tail = text[end : min(len(text), end + 180)]
                    if not (self.MRN_LABEL.search(tail) or self.RECORD_LABEL.search(tail)):
                        continue
                snippet = text[max(0, start - 60) : min(len(text), end + 60)]
                if self.NAME_EXCLUDE_CONTEXT.search(snippet):
                    continue
                if self._looks_like_non_patient_name(text[start:end]):
                    continue
                spans.append(
                    PIISpan(
                        start=start,
                        end=end,
                        tag="patient_name",
                        text=text[start:end],
                        source=self.detector_name,
                        reason="deterministic_name_pattern",
                    )
                )

        # Header format line (all caps or title case), with MRN on one of the next lines.
        for match in re.finditer(
            r"(?im)^(?P<name>[A-Z][A-Za-z'`-]+(?:,\s*[A-Z][A-Za-z'`-]+|\s+[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+)?))\s*$",
            text,
        ):
            start = match.start("name")
            end = match.end("name")
            after = text[end : min(len(text), end + 220)]
            if not (self.MRN_LABEL.search(after) or self.RECORD_LABEL.search(after)):
                continue
            snippet = text[max(0, start - 60) : min(len(text), end + 60)]
            if self.NAME_EXCLUDE_CONTEXT.search(snippet):
                continue
            if self._looks_like_non_patient_name(text[start:end]):
                continue
            spans.append(
                PIISpan(
                    start=start,
                    end=end,
                    tag="patient_name",
                    text=text[start:end],
                    source=self.detector_name,
                    reason="deterministic_header_name_near_mrn",
                )
            )

        for pattern in self.MRN_PATTERNS:
            for match in pattern.finditer(text):
                if match.lastindex:
                    start, end = match.start(1), match.end(1)
                else:
                    start, end = match.start(), match.end()
                snippet = text[max(0, start - 40) : min(len(text), end + 40)]
                # Parenthesized numeric tokens are only safe to treat as MRN in explicit MRN/record context.
                if pattern.pattern == self.MRN_PATTERNS[-1].pattern:
                    strong_ctx = bool(self.MRN_LABEL.search(snippet) or self.RECORD_LABEL.search(snippet))
                    if not strong_ctx:
                        continue
                if not self._valid_mrn(text[start:end], snippet):
                    continue
                spans.append(
                    PIISpan(
                        start=start,
                        end=end,
                        tag="mrn",
                        text=text[start:end],
                        source=self.detector_name,
                        reason="deterministic_mrn_pattern",
                    )
                )
        # Label/value fallback for OCR where separators split oddly:
        # capture value immediately to the right of "Record #" / "MR#" labels.
        for m in re.finditer(
            r"(?im)\b(?:record|medical\s*record|mr|m\.?\s*r\.?)\s*(?:#|no\.?|number)\s*[:\-]?\s*(?P<val>[A-Za-z0-9\- ]{2,24})",
            text,
        ):
            raw = str(m.group("val") or "").strip()
            raw = re.split(r"[\n\r]", raw)[0].strip(" ,;:")
            if not raw:
                continue
            if not self._valid_mrn(raw, text[max(0, m.start() - 24): min(len(text), m.end() + 24)]):
                continue
            s = m.start("val")
            e = s + len(raw)
            spans.append(
                PIISpan(
                    start=s,
                    end=e,
                    tag="mrn",
                    text=text[s:e],
                    source=self.detector_name,
                    reason="deterministic_mrn_label_value",
                )
            )

        for pattern in self.DATE_PATTERNS:
            for match in pattern.finditer(text):
                spans.append(
                    PIISpan(
                        start=match.start(),
                        end=match.end(),
                        tag="date",
                        text=match.group(0),
                        source=self.detector_name,
                        reason="deterministic_date_pattern",
                    )
                )

        spans = self._propagate_repeated_literals(text, spans)
        return merge_overlapping_spans(spans)


class RegexPHIDetector(BasePHIDetector):
    detector_name = "regex"

    PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
        ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
        (
            "phone",
            re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        ),
        ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
        (
            "dob",
            re.compile(
                r"\b(?:DOB|Date of Birth)\s*[:\-]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
                flags=re.IGNORECASE,
            ),
        ),
        (
            "mrn",
            re.compile(
                r"\b(?:MRN|Medical\s+Record\s+Number)\s*[:#\-]?\s*[A-Za-z0-9\-]{4,}\b",
                flags=re.IGNORECASE,
            ),
        ),
        (
            "address",
            re.compile(
                r"\b\d{1,6}\s+[A-Za-z0-9.\-\s]+\s(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct)\b",
                flags=re.IGNORECASE,
            ),
        ),
        (
            "name_prefix",
            re.compile(r"\b(?:Patient|Name)\s*[:\-]\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b"),
        ),
        ("date", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")),
    ]

    def detect(self, text: str, device: Optional[str] = None) -> List[PIISpan]:
        spans: List[PIISpan] = []
        if not text:
            return spans
        for tag, pattern in self.PATTERNS:
            for match in pattern.finditer(text):
                spans.append(
                    PIISpan(start=match.start(), end=match.end(), tag=tag, text=match.group(0))
                )
        return merge_overlapping_spans(spans)


class RegistryFocusedPHIDetector(BasePHIDetector):
    """
    Focused detector for registry ingest:
    - INCLUDE: patient name, DOB, any date/time, MRN, Study ID, Event ID
    - EXCLUDE: physician names
    """

    detector_name = "registry_phi"

    INCLUDE_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
        # Footer format: Last, First (MRN)
        (
            "patient_name",
            re.compile(
                r"\b[A-Z][A-Za-z'`-]+,\s*[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+)*\s*\(\s*[A-Za-z0-9-]{3,}\s*\)"
            ),
        ),
        (
            "patient_name",
            re.compile(
                r"\b(?:Patient\s*Name|Name)\s*[:\-]\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b"
            ),
        ),
        (
            "dob",
            re.compile(
                r"\b(?:DOB|Date\s+of\s+Birth)\s*[:\-]?\s*"
                r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}|\d{1,2}-[A-Za-z]{3}-\d{4})\b",
                flags=re.IGNORECASE,
            ),
        ),
        (
            "date",
            re.compile(
                r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|"
                r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}|\d{1,2}-[A-Za-z]{3}-\d{4})\b"
            ),
        ),
        (
            "date",
            re.compile(
                r"\b(?:study\s*date)\s*:\s*\d{1,2}-[A-Za-z]{3}-\d{4}\b",
                flags=re.IGNORECASE,
            ),
        ),
        (
            "time",
            re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?:\s?[APap][Mm])?\b"),
        ),
        (
            "mrn",
            re.compile(
                r"\(\s*[A-Za-z0-9-]{4,}\s*\)",
            ),
        ),
        (
            "mrn",
            re.compile(
                r"\b(?:MRN|Medical\s+Record\s+Number)\s*[:#\-]?\s*[A-Za-z0-9\-]{4,}\b",
                flags=re.IGNORECASE,
            ),
        ),
        (
            "study_id",
            re.compile(
                r"\b(?:Study\s*ID|Study\s*Identifier)\s*[:#\-]?\s*[A-Za-z0-9\-]{4,}\b",
                flags=re.IGNORECASE,
            ),
        ),
        (
            "event_id",
            re.compile(
                r"\b(?:Event\s*ID)\s*[:#\-]?\s*[A-Za-z0-9\-]{3,}\b",
                flags=re.IGNORECASE,
            ),
        ),
    ]

    EXCLUDE_PATTERNS: List[re.Pattern[str]] = [
        re.compile(r"\b(?:Dr\.?|Doctor|Referring\s+Physician|Ordering\s+Physician)\b", flags=re.IGNORECASE),
        re.compile(r"\b(?:Interpreting\s+Physician|Reading\s+Physician)\b", flags=re.IGNORECASE),
    ]

    def detect(self, text: str, device: Optional[str] = None) -> List[PIISpan]:
        spans: List[PIISpan] = []
        if not text:
            return spans

        for tag, pattern in self.INCLUDE_PATTERNS:
            for match in pattern.finditer(text):
                start = match.start()
                end = match.end()
                snippet = text[max(0, start - 40) : min(len(text), end + 40)]
                if any(ex.search(snippet) for ex in self.EXCLUDE_PATTERNS) and tag == "patient_name":
                    continue
                spans.append(PIISpan(start=start, end=end, tag=tag, text=match.group(0)))

        return merge_overlapping_spans(spans)


class HybridUnionPHIDetector(BasePHIDetector):
    detector_name = "hybrid_union"

    def __init__(
        self,
        allow_llama_fallback: bool = False,
        ai_model_path: Optional[str] = None,
        ai_prompt_template: Optional[str] = None,
    ) -> None:
        self.core = DeterministicCorePHIDetector()
        self.llama: Optional[BasePHIDetector] = None
        if allow_llama_fallback:
            try:
                # Keep Llama focused to patient names only to reduce over-redaction.
                self.llama = NameOnlyLlamaDetector(
                    model_path=ai_model_path,
                    prompt_template=ai_prompt_template,
                )
            except Exception as exc:
                logger.warning("Local Llama unavailable, hybrid runs deterministic-only: %s", exc)

    def detect(self, text: str, device: Optional[str] = None) -> List[PIISpan]:
        spans = self.core.detect(text, device=device)
        if self.llama is not None:
            spans.extend(self.llama.detect(text, device=device))
        return merge_overlapping_spans(spans)


def build_detector(
    detector_backend: str,
    ai_model_path: Optional[str] = None,
    ai_prompt_template: Optional[str] = None,
) -> BasePHIDetector:
    backend = detector_backend.strip().lower()
    if backend in {"openpipe", "openpipe_llama", "legacy_llama"}:
        return LlamaPHIDetector()
    if backend in {"llama", "ai", "ai_based"}:
        return NameOnlyLlamaDetector(
            model_path=ai_model_path,
            prompt_template=ai_prompt_template,
        )
    if backend in {"regex", "registry_phi", "registry", "nlp", "nlp_based"}:
        return DeterministicCorePHIDetector()
    if backend == "regex_legacy":
        return RegexPHIDetector()
    if backend in {"hybrid", "hybrid_union"}:
        return HybridUnionPHIDetector(
            ai_model_path=ai_model_path,
            ai_prompt_template=ai_prompt_template,
        )
    raise ValueError("detector_backend must be one of: ai, nlp, hybrid, openpipe")


_OCR_ENGINES: Dict[str, Any] = {}
_TEXTRACT_CLIENT: Optional[Any] = None


def _normalize_ocr_backend(ocr_backend: str) -> str:
    backend = str(ocr_backend or "paddle").strip().lower()
    if backend in {"paddle", "paddleocr"}:
        return "paddle"
    if backend in {"tesseract", "pytesseract"}:
        return "tesseract"
    if backend in {"glmocr", "glm_ocr", "glm-ocr"}:
        return "glmocr"
    if backend in {"textract", "aws_textract"}:
        return "textract"
    raise ValueError("ocr_backend must be one of: paddle, tesseract, glmocr, textract")


def _normalize_ocr_line_text(text: str) -> str:
    """
    Conservative OCR cleanup for sensitive label/date matching.
    """
    t = str(text or "").strip()
    if not t:
        return t

    # Common label OCR confusions.
    t = re.sub(r"\bRec[0oO]rd\b", "Record", t, flags=re.IGNORECASE)
    t = re.sub(r"\bMed[1lI]cal\b", "Medical", t, flags=re.IGNORECASE)
    t = re.sub(r"\bB[0oO]rn\b", "Born", t, flags=re.IGNORECASE)
    t = re.sub(r"\bD[0oO]B\b", "DOB", t, flags=re.IGNORECASE)

    # Month token can absorb punctuation in OCR output.
    t = re.sub(
        r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
        r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.(?=\s|$)",
        r"\1",
        t,
        flags=re.IGNORECASE,
    )
    # Day-only fragment can appear as "-1,".
    t = re.sub(r"^\s*-\s*([0-9Il]{1,2})(\s*,?)\s*$", r"\1\2", t)
    return t


def _ensure_backports_tarfile_available() -> None:
    """
    In some PyInstaller onefile builds, setuptools/jaraco imports
    ``backports.tarfile`` at runtime and it may be missing from the bundle.
    Provide a minimal runtime alias to stdlib tarfile so Paddle can initialize.
    """
    try:
        import backports.tarfile  # type: ignore # noqa: F401
        return
    except Exception:
        pass

    import tarfile as _stdlib_tarfile

    backports_pkg = sys.modules.get("backports")
    if backports_pkg is None:
        backports_pkg = types.ModuleType("backports")
        backports_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["backports"] = backports_pkg
    sys.modules["backports.tarfile"] = _stdlib_tarfile
    try:
        setattr(backports_pkg, "tarfile", _stdlib_tarfile)
    except Exception:
        pass


def _ensure_cython_cppsupport_available() -> None:
    """
    Paddle's cpp_extension path may require Cython Utility templates in frozen
    onefile mode. Ensure the expected file exists to avoid startup crashes.
    """
    if not hasattr(sys, "_MEIPASS"):
        return
    try:
        meipass = str(getattr(sys, "_MEIPASS"))
        utility_dir = os.path.join(meipass, "Cython", "Utility")
        os.makedirs(utility_dir, exist_ok=True)
        cpp_support = os.path.join(utility_dir, "CppSupport.cpp")
        if not os.path.exists(cpp_support):
            with open(cpp_support, "w", encoding="utf-8") as f:
                f.write("// Runtime fallback file for frozen onefile builds.\n")
    except Exception:
        # Non-fatal hardening only; if this fails we still attempt normal import.
        pass


def get_ocr_engine(ocr_backend: str = "paddle") -> Any:
    backend = _normalize_ocr_backend(ocr_backend)
    if backend in _OCR_ENGINES:
        return _OCR_ENGINES[backend]

    if backend == "paddle":
        logger.info("Initializing PaddleOCR...")
        try:
            # Some frozen onefile environments miss setuptools' backports dependency.
            _ensure_backports_tarfile_available()
            _ensure_cython_cppsupport_available()

            # In frozen builds, ensure bundled DLL locations are visible before importing paddle.
            if hasattr(sys, "_MEIPASS"):
                meipass = str(getattr(sys, "_MEIPASS"))
                try:
                    os.add_dll_directory(meipass)
                except Exception:
                    pass
                for extra in ("paddle", "paddle\\libs", "paddleocr"):
                    p = os.path.join(meipass, extra)
                    if os.path.isdir(p):
                        try:
                            os.add_dll_directory(p)
                        except Exception:
                            pass
            from paddleocr import PaddleOCR  # Lazy import to avoid startup crash when OCR is unused.
        except Exception as exc:
            raise RuntimeError(
                "PaddleOCR failed to initialize in this packaged build. "
                "Rebuild with PyInstaller --collect-all paddle --collect-all paddleocr "
                "--collect-all Cython "
                "or use onedir build. "
                f"Root cause: {exc.__class__.__name__}: {exc}"
            ) from exc
        _OCR_ENGINES[backend] = PaddleOCR(use_angle_cls=True, lang="en")
        return _OCR_ENGINES[backend]

    if backend == "tesseract":
        try:
            import pytesseract  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Tesseract backend requested but pytesseract is not installed."
            ) from exc
        _OCR_ENGINES[backend] = pytesseract
        return _OCR_ENGINES[backend]

    if backend == "glmocr":
        try:
            glm_mod = importlib.import_module("glmocr")  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "glmocr backend requested but package 'glmocr' is not installed."
            ) from exc
        # Best-effort constructor discovery for package variants.
        pipeline = None
        for cls_name in ("OCRPipeline", "GLMOCR", "GlmOCR", "Pipeline"):
            cls = getattr(glm_mod, cls_name, None)
            if cls is None:
                continue
            try:
                pipeline = cls()
                break
            except Exception:
                pipeline = None
        _OCR_ENGINES[backend] = pipeline if pipeline is not None else glm_mod
        return _OCR_ENGINES[backend]

    if backend == "textract":
        global _TEXTRACT_CLIENT
        if _TEXTRACT_CLIENT is None:
            try:
                import boto3  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "Textract backend requested but boto3 is not installed."
                ) from exc
            _TEXTRACT_CLIENT = boto3.client("textract")
        _OCR_ENGINES[backend] = _TEXTRACT_CLIENT
        return _OCR_ENGINES[backend]

    raise ValueError("ocr_backend must be one of: paddle, tesseract, glmocr, textract")


def _glmocr_output_to_blocks(result: Any, width: int, height: int) -> List[Dict[str, Any]]:
    """
    Convert common glmocr output variants to internal LINE blocks.
    """
    candidates: List[Any]
    if isinstance(result, dict):
        for key in ("lines", "text_lines", "results", "blocks"):
            if isinstance(result.get(key), list):
                candidates = list(result.get(key) or [])
                break
        else:
            candidates = []
    elif isinstance(result, list):
        candidates = list(result)
    else:
        candidates = []

    blocks: List[Dict[str, Any]] = []
    for item in candidates:
        if isinstance(item, str):
            text = _normalize_ocr_line_text(item)
            if text:
                blocks.append(
                    {
                        "BlockType": "LINE",
                        "Text": text,
                        "Confidence": 0.0,
                        "Geometry": {
                            "BoundingBox": {
                                "Left": 0.0,
                                "Top": 0.0,
                                "Width": 1.0,
                                "Height": 0.0,
                            }
                        },
                    }
                )
            continue
        if not isinstance(item, dict):
            continue
        text = _normalize_ocr_line_text(str(item.get("text", "") or item.get("Text", "") or ""))
        if not text:
            continue
        conf_raw = item.get("confidence", item.get("score", item.get("Confidence", 0.0)))
        try:
            conf = float(conf_raw)
        except Exception:
            conf = 0.0
        bbox = item.get("bbox_xyxy", item.get("bbox", item.get("box", None)))
        left, top, bw, bh = 0.0, 0.0, 1.0, 0.0
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
                # Handle normalized or absolute coordinates.
                if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 2.0:
                    left = max(0.0, x1)
                    top = max(0.0, y1)
                    bw = max(0.0, x2 - x1)
                    bh = max(0.0, y2 - y1)
                else:
                    left = max(0.0, x1 / float(max(1, width)))
                    top = max(0.0, y1 / float(max(1, height)))
                    bw = max(0.0, (x2 - x1) / float(max(1, width)))
                    bh = max(0.0, (y2 - y1) / float(max(1, height)))
            except Exception:
                pass
        blocks.append(
            {
                "BlockType": "LINE",
                "Text": text,
                "Confidence": conf,
                "Geometry": {
                    "BoundingBox": {
                        "Left": left,
                        "Top": top,
                        "Width": bw,
                        "Height": bh,
                    }
                },
            }
        )
    return blocks


def _reset_ocr_engine(ocr_backend: str) -> None:
    backend = _normalize_ocr_backend(ocr_backend)
    if backend in _OCR_ENGINES:
        try:
            del _OCR_ENGINES[backend]
        except Exception:
            _OCR_ENGINES.pop(backend, None)


def ensure_dir(path: Path, auto_resolve_file_conflict: bool = True) -> Path:
    if path.exists() and path.is_file():
        if not auto_resolve_file_conflict:
            raise NotADirectoryError(
                f"Expected directory path but found file: {path}. "
                "Rename/delete that file or choose a different output_dir."
            )

        # Safer than delete: preserve the file, then create the folder.
        backup = path.with_name(f"{path.name}_file_backup")
        i = 1
        while backup.exists():
            backup = path.with_name(f"{path.name}_file_backup_{i}")
            i += 1
        path.rename(backup)

    path.mkdir(parents=True, exist_ok=True)
    return path



def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem)


EXCLUDED_SCAN_DIR_NAMES = {
    "redaction_output",
    "approved_for_transfer",
    ".cache",
}


def _is_in_excluded_dir(path: Path) -> bool:
    path_parts = {part.lower() for part in path.parts}
    for excluded in EXCLUDED_SCAN_DIR_NAMES:
        if excluded.lower() in path_parts:
            return True
    if any(part.lower().startswith("redaction_output_") for part in path.parts):
        return True
    return False


def list_input_files(input_dir: Path, recursive: bool = False) -> List[Path]:
    """
    By default scans ONLY top-level files in `input_dir`.
    Set recursive=True only when explicitly needed.
    """
    files: List[Path] = []
    if recursive:
        iterator = sorted(input_dir.rglob("*"))
    else:
        iterator = sorted(input_dir.glob("*"))

    for path in iterator:
        if not path.is_file():
            continue
        if _is_in_excluded_dir(path):
            continue
        if path.suffix.lower() in SUPPORTED_DOC_EXTS:
            files.append(path)
    return files


def render_pdf_to_pil_images(pdf_path: Path, dpi: int = DEFAULT_DPI) -> List[Image.Image]:
    doc = fitz.open(pdf_path)
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    images: List[Image.Image] = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    doc.close()
    return images


def load_document_pages(file_path: Path, dpi: int = DEFAULT_DPI) -> List[Image.Image]:
    if file_path.suffix.lower() == ".pdf":
        return render_pdf_to_pil_images(file_path, dpi=dpi)
    return [Image.open(file_path).convert("RGB")]


def _pdf_page_to_ocr_page(doc: fitz.Document, page_index: int, dpi: int = DEFAULT_DPI) -> OCRPage:
    page = doc.load_page(page_index)
    words = page.get_text("words")
    if not words:
        raise ValueError("No words extracted from PDF page")

    scale = dpi / 72.0
    grouped: Dict[Tuple[int, int], List[Tuple[float, float, float, float, str]]] = {}
    for word in words:
        x0, y0, x1, y1, text, block_no, line_no, _word_no = word
        key = (int(block_no), int(line_no))
        grouped.setdefault(key, []).append((x0, y0, x1, y1, text))

    line_entries = sorted(
        grouped.values(),
        key=lambda items: (min(v[1] for v in items), min(v[0] for v in items)),
    )

    text_parts: List[str] = []
    lines: List[OCRLine] = []
    cursor = 0
    for items in line_entries:
        sorted_items = sorted(items, key=lambda value: value[0])
        # Preserve rough horizontal spacing from PDF word geometry.
        # This materially improves span->box alignment for footer/signature lines
        # where large gaps exist between tokens.
        weighted_char_width_sum = 0.0
        weighted_char_count = 0.0
        for wx0, _wy0, wx1, _wy1, wtxt in sorted_items:
            t = str(wtxt or "")
            tlen = len(t)
            if tlen <= 0:
                continue
            ww = max(0.0, float(wx1) - float(wx0))
            weighted_char_width_sum += ww
            weighted_char_count += float(tlen)
        avg_char_w = (weighted_char_width_sum / weighted_char_count) if weighted_char_count > 0 else 4.0
        avg_char_w = max(1.0, float(avg_char_w))

        line_chunks: List[str] = []
        prev_x1: Optional[float] = None
        for wx0, _wy0, wx1, _wy1, wtxt in sorted_items:
            token = str(wtxt or "")
            if not token:
                continue
            if prev_x1 is None:
                line_chunks.append(token)
            else:
                gap = max(0.0, float(wx0) - float(prev_x1))
                # Convert geometric gap to approximate spaces; keep bounded.
                spaces = int(round(gap / avg_char_w))
                spaces = max(1, min(16, spaces))
                line_chunks.append((" " * spaces) + token)
            prev_x1 = float(wx1)

        line_text = _normalize_ocr_line_text("".join(line_chunks))
        if not line_text:
            continue

        if text_parts:
            text_parts.append("\n")
            cursor += 1
        start = cursor
        text_parts.append(line_text)
        cursor += len(line_text)
        end = cursor

        x1 = min(value[0] for value in sorted_items) * scale
        y1 = min(value[1] for value in sorted_items) * scale
        x2 = max(value[2] for value in sorted_items) * scale
        y2 = max(value[3] for value in sorted_items) * scale
        lines.append(
            OCRLine(
                text=line_text,
                confidence=1.0,
                polygon=[],
                bbox_xyxy=(x1, y1, x2, y2),
                char_start=start,
                char_end=end,
            )
        )

    rect = page.rect
    return OCRPage(
        page_index=page_index,
        width=int(rect.width * scale),
        height=int(rect.height * scale),
        text="".join(text_parts),
        lines=lines,
        extraction_method="pdf_text",
    )


def _merge_bottom_footer_ocr(base_page: OCRPage, ocr_page: OCRPage) -> OCRPage:
    """
    Merge OCR footer lines into PDF-text page when stamped footer content is not
    present in embedded PDF text (common in scanned/overlay footers).
    """
    h = max(1.0, float(base_page.height))
    footer_hint = re.compile(
        r"\b(?:study\s*on|study\s*date|page\s*\d+\s*of\s*\d+|mrn|record\s*(?:#|no\.?|number)|"
        r"[A-Z][A-Za-z'`-]+,\s*[A-Z][A-Za-z'`-]+)\b",
        flags=re.IGNORECASE,
    )

    existing_lines = list(base_page.lines)
    existing_text = base_page.text
    cursor = len(existing_text)
    if existing_text and not existing_text.endswith("\n"):
        existing_text += "\n"
        cursor += 1

    for ln in ocr_page.lines:
        x1, y1, x2, y2 = ln.bbox_xyxy
        txt = (ln.text or "").strip()
        if not txt:
            continue
        # Keep only bottom-band lines likely to contain footer PHI/date/page markers.
        if y2 < (h * 0.84):
            continue
        if not footer_hint.search(txt):
            continue

        # Skip near-duplicates against existing lines (text or geometry overlap).
        duplicate = False
        for ex in existing_lines:
            ex_txt = (ex.text or "").strip().lower()
            if ex_txt and ex_txt == txt.lower():
                duplicate = True
                break
            iou = _boxes_iou(ex.bbox_xyxy, ln.bbox_xyxy)
            if iou > 0.75 and ex_txt:
                duplicate = True
                break
        if duplicate:
            continue

        start = cursor
        existing_text += txt
        cursor += len(txt)
        existing_text += "\n"
        cursor += 1

        existing_lines.append(
            OCRLine(
                text=txt,
                confidence=float(ln.confidence),
                polygon=ln.polygon,
                bbox_xyxy=ln.bbox_xyxy,
                char_start=start,
                char_end=start + len(txt),
            )
        )

    if existing_text.endswith("\n"):
        existing_text = existing_text[:-1]

    return OCRPage(
        page_index=base_page.page_index,
        width=base_page.width,
        height=base_page.height,
        text=existing_text,
        lines=existing_lines,
        extraction_method="pdf_text+footer_ocr",
    )


def run_ocr(local_file: str, ocr_backend: str = "paddle") -> Tuple[str, Dict[str, Any]]:
    image = cv2.imread(local_file)
    if image is None:
        raise ValueError(f"Could not read image for OCR: {local_file}")

    height, width, _ = image.shape
    backend = _normalize_ocr_backend(ocr_backend)
    blocks: List[Dict[str, Any]] = []

    if backend == "paddle":
        try:
            result = get_ocr_engine(backend).ocr(image)
        except Exception as exc:
            logger.warning(
                "Paddle OCR failed on %s; resetting engine and retrying once. Root cause: %s: %s",
                local_file,
                exc.__class__.__name__,
                exc,
            )
            _reset_ocr_engine(backend)
            try:
                result = get_ocr_engine(backend).ocr(image)
            except Exception as retry_exc:
                logger.warning(
                    "Paddle OCR retry failed on %s; attempting Tesseract fallback. Root cause: %s: %s",
                    local_file,
                    retry_exc.__class__.__name__,
                    retry_exc,
                )
                try:
                    return run_ocr(local_file, ocr_backend="tesseract")
                except Exception:
                    raise RuntimeError(
                        f"Paddle OCR failed after retry: {retry_exc.__class__.__name__}: {retry_exc}"
                    ) from retry_exc
        for line in result[0] if result else []:
            if not line or len(line) < 2:
                continue
            poly = line[0]
            text, confidence = line[1]
            text = _normalize_ocr_line_text(text)
            if not text:
                continue

            xs = [point[0] for point in poly]
            ys = [point[1] for point in poly]
            left = min(xs) / width
            top = min(ys) / height
            box_width = (max(xs) - min(xs)) / width
            box_height = (max(ys) - min(ys)) / height

            blocks.append(
                {
                    "BlockType": "LINE",
                    "Text": text,
                    "Confidence": float(confidence),
                    "Geometry": {
                        "BoundingBox": {
                            "Left": left,
                            "Top": top,
                            "Width": box_width,
                            "Height": box_height,
                        }
                    },
                }
            )
    elif backend == "tesseract":
        pytesseract = get_ocr_engine(backend)
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            config="--oem 3 --psm 6",
        )
        n = len(data.get("text", []))
        for i in range(n):
            text = _normalize_ocr_line_text(str(data["text"][i] or ""))
            if not text:
                continue
            try:
                conf = float(data.get("conf", [])[i])
            except Exception:
                conf = 0.0
            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])
            if w <= 0 or h <= 0:
                continue
            blocks.append(
                {
                    "BlockType": "LINE",
                    "Text": text,
                    "Confidence": conf,
                    "Geometry": {
                        "BoundingBox": {
                            "Left": float(x) / float(width),
                            "Top": float(y) / float(height),
                            "Width": float(w) / float(width),
                            "Height": float(h) / float(height),
                        }
                    },
                }
            )
    elif backend == "glmocr":
        engine = get_ocr_engine(backend)
        result: Any = None
        last_exc: Optional[Exception] = None
        # Try common API variants across package versions.
        for attr_name in ("ocr", "recognize", "predict", "infer"):
            fn = getattr(engine, attr_name, None)
            if not callable(fn):
                continue
            for arg in (local_file, image):
                try:
                    result = fn(arg)
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    continue
            if last_exc is None:
                break
        if result is None:
            # Module-level fallbacks.
            for attr_name in ("ocr", "recognize", "predict", "infer"):
                fn = getattr(engine, attr_name, None)
                if not callable(fn):
                    continue
                for arg in (local_file, image):
                    try:
                        result = fn(arg)
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                        continue
                if last_exc is None:
                    break
        if result is None:
            raise RuntimeError(
                "glmocr backend is installed but no compatible OCR API method was found."
                + (f" Last error: {last_exc}" if last_exc is not None else "")
            )
        blocks = _glmocr_output_to_blocks(result, width=width, height=height)
    elif backend == "textract":
        client = get_ocr_engine(backend)
        with open(local_file, "rb") as f:
            img_bytes = f.read()
        resp = client.detect_document_text(Document={"Bytes": img_bytes})
        for blk in resp.get("Blocks", []):
            if str(blk.get("BlockType", "")).upper() != "LINE":
                continue
            text = _normalize_ocr_line_text(str(blk.get("Text", "") or ""))
            if not text:
                continue
            blocks.append(
                {
                    "BlockType": "LINE",
                    "Text": text,
                    "Confidence": float(blk.get("Confidence", 0.0)),
                    "Geometry": {
                        "BoundingBox": {
                            "Left": float(blk.get("Geometry", {}).get("BoundingBox", {}).get("Left", 0.0)),
                            "Top": float(blk.get("Geometry", {}).get("BoundingBox", {}).get("Top", 0.0)),
                            "Width": float(blk.get("Geometry", {}).get("BoundingBox", {}).get("Width", 0.0)),
                            "Height": float(blk.get("Geometry", {}).get("BoundingBox", {}).get("Height", 0.0)),
                        }
                    },
                }
            )
    else:
        raise ValueError("ocr_backend must be one of: paddle, tesseract, glmocr, textract")

    flat_text = "\n".join(block["Text"] for block in blocks)
    return flat_text, {"Blocks": blocks}


def run_ocr_file_to_page(local_file: Path, page_index: int, ocr_backend: str = "paddle") -> OCRPage:
    backend = _normalize_ocr_backend(ocr_backend)
    _flat_text, response = run_ocr(str(local_file), ocr_backend=backend)
    image = cv2.imread(str(local_file))
    if image is None:
        raise ValueError(f"Could not read image: {local_file}")

    height, width, _ = image.shape
    text_parts: List[str] = []
    lines: List[OCRLine] = []
    cursor = 0
    for block in response["Blocks"]:
        text = str(block.get("Text", "")).strip()
        if not text:
            continue

        if text_parts:
            text_parts.append("\n")
            cursor += 1
        start = cursor
        text_parts.append(text)
        cursor += len(text)
        end = cursor

        box = block["Geometry"]["BoundingBox"]
        x1 = box["Left"] * width
        y1 = box["Top"] * height
        x2 = x1 + box["Width"] * width
        y2 = y1 + box["Height"] * height
        lines.append(
            OCRLine(
                text=text,
                confidence=float(block.get("Confidence", 1.0)),
                polygon=[],
                bbox_xyxy=(x1, y1, x2, y2),
                char_start=start,
                char_end=end,
            )
        )

    return OCRPage(
        page_index=page_index,
        width=width,
        height=height,
        text="".join(text_parts),
        lines=lines,
        extraction_method=f"ocr:{backend}",
    )


TAG_PATTERN = re.compile(r"<PII:(?P<tag>[a-zA-Z0-9_]+)>(?P<text>.*?)</PII:\1>", flags=re.DOTALL)


def parse_tagged_string(tagged_str: str) -> Tuple[str, List[PIISpan]]:
    spans: List[PIISpan] = []
    clean_parts: List[str] = []
    last_idx = 0
    clean_cursor = 0
    for match in TAG_PATTERN.finditer(tagged_str):
        before = tagged_str[last_idx : match.start()]
        tagged_text = match.group("text")
        tag = match.group("tag")

        clean_parts.append(before)
        clean_cursor += len(before)

        span_start = clean_cursor
        clean_parts.append(tagged_text)
        clean_cursor += len(tagged_text)
        span_end = clean_cursor
        spans.append(PIISpan(start=span_start, end=span_end, tag=tag, text=tagged_text))
        last_idx = match.end()

    clean_parts.append(tagged_str[last_idx:])
    return "".join(clean_parts), spans


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def find_best_span_in_original(
    needle: str,
    haystack: str,
    start_hint: int,
    window: int = 200,
) -> Optional[Tuple[int, int]]:
    if not needle or not haystack:
        return None

    search_start = max(0, start_hint - window)
    search_end = min(len(haystack), start_hint + max(window, len(needle) * 3))
    local = haystack[search_start:search_end]

    literal_idx = local.lower().find(needle.lower())
    if literal_idx != -1:
        start = search_start + literal_idx
        return start, start + len(needle)

    pattern = re.escape(needle.strip()).replace(r"\ ", r"\s+")
    match = re.search(pattern, local, flags=re.IGNORECASE)
    if match:
        return search_start + match.start(), search_start + match.end()

    literal_idx = haystack.lower().find(needle.lower())
    if literal_idx != -1:
        return literal_idx, literal_idx + len(needle)

    match = re.search(pattern, haystack, flags=re.IGNORECASE)
    if match:
        return match.start(), match.end()
    return None


def align_pii_spans_to_ocr_text(original_text: str, tagged_text: str) -> List[PIISpan]:
    cleaned, tagged_spans = parse_tagged_string(tagged_text)
    if normalize_for_match(cleaned) != normalize_for_match(original_text):
        logger.debug("Cleaned tagged text differs from OCR/PDF source text")

    aligned: List[PIISpan] = []
    for span in tagged_spans:
        found = find_best_span_in_original(span.text, original_text, start_hint=span.start)
        if found is None:
            logger.warning("Could not align tagged PHI span: %r", span.text)
            continue
        start, end = found
        aligned.append(PIISpan(start=start, end=end, tag=span.tag, text=original_text[start:end]))
    return merge_overlapping_spans(aligned)


def merge_overlapping_spans(spans: List[PIISpan]) -> List[PIISpan]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda value: (value.start, value.end))
    merged: List[PIISpan] = [ordered[0]]
    for span in ordered[1:]:
        last = merged[-1]
        if span.start <= last.end:
            chosen = last if (last.end - last.start) >= (span.end - span.start) else span
            merged_source = "|".join(
                value for value in [last.source, span.source] if str(value).strip()
            )
            merged_reason = "|".join(
                value for value in [last.reason, span.reason] if str(value).strip()
            )
            merged[-1] = PIISpan(
                start=last.start,
                end=max(last.end, span.end),
                tag=chosen.tag,
                text=chosen.text or "",
                source=merged_source,
                reason=merged_reason,
            )
        else:
            merged.append(span)
    return merged


def expand_patient_name_token_spans(text: str, spans: List[PIISpan]) -> List[PIISpan]:
    """
    If a full patient name is detected once, redact repeated first/last-name tokens
    elsewhere in the same page text.
    """
    if not text:
        return spans

    out: List[PIISpan] = list(spans)
    seen = {(s.tag, int(s.start), int(s.end)) for s in spans}
    stop = {"patient", "name", "demographics"}
    tokens_to_find: set[str] = set()

    for s in spans:
        if s.tag != "patient_name":
            continue
        value = (s.text or text[s.start:s.end]).strip()
        if not value:
            continue
        # Split on commas/space while keeping alpha name tokens.
        parts = re.findall(r"[A-Za-z][A-Za-z'`-]+", value)
        if len(parts) < 2:
            continue
        # Prefer first and last token for precision.
        cand = [parts[0], parts[-1]]
        for token in cand:
            t = token.strip()
            if len(t) < 3:
                continue
            if t.lower() in stop:
                continue
            tokens_to_find.add(t)

    for token in tokens_to_find:
        pattern = re.compile(rf"\b{re.escape(token)}\b", flags=re.IGNORECASE)
        for m in pattern.finditer(text):
            key = ("patient_name", int(m.start()), int(m.end()))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                PIISpan(
                    start=int(m.start()),
                    end=int(m.end()),
                    tag="patient_name",
                    text=text[m.start():m.end()],
                    source="deterministic_core",
                    reason="patient_name_token_propagation",
                )
            )

    return merge_overlapping_spans(out)


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def union_boxes(boxes: Sequence[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def pii_spans_to_redaction_boxes(page: OCRPage, spans: List[PIISpan]) -> List[RedactionBox]:
    """
    Approximate span-level boxes within line boxes (instead of redacting whole lines).
    This materially reduces over-redaction for lines like cardiac history.
    """
    boxes: List[RedactionBox] = []
    for span in spans:
        overlapping_lines = [
            line
            for line in page.lines
            if spans_overlap(span.start, span.end, line.char_start, line.char_end)
        ]
        if not overlapping_lines:
            continue

        partial_boxes: List[Tuple[float, float, float, float]] = []
        for line in overlapping_lines:
            overlap_start = max(span.start, line.char_start)
            overlap_end = min(span.end, line.char_end)
            if overlap_end <= overlap_start:
                continue

            line_text_len = max(1, line.char_end - line.char_start)
            # For day-only date redaction, prefer line-local day token localization
            # to avoid occasional month/day horizontal drift.
            if span.tag == "date_day":
                line_text = page.text[line.char_start:line.char_end]
                day_raw = (span.text or "").strip()
                rel_start = (overlap_start - line.char_start) / line_text_len
                rel_end = (overlap_end - line.char_start) / line_text_len
                try:
                    day_token = re.sub(r"\D", "", day_raw)
                    if day_token:
                        # Try to find explicit date(s) on this line and lock onto the
                        # day token that best matches both value and initial span location.
                        date_pat = re.compile(
                            r"\b(?:(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})|(\d{4})[/-](\d{1,2})[/-](\d{1,2})|([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4}))\b"
                        )
                        base_mid = (rel_start + rel_end) / 2.0
                        best_day_span: Optional[Tuple[int, int]] = None
                        best_score: Optional[float] = None
                        for dm in date_pat.finditer(line_text):
                            day_span: Tuple[int, int] | None = None
                            if dm.group(2):  # mm/dd/yyyy
                                day_span = (dm.start(2), dm.end(2))
                            elif dm.group(6):  # yyyy-mm-dd
                                day_span = (dm.start(6), dm.end(6))
                            elif dm.group(8):  # Month DD, YYYY
                                day_span = (dm.start(8), dm.end(8))
                            if day_span is None:
                                continue
                            ds, de = day_span
                            day_val = re.sub(r"\D", "", line_text[ds:de])
                            if day_val != day_token:
                                continue
                            cand_mid = ((ds / max(1, len(line_text))) + (de / max(1, len(line_text)))) / 2.0
                            score = abs(cand_mid - base_mid)
                            if best_score is None or score < best_score:
                                best_score = score
                                best_day_span = (ds, de)
                        if best_day_span is not None:
                            ds, de = best_day_span
                            rel_start = ds / max(1, len(line_text))
                            rel_end = de / max(1, len(line_text))
                except Exception:
                    pass
            else:
                rel_start = (overlap_start - line.char_start) / line_text_len
                rel_end = (overlap_end - line.char_start) / line_text_len
            rel_start = min(max(rel_start, 0.0), 1.0)
            rel_end = min(max(rel_end, 0.0), 1.0)
            if rel_end <= rel_start:
                continue

            x1, y1, x2, y2 = line.bbox_xyxy
            width = max(1e-6, x2 - x1)
            px1 = x1 + (width * rel_start)
            px2 = x1 + (width * rel_end)
            if span.tag == "date_day":
                # Day boxes should be precise but must not clip 2-digit days.
                day_digits = re.findall(r"[0-9Il]", str(span.text or ""))
                char_w = width / max(1, line_text_len)
                if len(day_digits) >= 2:
                    min_w = max(1.25 * char_w, (px2 - px1))
                    if (px2 - px1) < min_w:
                        px2 = px1 + min_w
                    # Small right buffer to catch the second digit under OCR drift.
                    px2 += max(0.45 * char_w, 1.0)
                    px1 -= max(0.05 * char_w, 0.0)
                else:
                    shrink = max(0.0, (px2 - px1) * 0.05)
                    px1 += shrink
                    px2 -= shrink
                if px2 <= px1:
                    continue
            partial_boxes.append((px1, y1, px2, y2))

        for partial in partial_boxes:
            boxes.append(
                RedactionBox(
                    page_index=page.page_index,
                    tag=span.tag,
                    text=span.text,
                    bbox_xyxy=partial,
                )
            )
    return boxes


def _has_legacy_footer_context(page: OCRPage) -> bool:
    text = (page.text or "").lower()
    has_page_counter = bool(re.search(r"\bpage\s*\d+\s*of\s*\d+\b", text))
    has_footer_date_marker = ("study on" in text) or bool(re.search(r"\bdate\b", text))
    # Keep this permissive for footer-template propagation across pages.
    return has_page_counter or has_footer_date_marker


def _line_looks_like_footer_phi(line_text: str) -> bool:
    t = (line_text or "").strip()
    if not t:
        return False
    # Signature lines are not patient footer PHI stamps.
    if re.search(r"\b(?:e-?signed|electronically\s+signed|signed\s+electronically)\b", t, flags=re.IGNORECASE):
        return False
    if re.search(r"\([A-Za-z0-9\- ]{3,}\)", t):
        return True
    if re.search(r"\b[A-Z][A-Za-z'`-]+,\s*[A-Z][A-Za-z'`-]+\b", t):
        return True
    if re.search(r"\b(?:mrn|record\s*(?:#|no\.?|number)?)\b", t, flags=re.IGNORECASE):
        return True
    return False


def _has_bottom_signature_line(page: OCRPage) -> bool:
    h = max(1.0, float(page.height))
    sig_pat = re.compile(
        r"\b(?:e-?signed|electronically\s+signed|signed\s+electronically)\b",
        flags=re.IGNORECASE,
    )
    for ln in page.lines:
        txt = str(ln.text or "").strip()
        if not txt:
            continue
        if not sig_pat.search(txt):
            continue
        if float(ln.bbox_xyxy[3]) >= (h * 0.80):
            return True
    return False


def _box_on_signature_line(page: OCRPage, box: RedactionBox) -> bool:
    sig_pat = re.compile(
        r"\b(?:e-?signed|electronically\s+signed|signed\s+electronically)\b",
        flags=re.IGNORECASE,
    )
    bx1, by1, bx2, by2 = box.bbox_xyxy
    for ln in page.lines:
        txt = str(ln.text or "").strip()
        if not txt or not sig_pat.search(txt):
            continue
        lx1, ly1, lx2, ly2 = ln.bbox_xyxy
        ox = max(0.0, min(float(bx2), float(lx2)) - max(float(bx1), float(lx1)))
        oy = max(0.0, min(float(by2), float(ly2)) - max(float(by1), float(ly1)))
        if ox > 0 and oy > 0:
            return True
    return False


def _is_patient_label_text(value: str) -> bool:
    low = re.sub(r"[^a-z]+", " ", (value or "").lower()).strip()
    if not low:
        return True
    if "demographics" in low:
        return True
    tokens = [t for t in low.split() if t]
    if not tokens:
        return True
    return set(tokens).issubset({"patient", "name", "demographics"})


def detect_patient_name_table_boxes(page: OCRPage) -> List[RedactionBox]:
    """
    OCR-line based fallback for table-style demographics blocks.
    Targets value cell for "Patient Name" and avoids redacting labels.
    """
    w = max(1.0, float(page.width))
    h = max(1.0, float(page.height))
    out: List[RedactionBox] = []

    lines = [ln for ln in page.lines if (ln.text or "").strip()]
    if not lines:
        return out

    anchor_pat = re.compile(r"\b(?:patient\s*name|name)\b", flags=re.IGNORECASE)
    patient_anchor_pat = re.compile(r"\bpatient\s*name\b", flags=re.IGNORECASE)
    demographics_ctx_pat = re.compile(
        r"\b(?:mrn|dob|date\s*of\s*birth|sex|age\s*at\s*study|demographics)\b",
        flags=re.IGNORECASE,
    )
    skip_val_pat = re.compile(
        r"\b(?:patient\s*name|demographics|study\s*time|signed\s*on|date\s*of\s*birth|dob|age\s*at\s*study|gender)\b",
        flags=re.IGNORECASE,
    )
    name_shape = re.compile(r"^[A-Za-z][A-Za-z'`.-]+(?:[ ,]+[A-Za-z][A-Za-z'`.-]+){1,4}$")
    inline_name_pat = re.compile(
        r"\b(?:patient\s*name|name)\s*[:\-]?\s*(?P<val>[A-Za-z][A-Za-z ,.'`-]{2,80})",
        flags=re.IGNORECASE,
    )

    anchors = [ln for ln in lines if anchor_pat.search(ln.text or "")]
    if not anchors:
        return out

    for a in anchors:
        a_text = (a.text or "").strip()
        ax1, ay1, ax2, ay2 = a.bbox_xyxy
        ah = max(1.0, ay2 - ay1)

        # First try inline extraction when label and value share the same OCR line.
        inline_match = inline_name_pat.search(a_text)
        if inline_match:
            raw_val = inline_match.group("val")
            raw_val = re.split(
                r"\b(?:study\s*time|signed\s*on|date\s*of\s*birth|dob|medical\s*record|mrn|age\s*at\s*study|gender|sex|height|weight|bsa)\b",
                raw_val,
                flags=re.IGNORECASE,
            )[0].strip(" ,;:-")
            if raw_val and not _is_patient_label_text(raw_val):
                raw_norm = re.sub(r"\s+", " ", raw_val).strip(" ,;:-")
                if (
                    not any(ch.isdigit() for ch in raw_norm)
                    and name_shape.match(raw_norm)
                    and len(raw_norm.split()) >= 2
                ):
                    line_text = a_text
                    lower_line = line_text.lower()
                    rel_idx = lower_line.find(raw_norm.lower())
                    if rel_idx < 0:
                        rel_idx = inline_match.start("val")
                        rel_end = inline_match.end("val")
                    else:
                        rel_end = rel_idx + len(raw_norm)
                    line_len = max(1, len(line_text))
                    width = max(1.0, ax2 - ax1)
                    rx1 = rel_idx / line_len
                    rx2 = rel_end / line_len
                    x1 = ax1 + (width * rx1)
                    x2 = ax1 + (width * rx2)
                    px = max(1.0, w * 0.002)
                    py = max(1.0, h * 0.002)
                    if x2 > x1:
                        out.append(
                            RedactionBox(
                                page_index=page.page_index,
                                tag="patient_name",
                                text=raw_norm,
                                bbox_xyxy=(
                                    max(0.0, x1 - px),
                                    max(0.0, ay1 - py),
                                    min(w, x2 + px),
                                    min(h, ay2 + py),
                                ),
                            )
                        )
                        # Inline hit is highest confidence for this anchor.
                        continue

        # If anchor is generic "Name" (not explicit "Patient Name"),
        # require nearby demographics cues to avoid non-patient name rows.
        if not patient_anchor_pat.search(a_text):
            yc = (ay1 + ay2) / 2.0
            has_demographics_context = False
            for ln in lines:
                lx1, ly1, lx2, ly2 = ln.bbox_xyxy
                lyc = (ly1 + ly2) / 2.0
                if abs(lyc - yc) > max(h * 0.14, ah * 4.0):
                    continue
                if demographics_ctx_pat.search(ln.text or ""):
                    has_demographics_context = True
                    break
            if not has_demographics_context:
                continue

        row_cands: List[OCRLine] = []
        for ln in lines:
            if ln is a:
                continue
            lx1, ly1, lx2, ly2 = ln.bbox_xyxy
            # Same row-ish.
            if ly2 < ay1 - (ah * 0.35) or ly1 > ay2 + (ah * 0.35):
                continue
            # Right of the label cell, but still in left-middle table area (avoid right column date/time).
            if lx1 <= ax2 + 4:
                continue
            if lx1 > (w * 0.68):
                continue
            text = (ln.text or "").strip()
            if len(text) < 2:
                continue
            if skip_val_pat.search(text):
                continue
            if _is_patient_label_text(text):
                continue
            # Avoid obvious numeric-only cells.
            if re.fullmatch(r"[0-9/\-: ]+", text):
                continue
            # Keep only plausible person-name cells.
            text_norm = re.sub(r"\s+", " ", text).strip(" ,;:-")
            if any(ch.isdigit() for ch in text_norm):
                continue
            if not name_shape.match(text_norm):
                continue
            row_cands.append(ln)

        if not row_cands:
            continue

        # Use the nearest plausible candidate to the right of "Patient Name"
        # to avoid spanning the full row into right-column fields.
        row_cands = sorted(row_cands, key=lambda ln: ln.bbox_xyxy[0])
        chosen = row_cands[0]
        chosen_text = re.sub(r"\s+", " ", (chosen.text or "")).strip(" ,;:-")
        x1, y1, x2, y2 = chosen.bbox_xyxy

        # Tight pad.
        px = max(1.0, w * 0.002)
        py = max(1.0, h * 0.002)
        # Safety clamp: keep compact box near the left value cell.
        max_x2 = min(w * 0.62, ax2 + (w * 0.30))
        x2 = min(float(x2), float(max_x2))
        if x2 <= x1:
            continue
        out.append(
            RedactionBox(
                page_index=page.page_index,
                tag="patient_name",
                text=chosen_text or "patient_name",
                bbox_xyxy=(
                    max(0.0, x1 - px),
                    max(0.0, y1 - py),
                    min(w, x2 + px),
                    min(h, y2 + py),
                ),
            )
        )

    return out


def detect_mrn_label_value_boxes(page: OCRPage) -> List[RedactionBox]:
    """
    OCR line-level fallback for MRN label-value forms:
    Record #, Medical Record #, MR#, MR No, MR Number.
    """
    out: List[RedactionBox] = []
    w = max(1.0, float(page.width))
    h = max(1.0, float(page.height))
    label_pat = re.compile(
        r"\b(?:record|medical\s*record|mr|m\.?\s*r\.?)\b[^\n\r]{0,12}?(?:#|no\.?|number)?\s*[:\-]?\s*",
        flags=re.IGNORECASE,
    )
    # MRN token can include short separators; keep conservative and require digits.
    val_pat = re.compile(r"(?P<val>[A-Za-z0-9][A-Za-z0-9\-_/ ]{1,31})")
    banned_value_words = re.compile(
        r"\b(?:name|born|dob|date|age|gender|sex|height|weight|bsa|doctor)\b",
        flags=re.IGNORECASE,
    )

    def _is_strong_mrn_label(label_fragment: str) -> bool:
        lf = str(label_fragment or "")
        if re.search(r"\b(?:record|medical\s*record|mrn)\b", lf, flags=re.IGNORECASE):
            return True
        if re.search(r"\bm\.?\s*r\.?\b", lf, flags=re.IGNORECASE) and re.search(
            r"(?:#|no\.?|number)",
            lf,
            flags=re.IGNORECASE,
        ):
            return True
        return False

    def _is_plausible_mrn_value(raw: str) -> bool:
        cand = re.sub(r"\s+", " ", str(raw or "")).strip(" -:")
        if not cand:
            return False
        # Hard guardrail: MRN should not include special chars (dates/codes often do).
        if re.search(r"[^\w\- ]", cand):
            return False
        if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", cand):
            return False
        if banned_value_words.search(cand):
            return False
        digit_count = len(re.findall(r"\d", cand))
        compact = re.sub(r"[^A-Za-z0-9]", "", cand)
        # Keep permissive but avoid short non-MRN numeric fragments.
        return digit_count >= 2 and len(compact) >= 4

    def _looks_non_mrn_tail(raw: str) -> bool:
        t = str(raw or "").strip()
        if not t:
            return False
        if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", t):
            return True
        if re.search(r"[^\w\- ]", t):
            return True
        return False

    for ln in page.lines:
        text = str(ln.text or "").strip()
        if not text:
            continue
        m = label_pat.search(text)
        if not m:
            continue
        label_text = text[m.start() : m.end()]
        if not _is_strong_mrn_label(label_text):
            # Avoid false positives like "Cardiovascular MR examination".
            continue
        tail = text[m.end():]
        vm = None
        for cand in val_pat.finditer(tail):
            cval = str(cand.group("val") or "").strip()
            if _is_plausible_mrn_value(cval):
                vm = cand
                break
        val = ""
        val_start = -1
        val_end = -1
        if vm:
            val = str(vm.group("val") or "").strip()
            val_start = m.end() + vm.start("val")
            val_end = m.end() + vm.end("val")
        else:
            # Split-line fallback: value sometimes lands on nearby line to the right.
            lx1, ly1, lx2, ly2 = ln.bbox_xyxy
            yc = (float(ly1) + float(ly2)) / 2.0
            candidates: List[Tuple[float, OCRLine, re.Match[str]]] = []
            for rn in page.lines:
                rx1, ry1, rx2, ry2 = rn.bbox_xyxy
                if float(rx2) <= float(lx2):
                    continue
                rcy = (float(ry1) + float(ry2)) / 2.0
                if abs(rcy - yc) > max(72.0, 3.2 * max(1.0, float(ly2 - ly1))):
                    continue
                rtxt = str(rn.text or "").strip()
                if not rtxt:
                    continue
                for rv in val_pat.finditer(rtxt):
                    cval = str(rv.group("val") or "").strip()
                    if not _is_plausible_mrn_value(cval):
                        continue
                    # Prefer nearest-right and digit-richer candidates.
                    digit_bonus = -2.0 * len(re.findall(r"\d", cval))
                    score = abs(float(rx1) - float(lx2)) + abs(rcy - yc) + digit_bonus
                    candidates.append((score, rn, rv))
            if candidates:
                candidates.sort(key=lambda t: t[0])
                picked = False
                for _, rn, rv in candidates:
                    rtxt = str(rn.text or "")
                    cand = str(rv.group("val") or "").strip()
                    if not cand:
                        continue
                    if not _is_plausible_mrn_value(cand):
                        # Avoid selecting nearby non-MRN text (e.g. Name line) as split-line value.
                        continue
                    val = cand
                    val_start = int(rv.start("val"))
                    val_end = int(rv.end("val"))
                    text = rtxt
                    ln = rn
                    picked = True
                    break
                if not picked:
                    if _looks_non_mrn_tail(tail):
                        # Do not anchor-fallback on date-like/symbol-heavy value tails.
                        continue
                    # Label-anchored fallback: OCR may miss MRN glyphs entirely.
                    bw = max(1.0, float(lx2 - lx1))
                    label_end_x = float(lx1) + (bw * (m.end() / max(1, len(text))))
                    char_w = bw / max(1, len(text))
                    fb_x1 = label_end_x + max(1.0, 0.5 * char_w)
                    # Prefer nearby right-side line geometry if present.
                    nearby_right: List[OCRLine] = []
                    for rn in page.lines:
                        rx1, ry1, rx2, ry2 = rn.bbox_xyxy
                        rcy = (float(ry1) + float(ry2)) / 2.0
                        if abs(rcy - yc) > max(72.0, 3.2 * max(1.0, float(ly2 - ly1))):
                            continue
                        if float(rx2) <= float(lx2):
                            continue
                        rtxt = str(rn.text or "").strip()
                        if not rtxt:
                            continue
                        if banned_value_words.search(rtxt):
                            continue
                        nearby_right.append(rn)
                    if nearby_right:
                        fb_x1 = max(fb_x1, min(float(rn.bbox_xyxy[0]) for rn in nearby_right))
                        fb_x2 = min(w, max(float(rn.bbox_xyxy[2]) for rn in nearby_right))
                    else:
                        fb_x2 = min(w, fb_x1 + min(0.30 * w, 320.0))
                    if fb_x2 <= fb_x1:
                        continue
                    px = max(1.0, w * 0.002)
                    py = max(1.0, h * 0.002)
                    out.append(
                        RedactionBox(
                            page_index=page.page_index,
                            tag="mrn",
                            text="mrn_label_anchor_fallback",
                            bbox_xyxy=(
                                max(0.0, fb_x1 - px),
                                max(0.0, float(ly1) - py),
                                min(w, fb_x2 + px),
                                min(h, float(ly2) + py),
                            ),
                        )
                    )
                    continue
            else:
                if _looks_non_mrn_tail(tail):
                    # Do not anchor-fallback on date-like/symbol-heavy value tails.
                    continue
                # Label-anchored fallback when no split-line candidate was found.
                bw = max(1.0, float(lx2 - lx1))
                label_end_x = float(lx1) + (bw * (m.end() / max(1, len(text))))
                char_w = bw / max(1, len(text))
                fb_x1 = label_end_x + max(1.0, 0.5 * char_w)
                fb_x2 = min(w, fb_x1 + min(0.30 * w, 320.0))
                if fb_x2 <= fb_x1:
                    continue
                px = max(1.0, w * 0.002)
                py = max(1.0, h * 0.002)
                out.append(
                    RedactionBox(
                        page_index=page.page_index,
                        tag="mrn",
                        text="mrn_label_anchor_fallback",
                        bbox_xyxy=(
                            max(0.0, fb_x1 - px),
                            max(0.0, float(ly1) - py),
                            min(w, fb_x2 + px),
                            min(h, float(ly2) + py),
                        ),
                    )
                )
                continue
        if not val:
            continue
        val = re.sub(r"\s+", " ", val).strip(" -:")
        if not _is_plausible_mrn_value(val):
            continue

        rel_s = val_start / max(1, len(text))
        rel_e = val_end / max(1, len(text))
        x1, y1, x2, y2 = ln.bbox_xyxy
        bw = max(1.0, float(x2 - x1))
        bx1 = float(x1) + (bw * rel_s)
        bx2 = float(x1) + (bw * rel_e)
        # Safety: never allow MRN value box to extend into label segment.
        if vm:
            label_end_x = float(x1) + (bw * (m.end() / max(1, len(text))))
            min_left = label_end_x + max(1.0, (bw / max(1, len(text))) * 0.5)
            if bx1 < min_left:
                shift = (min_left - bx1)
                bx1 += shift
                bx2 += shift
        if bx2 <= bx1:
            continue
        px = max(1.0, w * 0.002)
        py = max(1.0, h * 0.002)
        out.append(
            RedactionBox(
                page_index=page.page_index,
                tag="mrn",
                text=val,
                bbox_xyxy=(
                    max(0.0, bx1 - px),
                    max(0.0, float(y1) - py),
                    min(w, bx2 + px),
                    min(h, float(y2) + py),
                ),
            )
        )
    return out


def _normalize_day_token_to_int(raw: str) -> Optional[int]:
    token = str(raw or "").strip()
    if not token:
        return None
    if token in {"I", "i", "l", "L"}:
        return 1
    if token.isdigit():
        return int(token)
    return None


def _choose_day_group_for_slash_date(
    raw_a: str,
    raw_b: str,
    prefer_day_first: bool = False,
) -> str:
    val_a = _normalize_day_token_to_int(raw_a)
    val_b = _normalize_day_token_to_int(raw_b)
    if val_a is None or val_b is None:
        # Preserve legacy behavior for noisy/ambiguous OCR.
        return "b"
    if val_a > 12 and val_b <= 12:
        return "a"
    if val_b > 12 and val_a <= 12:
        return "b"
    # Ambiguous numeric dates (both <= 12) default to legacy month-first behavior.
    # This avoids regressions when locale cannot be inferred confidently.
    return "b"


def _extract_day_token_span(
    text: str,
    prefer_day_first: bool = False,
) -> Optional[Tuple[int, int, str, bool, int]]:
    """
    Return day-token span within text plus display metadata:
    (start, end, normalized_day_text, month_name_mode, month_name_len).
    """
    t = str(text or "")
    if not t:
        return None

    # dd/mm/yyyy or mm/dd/yyyy -> choose day token using disambiguation + preference.
    m = re.search(
        r"\b(?P<a>\d{1,2}|[Il])[/-](?P<b>\d{1,2}|[Il])[/-](?P<yy>\d{2,4})\b",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        raw_a = str(m.group("a") or "")
        raw_b = str(m.group("b") or "")
        key = _choose_day_group_for_slash_date(raw_a, raw_b, prefer_day_first=prefer_day_first)

        raw = str(m.group(key) or "")
        day = "1" if raw in {"I", "i", "l", "L"} else raw
        return (m.start(key), m.end(key), day, False, 0)

    # yyyy-mm-dd -> day group.
    m = re.search(r"\b(?P<yy>\d{4})[/-](?P<mm>\d{1,2})[/-](?P<dd>\d{1,2}|[Il])\b", t)
    if m:
        raw = str(m.group("dd") or "")
        day = "1" if raw in {"I", "i", "l", "L"} else raw
        return (m.start("dd"), m.end("dd"), day, False, 0)

    # DD-MMM-YYYY / DD-Month-YYYY -> day is the leading token.
    m = re.search(
        r"\b(?P<dd>\d{1,2}|[Il])[\s,./:-]*(?P<mon>[A-Za-z]{3,9})[\s,./:-]*(?P<yy>\d{4})\b",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        raw = str(m.group("dd") or "")
        day = "1" if raw in {"I", "i", "l", "L"} else raw
        return (m.start("dd"), m.end("dd"), day, False, 0)

    # Month-name day year; tolerate punctuation or cramped OCR separators.
    m = re.search(
        r"\b(?P<mon>[A-Za-z]{3,9})[\s,./:-]*(?P<dd>\d{1,2}|[Il])[\s,./:-]*,?[\s,./:-]*(?P<yy>\d{4})\b",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        raw = str(m.group("dd") or "")
        day = "1" if raw in {"I", "i", "l", "L"} else raw
        mon_len = len(str(m.group("mon") or "").strip())
        return (m.start("dd"), m.end("dd"), day, True, mon_len)

    # OCR-messy fallback: month ... year with noisy day token in between.
    month_year = re.search(
        r"\b([A-Za-z]{3,9})\b(?P<body>[^\n\r]{0,24}?)\b(\d{4})\b",
        t,
        flags=re.IGNORECASE,
    )
    if not month_year:
        return None
    body = str(month_year.group("body") or "")
    day_guess = re.search(r"\b(\d{1,2}|[Il])\b", body)
    if day_guess:
        raw = str(day_guess.group(1) or "")
        day = "1" if raw in {"I", "i", "l", "L"} else raw
        ds = month_year.start("body") + day_guess.start(1)
        de = month_year.start("body") + day_guess.end(1)
        mon_len = len(str(month_year.group(1) or "").strip())
        return (ds, de, day, True, mon_len)

    # Geometry-first backup if day OCR became punctuation/symbols.
    token_cands = list(re.finditer(r"[A-Za-z0-9Il]{1,3}", body))
    if token_cands:
        tg = token_cands[-1]
        ds = month_year.start("body") + tg.start(0)
        de = month_year.start("body") + tg.end(0)
        raw = str(tg.group(0) or "")
        mon_len = len(str(month_year.group(1) or "").strip())
        return (ds, de, raw, True, mon_len)
    year_s = month_year.start(3)
    ds = max(month_year.start("body"), year_s - 2)
    de = max(ds + 1, year_s)
    mon_len = len(str(month_year.group(1) or "").strip())
    return (ds, de, t[ds:de], True, mon_len)


def detect_labeled_date_day_boxes(
    page: OCRPage,
    prefer_day_first: bool = False,
) -> List[RedactionBox]:
    """
    OCR line-level fallback for date labels ('Study Date', 'Date', 'DOB', 'Born').
    Redacts day token only using line-local date parsing.
    """
    out: List[RedactionBox] = []
    w = max(1.0, float(page.width))
    h = max(1.0, float(page.height))
    label_pat = re.compile(
        r"\b(?:study\s*date|date(?:\s*of\s*birth)?|dob|born)\b\s*[:\-]?\s*",
        flags=re.IGNORECASE,
    )

    for ln in page.lines:
        text = str(ln.text or "").strip()
        if not text:
            continue
        label_hits = list(label_pat.finditer(text))
        if not label_hits:
            continue
        for i, lm in enumerate(label_hits):
            day_text = ""
            month_name_len = 0
            month_name_mode = False
            next_label_start = label_hits[i + 1].start() if (i + 1) < len(label_hits) else len(text)
            tail = text[lm.end() : next_label_start]
            day_hit = _extract_day_token_span(tail, prefer_day_first=prefer_day_first)
            target_line = ln
            target_text = text
            label_end_offset = lm.end()
            if not day_hit:
                # Split-line fallback: date value can sit on nearby line to the right/below.
                lx1, ly1, lx2, ly2 = ln.bbox_xyxy
                yc = (float(ly1) + float(ly2)) / 2.0
                near_lines: List[Tuple[float, OCRLine, Tuple[int, int, str, bool, int]]] = []
                for rn in page.lines:
                    rtxt = str(rn.text or "").strip()
                    if not rtxt:
                        continue
                    rx1, ry1, rx2, ry2 = rn.bbox_xyxy
                    if float(rx2) <= float(lx1):
                        continue
                    rcy = (float(ry1) + float(ry2)) / 2.0
                    if abs(rcy - yc) > max(72.0, 3.2 * max(1.0, float(ly2 - ly1))):
                        continue
                    maybe_day = _extract_day_token_span(rtxt, prefer_day_first=prefer_day_first)
                    if not maybe_day:
                        continue
                    score = abs(float(rx1) - float(lx2)) + abs(rcy - yc)
                    near_lines.append((score, rn, maybe_day))
                if near_lines:
                    near_lines.sort(key=lambda t: t[0])
                    _, target_line, day_hit = near_lines[0]
                    target_text = str(target_line.text or "").strip()
            if not day_hit:
                # Final fallback: if we can find a year, mask a small day slot immediately before it.
                year_m = re.search(r"\b(?:19|20)\d{2}\b", tail)
                if not year_m:
                    continue
                pre = tail[: year_m.start()]
                tok = list(re.finditer(r"[A-Za-z0-9]{1,3}", pre))
                if tok:
                    ds = tok[-1].start(0)
                    de = tok[-1].end(0)
                    day_text = str(tok[-1].group(0) or "")
                else:
                    ds = max(0, year_m.start() - 2)
                    de = max(ds + 1, year_m.start())
                    day_text = tail[ds:de]
                day_hit = (ds, de, day_text, False, 0)
            ds, de, day_text, month_name_mode, month_name_len = day_hit
            # On same-line labels, day indexes are relative to `tail`; convert to `text`.
            abs_s = ds + (label_end_offset if target_line is ln else 0)
            abs_e = de + (label_end_offset if target_line is ln else 0)
            if abs_e <= abs_s:
                continue
            rel_s = abs_s / max(1, len(target_text))
            rel_e = abs_e / max(1, len(target_text))
            x1, y1, x2, y2 = target_line.bbox_xyxy
            bw = max(1.0, float(x2 - x1))
            bx1 = float(x1) + (bw * rel_s)
            bx2 = float(x1) + (bw * rel_e)
            # Ensure day boxes are not too narrow for proportional mapping drift.
            min_w = max((bw / max(1, len(target_text))) * 1.8, w * 0.006)
            if (bx2 - bx1) < min_w:
                mid = (bx1 + bx2) / 2.0
                bx1 = mid - (min_w / 2.0)
                bx2 = mid + (min_w / 2.0)
            # Safety: on same-line labels, day token should sit to the right of label segment.
            if target_line is ln:
                label_end_x = float(x1) + (bw * (label_end_offset / max(1, len(target_text))))
                min_left = label_end_x + max(1.0, (bw / max(1, len(target_text))) * 0.5)
                if bx1 < min_left:
                    shift = (min_left - bx1)
                    bx1 += shift
                    bx2 += shift
            if month_name_mode and month_name_len > 3:
                # Month-name OCR lines tend to under-estimate right-side placement with linear char mapping.
                # Nudge day anchor right based on extra month-name length beyond a short month token.
                char_w = bw / max(1, len(target_text))
                shift = max(0.0, (month_name_len - 3) * 0.42 * char_w)
                bx1 += shift
                bx2 += shift
            if bx2 <= bx1:
                continue
            px = max(1.0, w * 0.0015)
            py = max(1.0, h * 0.0015)
            out.append(
                RedactionBox(
                    page_index=page.page_index,
                    tag="date_day",
                    text=day_text or target_text[abs_s:abs_e],
                    bbox_xyxy=(
                        max(0.0, bx1 - px),
                        max(0.0, float(y1) - py),
                        min(w, bx2 + px),
                        min(h, float(y2) + py),
                    ),
                )
            )
    return out


def _prefer_label_geometry_boxes(
    boxes: Sequence[RedactionBox],
    label_boxes: Sequence[RedactionBox],
) -> List[RedactionBox]:
    out: List[RedactionBox] = []
    for b in boxes:
        replaced = False
        for lb in label_boxes:
            if b.tag != lb.tag:
                continue
            if _boxes_iou(b.bbox_xyxy, lb.bbox_xyxy) >= 0.15:
                replaced = True
                break
        if not replaced:
            out.append(b)
    out.extend(label_boxes)
    return out


def _line_has_label(line_text: str, kind: str) -> bool:
    t = str(line_text or "")
    if kind == "mrn":
        return bool(
            re.search(
                r"\b(?:record|medical\s*record|mr|m\.?\s*r\.?)\s*(?:#\s*:?|no\.?|number)\b",
                t,
                flags=re.IGNORECASE,
            )
        )
    if kind == "date_day":
        return bool(
            re.search(
                r"\b(?:study\s*date|date(?:\s*of\s*birth)?|dob|born)\b",
                t,
                flags=re.IGNORECASE,
            )
        )
    return False


def _force_label_boxes_on_labeled_lines(
    page: OCRPage,
    boxes: Sequence[RedactionBox],
    label_boxes: Sequence[RedactionBox],
    kind: str,
) -> List[RedactionBox]:
    """
    For labeled form lines, remove drifting same-kind boxes and keep label-anchored boxes.
    """
    if not label_boxes:
        # Do not strip existing boxes when labeled fallback found nothing.
        return list(boxes)
    labeled_lines = [ln for ln in page.lines if _line_has_label(ln.text, kind)]
    if not labeled_lines:
        return list(boxes)
    out: List[RedactionBox] = []
    for b in boxes:
        if b.tag != kind:
            out.append(b)
            continue
        bx1, by1, bx2, by2 = b.bbox_xyxy
        on_labeled_line = False
        for ln in labeled_lines:
            lx1, ly1, lx2, ly2 = ln.bbox_xyxy
            ox = max(0.0, min(float(bx2), float(lx2)) - max(float(bx1), float(lx1)))
            oy = max(0.0, min(float(by2), float(ly2)) - max(float(by1), float(ly1)))
            if ox > 0 and oy > 0:
                on_labeled_line = True
                break
        if not on_labeled_line:
            out.append(b)
    out.extend([lb for lb in label_boxes if lb.tag == kind])
    return out


def _boxes_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter + 1e-6)


def _rel_box(box: Tuple[float, float, float, float], page: OCRPage) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    w = max(1.0, float(page.width))
    h = max(1.0, float(page.height))
    return (x1 / w, y1 / h, x2 / w, y2 / h)


def _abs_box(rel: Tuple[float, float, float, float], page: OCRPage) -> Tuple[float, float, float, float]:
    rx1, ry1, rx2, ry2 = rel
    w = max(1.0, float(page.width))
    h = max(1.0, float(page.height))
    return (rx1 * w, ry1 * h, rx2 * w, ry2 * h)


def detect_footer_stamp_boxes_from_lines(page: OCRPage) -> List[RedactionBox]:
    """
    Footer-specific detector using OCR line geometry.
    This avoids a big static blob and targets bottom-left stamp text more tightly.
    """
    if not _has_legacy_footer_context(page):
        return []

    w = max(1.0, float(page.width))
    h = max(1.0, float(page.height))
    candidates: List[OCRLine] = []
    for line in page.lines:
        x1, y1, x2, y2 = line.bbox_xyxy
        txt = (line.text or "").strip()
        strong_footer_text = bool(
            re.search(r"\b[A-Z][A-Za-z'`-]+,\s*[A-Z][A-Za-z'`-]+\b", txt)
            and re.search(r"\([A-Za-z0-9\- ]{3,}\)", txt)
        )
        # Restrict to the bottom band, with a small relaxation for strong footer-text matches.
        min_bottom = h * (0.90 if strong_footer_text else 0.94)
        if y2 < min_bottom:
            continue
        if x1 > (w * 0.45):
            continue
        line_w = max(1.0, x2 - x1)
        line_h = max(1.0, y2 - y1)
        # Guardrails: footer stamp should be compact, not a page-wide bar.
        if line_w > (w * 0.45):
            continue
        if line_h > (h * 0.06):
            continue
        if len(txt) > 80:
            continue
        if _line_looks_like_footer_phi(txt):
            candidates.append(line)

    if not candidates:
        return []

    # Keep only lines tightly clustered with the bottom-most candidate.
    bottom_y2 = max(line.bbox_xyxy[3] for line in candidates)
    keep = [
        line
        for line in candidates
        if (bottom_y2 - line.bbox_xyxy[3]) <= (h * 0.02)
    ]
    if not keep:
        keep = candidates

    x1 = min(line.bbox_xyxy[0] for line in keep)
    y1 = min(line.bbox_xyxy[1] for line in keep)
    x2 = max(line.bbox_xyxy[2] for line in keep)
    y2 = max(line.bbox_xyxy[3] for line in keep)

    # Small pad for OCR jitter.
    pad_x = max(2.0, w * 0.003)
    pad_y = max(2.0, h * 0.003)
    box = (
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(w, x2 + pad_x),
        min(h, y2 + pad_y),
    )
    # Final safety clamp: do not allow large footer coverage due to noisy OCR.
    max_w = w * 0.42
    max_h = h * 0.07
    bw = box[2] - box[0]
    bh = box[3] - box[1]
    if bw > max_w:
        box = (box[0], box[1], min(w, box[0] + max_w), box[3])
    if bh > max_h:
        box = (box[0], box[1], box[2], min(h, box[1] + max_h))
    return [
        RedactionBox(
            page_index=page.page_index,
            tag="footer_phi",
            text="footer_stamp_detected",
            bbox_xyxy=box,
        )
    ]


def detect_footer_stamp_proxy_boxes(page: OCRPage) -> List[RedactionBox]:
    """
    Backward-compatible wrapper used by older call sites.
    """
    return detect_footer_stamp_boxes_from_lines(page)


def detect_footer_boxes_from_patient_tokens(
    page: OCRPage,
    spans: Sequence[PIISpan],
) -> List[RedactionBox]:
    """
    Aggressive fallback: if name/MRN tokens are known from this page text,
    find matching bottom-left OCR lines and redact tightly there.
    """
    w = max(1.0, float(page.width))
    h = max(1.0, float(page.height))
    tokens: set[str] = set()

    for s in spans:
        if s.tag not in {"patient_name", "mrn"}:
            continue
        raw = (s.text or page.text[s.start:s.end] or "").strip()
        if not raw:
            continue
        if s.tag == "mrn":
            t = re.sub(r"[^A-Za-z0-9]", "", raw)
            if len(t) >= 4:
                tokens.add(t.lower())
        else:
            for t in re.findall(r"[A-Za-z][A-Za-z'`-]+", raw):
                if len(t) >= 3 and t.lower() not in {"patient", "name", "demographics"}:
                    tokens.add(t.lower())

    if not tokens:
        return []

    matched: List[OCRLine] = []
    for ln in page.lines:
        x1, y1, x2, y2 = ln.bbox_xyxy
        txt = (ln.text or "").strip()
        if not txt:
            continue
        if y2 < (h * 0.84):
            continue
        if x1 > (w * 0.52):
            continue
        line_norm = re.sub(r"[^A-Za-z0-9]", "", txt).lower()
        line_low = txt.lower()
        # Token match by exact word OR compact substring (for OCR punctuation drift).
        has_match = False
        for tok in tokens:
            if re.search(rf"\b{re.escape(tok)}\b", line_low):
                has_match = True
                break
            if len(tok) >= 4 and tok in line_norm:
                has_match = True
                break
        if has_match:
            matched.append(ln)

    if not matched:
        return []

    x1 = min(ln.bbox_xyxy[0] for ln in matched)
    y1 = min(ln.bbox_xyxy[1] for ln in matched)
    x2 = max(ln.bbox_xyxy[2] for ln in matched)
    y2 = max(ln.bbox_xyxy[3] for ln in matched)

    # Tight pad and clamp.
    px = max(1.0, w * 0.002)
    py = max(1.0, h * 0.002)
    box = (
        max(0.0, x1 - px),
        max(0.0, y1 - py),
        min(w, x2 + px),
        min(h, y2 + py),
    )
    max_w = w * 0.45
    max_h = h * 0.10
    bw = box[2] - box[0]
    bh = box[3] - box[1]
    if bw > max_w:
        box = (box[0], box[1], min(w, box[0] + max_w), box[3])
    if bh > max_h:
        box = (box[0], box[1], box[2], min(h, box[1] + max_h))

    return [
        RedactionBox(
            page_index=page.page_index,
            tag="footer_phi",
            text="footer_token_match",
            bbox_xyxy=box,
        )
    ]


def _extract_date_day_only_spans(
    page_text: str,
    span: PIISpan,
    prefer_day_first: bool = False,
    redact_full_dates: bool = False,
) -> List[PIISpan]:
    """
    For date/DOB spans, redact day component only.
    """
    if span.tag not in {"date", "dob"}:
        return [span]
    if redact_full_dates:
        return [span]

    subtext = page_text[span.start : span.end]
    out: List[PIISpan] = []

    # yyyy-mm-dd or yyyy/mm/dd -> day is group 3
    for match in re.finditer(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", subtext):
        day_start = span.start + match.start(3)
        day_end = span.start + match.end(3)
        out.append(
            PIISpan(
                start=day_start,
                end=day_end,
                tag="date_day",
                text=page_text[day_start:day_end],
                source=span.source,
                reason=f"{span.reason or span.tag}_day_only",
            )
        )

    # dd-mm-yyyy or mm-dd-yyyy -> choose day component with numeric disambiguation.
    for match in re.finditer(
        r"\b(?P<a>\d{1,2}|[Il])[-/](?P<b>\d{1,2}|[Il])[-/](?P<y>\d{2,4})\b",
        subtext,
        flags=re.IGNORECASE,
    ):
        raw_a = str(match.group("a") or "")
        raw_b = str(match.group("b") or "")
        day_group = _choose_day_group_for_slash_date(raw_a, raw_b, prefer_day_first=prefer_day_first)

        day_start = span.start + match.start(day_group)
        day_end = span.start + match.end(day_group)
        out.append(
            PIISpan(
                start=day_start,
                end=day_end,
                tag="date_day",
                text=page_text[day_start:day_end],
                source=span.source,
                reason=f"{span.reason or span.tag}_day_only",
            )
        )

    # Month DD, YYYY (also handles OCR-noisy punctuation/newline variants) -> day is group 2
    for match in re.finditer(
        r"\b([A-Za-z]{3,9})[\s\.,:/-]*(-?[0-9Il]{1,2})\s*,?\s*(\d{4})\b",
        subtext,
        flags=re.IGNORECASE,
    ):
        ds = match.start(2)
        de = match.end(2)
        raw_day = subtext[ds:de]
        inner = re.search(r"[0-9Il]{1,2}", raw_day)
        if inner:
            day_start = span.start + ds + inner.start()
            day_end = span.start + ds + inner.end()
        else:
            day_start = span.start + ds
            day_end = span.start + de
        out.append(
            PIISpan(
                start=day_start,
                end=day_end,
                tag="date_day",
                text=page_text[day_start:day_end],
                source=span.source,
                reason=f"{span.reason or span.tag}_day_only",
            )
        )

    # DD-MMM-YYYY -> day is group 1
    for match in re.finditer(
        r"\b(\d{1,2})-([A-Za-z]{3})-(\d{4})\b",
        subtext,
        flags=re.IGNORECASE,
    ):
        day_start = span.start + match.start(1)
        day_end = span.start + match.end(1)
        out.append(
            PIISpan(
                start=day_start,
                end=day_end,
                tag="date_day",
                text=page_text[day_start:day_end],
                source=span.source,
                reason=f"{span.reason or span.tag}_day_only",
            )
        )

    # If we cannot isolate an explicit day token, do NOT redact month/year-only dates.
    # This avoids unintended redaction of values like "7/2013".
    return out


def _normalize_date_for_overlay(value: str) -> Optional[str]:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        def _expand_two_digit_year(year_2: int) -> int:
            return 2000 + year_2 if year_2 <= 30 else 1900 + year_2

        # Prefer explicit patterns to reduce locale ambiguity.
        m = re.match(r"^\s*(\d{1,2})[/-](\d{4})\s*$", raw)
        if m:
            mm = int(m.group(1))
            yyyy = int(m.group(2))
            if 1 <= mm <= 12:
                return f"{mm:02d}/01/{yyyy:04d}"
        m = re.match(r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\s*$", raw)
        if m:
            mm = int(m.group(1))
            yyyy = int(m.group(3))
            if yyyy < 100:
                yyyy = _expand_two_digit_year(yyyy)
            if 1 <= mm <= 12:
                return f"{mm:02d}/01/{yyyy:04d}"
        m = re.match(r"^\s*(\d{4})[/-](\d{1,2})[/-](\d{1,2})\s*$", raw)
        if m:
            yyyy = int(m.group(1))
            mm = int(m.group(2))
            if 1 <= mm <= 12:
                return f"{mm:02d}/01/{yyyy:04d}"
        # Month-name forms.
        ts = pd.to_datetime(raw, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.strftime("%m/01/%Y")
    except Exception:
        return None


def _parse_date_for_age_days(value: str) -> Optional[pd.Timestamp]:
    raw = str(value or "").strip()
    if not raw:
        return None
    raw = raw.strip("[](){}.,;: ")
    try:
        def _expand_two_digit_year(year_2: int) -> int:
            return 2000 + year_2 if year_2 <= 30 else 1900 + year_2

        # Month/year forms: assume day=1.
        m = re.match(r"^\s*(\d{1,2})[/-](\d{4})\s*$", raw)
        if m:
            mm = int(m.group(1))
            yyyy = int(m.group(2))
            if 1 <= mm <= 12:
                return pd.Timestamp(year=yyyy, month=mm, day=1)
        m = re.match(r"^\s*(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\s*$", raw)
        if m:
            mm = int(m.group(1))
            dd = int(m.group(2))
            yyyy = int(m.group(3))
            if yyyy < 100:
                yyyy = _expand_two_digit_year(yyyy)
            if 1900 <= yyyy <= 2100 and 1 <= mm <= 12 and 1 <= dd <= 31:
                return pd.Timestamp(year=yyyy, month=mm, day=dd)
        m = re.match(r"^\s*(\d{4})[./-](\d{1,2})[./-](\d{1,2})\s*$", raw)
        if m:
            yyyy = int(m.group(1))
            mm = int(m.group(2))
            dd = int(m.group(3))
            if 1900 <= yyyy <= 2100 and 1 <= mm <= 12 and 1 <= dd <= 31:
                return pd.Timestamp(year=yyyy, month=mm, day=dd)
        # Reject day-only or other short numerics that pandas may coerce to year 2001-style dates.
        if re.fullmatch(r"\d{1,2}", raw):
            return None
        m = re.match(r"^\s*([A-Za-z]{3,9})[-/\s]+(\d{4})\s*$", raw)
        if m:
            ts = pd.to_datetime(f"01 {m.group(1)} {m.group(2)}", errors="coerce")
            if not pd.isna(ts):
                return pd.Timestamp(ts).normalize()
        # Common OCR prefix noise like "-Jan-2004".
        m = re.match(r"^\s*[-_/\\]*([A-Za-z]{3,9})[-/\s]+(\d{4})\s*$", raw)
        if m:
            ts = pd.to_datetime(f"01 {m.group(1)} {m.group(2)}", errors="coerce")
            if not pd.isna(ts):
                return pd.Timestamp(ts).normalize()
        ts = pd.to_datetime(raw, errors="coerce", dayfirst=False)
        if pd.isna(ts):
            return None
        if int(ts.year) < 1900 or int(ts.year) > 2100:
            return None
        return pd.Timestamp(ts).normalize()
    except Exception:
        return None


def _build_age_days_overlays_from_items(
    redaction_items: Sequence[Dict[str, Any]],
    dob_value: str,
    fallback_age_days: Optional[int] = None,
    exclude_bboxes: Optional[Sequence[Sequence[int]]] = None,
) -> List[Dict[str, Any]]:
    overlays: List[Dict[str, Any]] = []
    dob_ts = _parse_date_for_age_days(dob_value)
    fallback_days_int: Optional[int] = None
    if fallback_age_days is not None:
        try:
            parsed_fallback = int(fallback_age_days)
            if parsed_fallback >= 0:
                fallback_days_int = parsed_fallback
        except Exception:
            fallback_days_int = None
    if dob_ts is None and fallback_days_int is None:
        return overlays

    exclusion_regions: List[List[int]] = []
    for raw in list(exclude_bboxes or []):
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            continue
        try:
            rx1, ry1, rx2, ry2 = [int(float(v)) for v in raw]
        except Exception:
            continue
        ex1, ex2 = sorted([rx1, rx2])
        ey1, ey2 = sorted([ry1, ry2])
        exclusion_regions.append([ex1, ey1, ex2, ey2])

    def _overlaps_exclusion(box: List[int]) -> bool:
        bx1, by1, bx2, by2 = box
        for ex1, ey1, ex2, ey2 in exclusion_regions:
            if bx2 < ex1 or ex2 < bx1 or by2 < ey1 or ey2 < by1:
                continue
            return True
        return False
    candidates: List[Dict[str, Any]] = []
    for item in list(redaction_items or []):
        tag = str(item.get("tag", "") or "").strip().lower()
        # EU display: overlay age on event-date boxes only.
        # Never overlay on DOB/Born redaction boxes.
        if tag not in {"date", "date_history", "date_day"}:
            continue
        date_ts: Optional[pd.Timestamp] = None
        if tag != "date_day":
            date_ts = _parse_date_for_age_days(str(item.get("text", "") or ""))
        if date_ts is None:
            if fallback_days_int is not None:
                age_days = fallback_days_int
            else:
                continue
        else:
            age_days = int((date_ts - dob_ts).days)
        # Do not annotate DOB boxes/dates as "age at event".
        if age_days <= 0:
            continue
        bbox = item.get("bbox_xyxy", [0, 0, 0, 0])
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(float(v)) for v in bbox]
        except Exception:
            continue
        xa, xb = sorted([x1, x2])
        ya, yb = sorted([y1, y2])
        if _overlaps_exclusion([int(xa), int(ya), int(xb), int(yb)]):
            continue
        candidates.append(
            {
                "age_days": int(age_days),
                "bbox_xyxy": [int(xa), int(ya), int(xb), int(yb)],
            }
        )

    if not candidates:
        return overlays

    # Merge candidates that likely represent the same entity segmented into
    # multiple nearby/overlapping boxes (common after OCR/line splitting).
    def _boxes_close_or_overlap(a: List[int], b: List[int]) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        if ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1:
            # Not overlapping; allow small proximity merge.
            x_gap = max(0, max(bx1 - ax2, ax1 - bx2))
            y_gap = max(0, max(by1 - ay2, ay1 - by2))
            return x_gap <= 14 and y_gap <= 8
        return True

    clusters: List[Dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda c: (c["age_days"], c["bbox_xyxy"][1], c["bbox_xyxy"][0])):
        matched_idx = None
        for idx, cluster in enumerate(clusters):
            if int(cluster["age_days"]) != int(cand["age_days"]):
                continue
            if _boxes_close_or_overlap(cluster["bbox_xyxy"], cand["bbox_xyxy"]):
                matched_idx = idx
                break
        if matched_idx is None:
            clusters.append(
                {
                    "age_days": int(cand["age_days"]),
                    "bbox_xyxy": list(cand["bbox_xyxy"]),
                }
            )
            continue
        cx1, cy1, cx2, cy2 = clusters[matched_idx]["bbox_xyxy"]
        nx1, ny1, nx2, ny2 = cand["bbox_xyxy"]
        clusters[matched_idx]["bbox_xyxy"] = [
            int(min(cx1, nx1)),
            int(min(cy1, ny1)),
            int(max(cx2, nx2)),
            int(max(cy2, ny2)),
        ]

    for cluster in clusters:
        xa, ya, xb, yb = cluster["bbox_xyxy"]
        box_h = max(8, yb - ya)
        overlays.append(
            {
                "text": f"{int(cluster['age_days'])}d",
                "x": int(max(0, xa + 2)),
                "y": int(max(0, ya + 1)),
                "size": int(max(8, min(20, round(box_h * 0.85)))),
                # High-contrast EU overlay style: non-black text for readability.
                "color": "#00D7FF",
                "stroke_fill": "black",
                "stroke_width": 2,
                "bbox_xyxy": [int(xa), int(ya), int(xb), int(yb)],
            }
        )
    return overlays


def _dob_label_exclusion_regions(page: OCRPage) -> List[List[int]]:
    """
    Build regions around DOB/Born label lines so EU age overlays are not placed
    on DOB redaction spots.
    """
    regions: List[List[int]] = []
    if not page.lines:
        return regions
    pat = re.compile(r"\b(?:d\.?\s*o\.?\s*b\.?|date\s*of\s*birth|birth\s*date|born)\b", flags=re.IGNORECASE)
    page_w = max(1, int(page.width))
    page_h = max(1, int(page.height))
    for ln in page.lines:
        txt = str(ln.text or "")
        if not pat.search(txt):
            continue
        lx1, ly1, lx2, ly2 = [float(v) for v in ln.bbox_xyxy]
        lh = max(1.0, ly2 - ly1)
        # Extend to the right where the DOB value usually appears on same row.
        rx1 = max(0, int(round(lx1 - 8)))
        ry1 = max(0, int(round(ly1 - 0.6 * lh)))
        rx2 = min(page_w, int(round(max(lx2 + 0.45 * page_w, lx2 + 12))))
        ry2 = min(page_h, int(round(ly2 + 0.8 * lh)))
        if rx2 > rx1 and ry2 > ry1:
            regions.append([rx1, ry1, rx2, ry2])
    return regions


def date_spans_to_text_overlays(page: OCRPage, spans: Sequence[PIISpan]) -> List[Dict[str, Any]]:
    """
    For date-like spans, generate text overlays that write normalized date after redaction.
    """
    overlays: List[Dict[str, Any]] = []
    date_like = [s for s in spans if s.tag in {"date", "dob", "date_day", "date_history"}]
    if not date_like:
        return overlays

    for span in date_like:
        normalized = _normalize_date_for_overlay(span.text or page.text[span.start:span.end])
        if not normalized:
            continue
        overlapping_lines = [
            line
            for line in page.lines
            if spans_overlap(span.start, span.end, line.char_start, line.char_end)
        ]
        if not overlapping_lines:
            continue
        boxes: List[Tuple[float, float, float, float]] = []
        for line in overlapping_lines:
            overlap_start = max(span.start, line.char_start)
            overlap_end = min(span.end, line.char_end)
            if overlap_end <= overlap_start:
                continue
            line_len = max(1, line.char_end - line.char_start)
            rel_start = (overlap_start - line.char_start) / line_len
            rel_end = (overlap_end - line.char_start) / line_len
            rel_start = min(max(rel_start, 0.0), 1.0)
            rel_end = min(max(rel_end, 0.0), 1.0)
            if rel_end <= rel_start:
                continue
            x1, y1, x2, y2 = line.bbox_xyxy
            w = max(1e-6, x2 - x1)
            bx1 = x1 + (w * rel_start)
            bx2 = x1 + (w * rel_end)
            boxes.append((bx1, y1, bx2, y2))
        if not boxes:
            continue
        ux1, uy1, ux2, uy2 = union_boxes(boxes)
        height = max(10.0, uy2 - uy1)
        overlays.append(
            {
                "text": normalized,
                "x": int(round(ux1 + 1)),
                "y": int(round(uy1 + 1)),
                "size": int(max(10, min(18, round(height * 0.85)))),
                "color": "white",
                "bbox_xyxy": [int(round(ux1)), int(round(uy1)), int(round(ux2)), int(round(uy2))],
            }
        )
    return overlays


def extract_history_date_spans(page_text: str) -> List[PIISpan]:
    """
    Targeted date extraction for cardiac history/procedure lines where we prefer
    full-date redaction + normalized overlay (MM/01/YYYY).
    """
    spans: List[PIISpan] = []
    if not page_text:
        return spans

    line_ctx_pat = re.compile(
        r"\b(?:list\s+of\s+interventions?|interventions?|surg|surgery|cath|catheterization|history)\b",
        flags=re.IGNORECASE,
    )
    date_pat = re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}[/-]\d{4})\b"
    )

    cursor = 0
    for line in page_text.splitlines(keepends=True):
        line_text = line.rstrip("\r\n")
        line_start = cursor
        if line_ctx_pat.search(line_text):
            for m in date_pat.finditer(line_text):
                gs = line_start + m.start()
                ge = line_start + m.end()
                spans.append(
                    PIISpan(
                        start=gs,
                        end=ge,
                        tag="date_history",
                        text=page_text[gs:ge],
                        source="deterministic_core",
                        reason="history_full_date",
                    )
                )
        cursor += len(line)

    return merge_overlapping_spans(spans)


def _patient_name_context_ok(page_text: str, span: PIISpan) -> bool:
    """
    Reduce over-redaction by requiring patient-name-like spans to appear in
    demographic/footer context, not arbitrary capitalized sections (e.g. meds/stages).
    """
    if span.tag != "patient_name":
        return True
    literal = (span.text or page_text[span.start:span.end]).strip().lower()
    if literal in {"patient name", "name", "demographics"}:
        return False

    start = max(0, int(span.start))
    end = min(len(page_text), int(span.end))
    local = page_text[max(0, start - 180) : min(len(page_text), end + 180)]
    before = page_text[max(0, start - 120) : start]
    after = page_text[end : min(len(page_text), end + 120)]
    line_start = page_text.rfind("\n", 0, start) + 1
    line_end = page_text.find("\n", end)
    if line_end < 0:
        line_end = len(page_text)
    line = page_text[line_start:line_end]

    patient_ctx = re.compile(
        r"\b(?:patient\s*name|mrn|medical\s+record|record\s*(?:#|no\.?|number)|dob|date\s*of\s*birth|born)\b",
        flags=re.IGNORECASE,
    )
    footer_ctx = re.compile(
        r"\b(?:study\s*on|study\s*date|page\s+\d+\s+of\s+\d+)\b",
        flags=re.IGNORECASE,
    )
    demographic_ctx = re.compile(
        r"\b\d{1,3}\s*y\.?\s*o\.?\s*(?:male|female)\b",
        flags=re.IGNORECASE,
    )
    bad_ctx = re.compile(
        r"\b(?:medication|indications|comments|stage|duration|speed|workload|hr|bp|exercise|ectopy|qt)\b",
        flags=re.IGNORECASE,
    )

    has_patient_ctx = bool(patient_ctx.search(local))
    has_footer_ctx = bool(footer_ctx.search(local))
    has_demographic_ctx = bool(demographic_ctx.search(local))
    has_bad_ctx = bool(bad_ctx.search(local))

    # Strong acceptors:
    if has_patient_ctx:
        return True
    if has_demographic_ctx:
        return True
    if has_footer_ctx and ("(" in after[:40] or ")" in after[:40] or "(" in line or ")" in line):
        return True

    # Strong rejector: meds/vitals sections without patient-context anchors.
    if has_bad_ctx and not has_patient_ctx:
        return False

    # Default conservative behavior: only keep if there is some anchor nearby.
    return has_footer_ctx


def _is_probable_stress_test_layout(page_text: str) -> bool:
    """
    Conservative stress-test detector used only to switch date day-order parsing.
    """
    text = str(page_text or "")
    if not text:
        return False
    patterns = (
        r"\bstress\s*test\b",
        r"\btreadmill\b",
        r"\bbruce\s*protocol\b",
        r"\bexercise\s*time\b",
        r"\bpeak\s*hr\b",
        r"\bmets?\b",
        r"\bstage\s+\d+\b",
        r"\brest(?:ing)?\s*(?:hr|bp)\b",
    )
    hits = 0
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            hits += 1
            if hits >= 2:
                return True
    return False


_SIGNATURE_LINE_PATTERN = re.compile(
    r"\b(?:e-?signed|electronically\s+signed|signed\s+electronically)\b",
    flags=re.IGNORECASE,
)


def _span_on_signature_line(page_text: str, span: PIISpan) -> bool:
    if not page_text:
        return False
    start = max(0, int(span.start))
    end = max(start, int(span.end))
    line_start = page_text.rfind("\n", 0, start) + 1
    line_end = page_text.find("\n", end)
    if line_end < 0:
        line_end = len(page_text)
    line = page_text[line_start:line_end]
    return bool(_SIGNATURE_LINE_PATTERN.search(line))


def _extract_signature_day_time_spans(page_text: str) -> List[PIISpan]:
    """
    Signature line hardening:
    - Keep physician identity/titles untouched.
    - Redact only day token(s) and time token(s) in e-signature lines.
    """
    if not page_text:
        return []

    out: List[PIISpan] = []
    lines = list(re.finditer(r"(?m)^.*$", page_text))
    for lm in lines:
        line_start, line_end = lm.start(), lm.end()
        line = page_text[line_start:line_end]
        if not _SIGNATURE_LINE_PATTERN.search(line):
            continue

        # yyyy-mm-dd / yyyy/mm/dd
        for m in re.finditer(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", line):
            ds = line_start + m.start(3)
            de = line_start + m.end(3)
            out.append(
                PIISpan(
                    start=ds,
                    end=de,
                    tag="date_day",
                    text=page_text[ds:de],
                    source="deterministic_core",
                    reason="signature_line_date_day",
                )
            )

        # mm-dd-yyyy / mm/dd/yyyy
        for m in re.finditer(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})\b", line):
            ds = line_start + m.start(2)
            de = line_start + m.end(2)
            out.append(
                PIISpan(
                    start=ds,
                    end=de,
                    tag="date_day",
                    text=page_text[ds:de],
                    source="deterministic_core",
                    reason="signature_line_date_day",
                )
            )

        # Month DD, YYYY (also handles Month DD,YYYY)
        for m in re.finditer(
            r"\b([A-Za-z]{3,9})\s+(\d{1,2})\s*,?\s*(\d{4})\b",
            line,
            flags=re.IGNORECASE,
        ):
            ds = line_start + m.start(2)
            de = line_start + m.end(2)
            out.append(
                PIISpan(
                    start=ds,
                    end=de,
                    tag="date_day",
                    text=page_text[ds:de],
                    source="deterministic_core",
                    reason="signature_line_date_day",
                )
            )

        # DD-MMM-YYYY
        for m in re.finditer(r"\b(\d{1,2})-([A-Za-z]{3})-(\d{4})\b", line, flags=re.IGNORECASE):
            ds = line_start + m.start(1)
            de = line_start + m.end(1)
            out.append(
                PIISpan(
                    start=ds,
                    end=de,
                    tag="date_day",
                    text=page_text[ds:de],
                    source="deterministic_core",
                    reason="signature_line_date_day",
                )
            )

        for m in re.finditer(r"\b((?:[01]?\d|2[0-3]):[0-5]\d(?:\s?[APap][Mm])?)\b", line):
            ts = line_start + m.start(1)
            te = line_start + m.end(1)
            out.append(
                PIISpan(
                    start=ts,
                    end=te,
                    tag="time",
                    text=page_text[ts:te],
                    source="deterministic_core",
                    reason="signature_line_time",
                )
            )
    return out


def build_effective_redaction_spans(
    page_text: str,
    spans: List[PIISpan],
    prefer_day_first_dates: bool = False,
    redact_full_dates: bool = False,
) -> List[PIISpan]:
    effective: List[PIISpan] = []
    for span in spans:
        # Signature e-sign lines should only redact day/time tokens.
        # Skip all raw detector spans on those lines; we add precise day/time below.
        if _span_on_signature_line(page_text, span):
            continue
        if span.tag == "mrn":
            mrn_literal = (span.text or page_text[span.start:span.end] or "").strip()
            if re.search(r"[^\w\- ]", mrn_literal):
                # Final guardrail: any special chars imply non-MRN.
                continue
            if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", mrn_literal):
                # Final guardrail against misclassified date-like strings as MRN.
                continue
        if span.tag == "patient_name":
            literal = (span.text or page_text[span.start:span.end]).strip()
            if _is_patient_label_text(literal):
                continue
            if (
                (span.reason or "") != "patient_name_token_propagation"
                and not _patient_name_context_ok(page_text, span)
            ):
                continue
        effective.extend(
            _extract_date_day_only_spans(
                page_text,
                span,
                prefer_day_first=prefer_day_first_dates,
                redact_full_dates=redact_full_dates,
            )
        )
    # Also redact time only when tied to date-like fields (avoid stage/duration table times).
    time_pat = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?:\s?[APap][Mm])?\b")
    label_time_pat = re.compile(
        r"\b(?:study\s*time|signed\s*on)\b\s*[:\-]?\s*"
        r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}|\d{1,2}-[A-Za-z]{3}-\d{4})?\s*"
        r"(?P<t>(?:[01]?\d|2[0-3]):[0-5]\d(?:\s?[APap][Mm])?)",
        flags=re.IGNORECASE,
    )
    for s in list(effective):
        if s.tag not in {"date", "date_day", "dob"}:
            continue
        look = page_text[s.end : min(len(page_text), s.end + 24)]
        m = time_pat.search(look)
        if m and m.start() <= 8:
            ts = s.end + m.start()
            te = s.end + m.end()
            effective.append(
                PIISpan(
                    start=ts,
                    end=te,
                    tag="time",
                    text=page_text[ts:te],
                    source=s.source,
                    reason="time_near_date",
                )
            )
    for m in label_time_pat.finditer(page_text):
        ts, te = m.start("t"), m.end("t")
        effective.append(
            PIISpan(
                start=ts,
                end=te,
                tag="time",
                text=page_text[ts:te],
                source="deterministic_core",
                reason="time_label_context",
            )
        )
    effective.extend(_extract_signature_day_time_spans(page_text))
    return merge_overlapping_spans(effective)


def normalize_dates_to_first_of_month(text: str) -> str:
    """
    Normalize date-like values so day component becomes 01.
    """
    out = text
    out = re.sub(
        r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b",
        lambda m: f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-01",
        out,
    )
    out = re.sub(
        r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})\b",
        lambda m: f"{int(m.group(3)):04d}-{int(m.group(1)):02d}-01",
        out,
    )
    month_lookup = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }

    def _dd_mmm_yyyy(m: re.Match[str]) -> str:
        mon = month_lookup.get(m.group(2).lower(), 1)
        yr = int(m.group(3))
        return f"{yr:04d}-{mon:02d}-01"

    out = re.sub(
        r"\b(\d{1,2})-([A-Za-z]{3})-(\d{4})\b",
        _dd_mmm_yyyy,
        out,
        flags=re.IGNORECASE,
    )
    return out


def _load_overlay_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=max(1, int(size)))
    except Exception:
        return ImageFont.load_default()


def _measure_overlay_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> Tuple[float, float, float, float]:
    try:
        tx1, ty1, tx2, ty2 = draw.textbbox((0, 0), text, font=font)
        return float(tx1), float(ty1), float(tx2), float(ty2)
    except Exception:
        est_size = float(getattr(font, "size", 12) or 12)
        est_w = max(1.0, est_size * max(1.0, float(len(text))) * 0.55)
        est_h = max(1.0, est_size * 1.2)
        return 0.0, 0.0, est_w, est_h


def _resolve_overlay_text_layout(
    draw: ImageDraw.ImageDraw,
    overlay: Dict[str, Any],
    width: int,
    height: int,
    *,
    default_size: int = 14,
    min_size: int = 8,
    max_size: int = 96,
) -> Optional[Dict[str, Any]]:
    text = str(overlay.get("text", "") or "").strip()
    if not text:
        return None

    req_size = int(float(overlay.get("size", default_size) or default_size))
    req_size = max(min_size, min(req_size, max_size))

    bbox = overlay.get("bbox_xyxy")
    has_bbox = isinstance(bbox, (list, tuple)) and len(bbox) == 4
    if has_bbox:
        bx1, by1, bx2, by2 = [int(float(v)) for v in bbox]
        xa, xb = sorted([bx1, bx2])
        ya, yb = sorted([by1, by2])
        xa = max(0, min(xa, width))
        xb = max(0, min(xb, width))
        ya = max(0, min(ya, height))
        yb = max(0, min(yb, height))
        box_w = max(1, xb - xa)
        box_h = max(1, yb - ya)
        target_w = max(1.0, box_w * 0.88)
        target_h = max(1.0, box_h * 0.82)
        start_size = min(max_size, max(req_size, int(round(box_h * 0.9))))
        fit_size = max(min_size, min(start_size, max_size))
        fit_font = _load_overlay_font(fit_size)
        fit_bbox = _measure_overlay_text(draw, text, fit_font)
        for cand in range(start_size, min_size - 1, -1):
            cand_font = _load_overlay_font(cand)
            tx1, ty1, tx2, ty2 = _measure_overlay_text(draw, text, cand_font)
            tw = max(1.0, tx2 - tx1)
            th = max(1.0, ty2 - ty1)
            if tw <= target_w and th <= target_h:
                fit_size = cand
                fit_font = cand_font
                fit_bbox = (tx1, ty1, tx2, ty2)
                break
        tx1, ty1, tx2, ty2 = fit_bbox
        tw = max(1.0, tx2 - tx1)
        th = max(1.0, ty2 - ty1)
        x = int(round(xa + (box_w - tw) / 2.0 - tx1))
        y = int(round(ya + (box_h - th) / 2.0 - ty1))
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        return {
            "text": text,
            "x": x,
            "y": y,
            "size": int(fit_size),
            "font": fit_font,
            "bbox_xyxy": [int(xa), int(ya), int(xb), int(yb)],
            "has_bbox": True,
        }

    x = int(float(overlay.get("x", 0) or 0))
    y = int(float(overlay.get("y", 0) or 0))
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    font = _load_overlay_font(req_size)
    return {
        "text": text,
        "x": x,
        "y": y,
        "size": int(req_size),
        "font": font,
        "bbox_xyxy": [int(x), int(y), int(x), int(y)],
        "has_bbox": False,
    }


def apply_redaction_boxes(
    image: Image.Image,
    boxes: Sequence[RedactionBox],
    pad_ratio_x: float = DEFAULT_PAD_RATIO_X,
    pad_ratio_y: float = DEFAULT_PAD_RATIO_Y,
    text_overlays: Optional[Sequence[Dict[str, Any]]] = None,
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    width, height = output.size
    for box in boxes:
        x1, y1, x2, y2 = box.bbox_xyxy
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        # Adaptive padding to avoid oversized redaction bars.
        pad_x = min(max(0.0, pad_ratio_x * width), max(1.0, box_w * 0.08))
        pad_y = min(max(0.0, pad_ratio_y * height), max(1.0, box_h * 0.15))
        if box.tag == "date_day":
            # Keep day-only redactions very tight; avoid swallowing month/day separators.
            pad_x = min(pad_x, 0.35)
            pad_y = min(pad_y, 0.9)
        if box.tag in {"footer_phi", "footer_stamp_proxy", "footer_stamp_template"}:
            # Footer stamp boxes are already tight; keep padding minimal.
            pad_x = min(pad_x, 1.0)
            pad_y = min(pad_y, 1.0)
        draw.rectangle(
            [max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)],
            fill="black",
        )
    for overlay in list(text_overlays or []):
        layout = _resolve_overlay_text_layout(
            draw,
            overlay,
            width,
            height,
            default_size=14,
            min_size=8,
            max_size=48,
        )
        if not layout:
            continue
        text = str(layout["text"])
        x = int(layout["x"])
        y = int(layout["y"])
        size = int(layout["size"])
        font = layout["font"]
        color = str(overlay.get("color", "white") or "white")
        bg = str(overlay.get("bg", "") or "").strip().lower()
        stroke_fill = str(overlay.get("stroke_fill", "") or "").strip()
        stroke_width_raw = overlay.get("stroke_width", 0)
        try:
            stroke_width = max(0, int(float(stroke_width_raw)))
        except Exception:
            stroke_width = 0
        if bg:
            try:
                tx1, ty1, tx2, ty2 = draw.textbbox((x, y), text, font=font)
            except Exception:
                tx1, ty1, tx2, ty2 = (
                    x,
                    y,
                    x + max(1, int(size * max(1, len(text)) * 0.55)),
                    y + max(1, int(size * 1.2)),
                )
            pad = max(1, int(round(size * 0.12)))
            draw.rectangle([tx1 - pad, ty1 - pad, tx2 + pad, ty2 + pad], fill=bg)
        draw.text(
            (x, y),
            text,
            fill=color,
            font=font,
            stroke_width=stroke_width,
            stroke_fill=(stroke_fill or None),
        )
    return output



def refine_date_day_boxes_with_image(
    image: Image.Image,
    boxes: Sequence[RedactionBox],
) -> List[RedactionBox]:
    """
    Date-focused pass-2 refinement.
    Tightens `date_day` boxes to local rendered ink so day masking is more precise.
    """
    if not boxes:
        return []

    gray = np.array(image.convert("L"))
    if gray.size == 0:
        return list(boxes)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_h, img_w = ink.shape[:2]

    out: List[RedactionBox] = []
    for box in boxes:
        if box.tag != "date_day":
            out.append(box)
            continue

        x1, y1, x2, y2 = box.bbox_xyxy
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = (x1 + x2) / 2.0

        sx1 = int(max(0, np.floor(x1 - 0.9 * bw)))
        sx2 = int(min(img_w, np.ceil(x2 + 0.9 * bw)))
        sy1 = int(max(0, np.floor(y1 - 0.35 * bh)))
        sy2 = int(min(img_h, np.ceil(y2 + 0.35 * bh)))
        if sx2 <= sx1 or sy2 <= sy1:
            out.append(box)
            continue

        roi = ink[sy1:sy2, sx1:sx2]
        ys, xs = np.where(roi > 0)
        if xs.size < 8:
            out.append(box)
            continue

        abs_x = xs.astype(np.float32) + float(sx1)
        abs_y = ys.astype(np.float32) + float(sy1)

        # Keep only glyphs near current day center to avoid grabbing month/year digits.
        day_digits = re.findall(r"[0-9Il]", str(box.text or ""))
        keep = np.abs(abs_x - cx) <= max(6.0, (1.55 if len(day_digits) >= 2 else 1.25) * bw)
        if int(np.count_nonzero(keep)) < 6:
            out.append(box)
            continue
        abs_x = abs_x[keep]
        abs_y = abs_y[keep]

        nx1 = max(0.0, float(np.min(abs_x)) - 1.0)
        nx2 = min(float(img_w), float(np.max(abs_x)) + 1.0)
        ny1 = max(0.0, float(np.min(abs_y)) - 1.0)
        ny2 = min(float(img_h), float(np.max(abs_y)) + 1.0)
        new_w = max(1.0, nx2 - nx1)

        # Guardrails: keep realistic adjustment sizes only.
        if new_w > (2.4 * bw) or new_w < (0.35 * bw):
            out.append(box)
            continue
        if len(day_digits) >= 2:
            # Minimal right-side guard band for 2-digit day values.
            nx2 = min(float(img_w), nx2 + max(0.6, 0.06 * bw))

        out.append(
            RedactionBox(
                page_index=box.page_index,
                tag=box.tag,
                text=box.text,
                bbox_xyxy=(nx1, ny1, nx2, ny2),
            )
        )

    return out


def refine_date_day_boxes_pass3(
    image: Image.Image,
    page: OCRPage,
    boxes: Sequence[RedactionBox],
) -> List[RedactionBox]:
    """
    Date-focused pass-3 refinement.
    Re-anchors date-day boxes to nearest OCR line token geometry and then
    snaps tightly to local rendered ink near that anchor.
    """
    if not boxes:
        return []
    gray = np.array(image.convert("L"))
    if gray.size == 0:
        return list(boxes)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_h, img_w = ink.shape[:2]

    out: List[RedactionBox] = []
    for box in boxes:
        if box.tag != "date_day":
            out.append(box)
            continue

        x1, y1, x2, y2 = box.bbox_xyxy
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        day_token = re.sub(r"\D", "", str(box.text or ""))
        if not day_token:
            out.append(box)
            continue

        best_line: Optional[OCRLine] = None
        best_idx: Optional[Tuple[int, int]] = None
        best_bias_chars: float = 0.0
        best_month_name: bool = False
        best_score: Optional[float] = None

        for ln in page.lines:
            lx1, ly1, lx2, ly2 = ln.bbox_xyxy
            # Require line to be in similar vertical band.
            if (ly2 < (cy - 2.2 * bh)) or (ly1 > (cy + 2.2 * bh)):
                continue
            line_text = str(page.text[ln.char_start:ln.char_end] if ln.char_end > ln.char_start else ln.text or "")
            if not line_text:
                continue
            local_candidates: List[Tuple[int, int, float, bool]] = []
            # Preferred: exact day-token occurrences.
            for m in re.finditer(rf"\b{re.escape(day_token)}\b", line_text):
                local_candidates.append((m.start(), m.end(), 0.0, False))
            # Fallback: parse explicit date structure and use its day group.
            if not local_candidates:
                dm = re.search(
                    r"\b(?:(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})|(\d{4})[/-](\d{1,2})[/-](\d{1,2})|([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4}))\b",
                    line_text,
                    flags=re.IGNORECASE,
                )
                if dm:
                    day_span: Optional[Tuple[int, int]] = None
                    bias_chars = 0.0
                    month_name_form = False
                    if dm.group(2):
                        day_span = (dm.start(2), dm.end(2))
                    elif dm.group(6):
                        day_span = (dm.start(6), dm.end(6))
                    elif dm.group(8):
                        month_name_form = True
                        # Month-name form: prefer token immediately before comma/year.
                        body_s = dm.start(7) + len(str(dm.group(7) or ""))
                        body_e = dm.start(9)
                        between = line_text[body_s:body_e]
                        cands: List[re.Match[str]] = []
                        comma_i = between.rfind(",")
                        if comma_i >= 0:
                            left = between[:comma_i]
                            cands = list(re.finditer(r"\b(\d{1,2}|[Il])\b", left))
                        if not cands:
                            cands = list(re.finditer(r"\b(\d{1,2}|[Il])\b", between))
                        if cands:
                            c = cands[-1]
                            day_span = (body_s + c.start(1), body_s + c.end(1))
                        else:
                            day_span = (dm.start(8), dm.end(8))
                        month_len = len(str(dm.group(7) or "").strip())
                        if month_len > 3:
                            bias_chars = float(month_len - 3) * 0.42
                    if day_span is not None:
                        local_candidates.append((day_span[0], day_span[1], bias_chars, month_name_form))
            for ds, de, bias_chars, month_name_form in local_candidates:
                rel_mid = (ds + de) / 2.0 / max(1, len(line_text))
                cand_cx = float(lx1) + (float(lx2 - lx1) * rel_mid)
                score = abs(cand_cx - cx) + (0.35 * abs(((ly1 + ly2) / 2.0) - cy))
                if best_score is None or score < best_score:
                    best_score = score
                    best_line = ln
                    best_idx = (ds, de)
                    best_bias_chars = float(bias_chars)
                    best_month_name = bool(month_name_form)

        if best_line is None or best_idx is None:
            out.append(box)
            continue

        ls, le = best_idx
        lx1, ly1, lx2, ly2 = best_line.bbox_xyxy
        line_text = str(page.text[best_line.char_start:best_line.char_end] if best_line.char_end > best_line.char_start else best_line.text or "")
        line_len = max(1, len(line_text))
        rel_start = ls / line_len
        rel_end = le / line_len
        pred_x1 = float(lx1) + (float(lx2 - lx1) * rel_start)
        pred_x2 = float(lx1) + (float(lx2 - lx1) * rel_end)
        if best_bias_chars > 0:
            char_w = max(1.0, float(lx2 - lx1)) / max(1, line_len)
            shift = best_bias_chars * char_w
            pred_x1 += shift
            pred_x2 += shift
        pred_w = max(1.0, pred_x2 - pred_x1)
        pred_cx = (pred_x1 + pred_x2) / 2.0

        # Local ink snap around predicted day token region.
        sx1 = int(max(0, np.floor(pred_x1 - max(8.0, 0.8 * pred_w))))
        sx2 = int(min(img_w, np.ceil(pred_x2 + max(8.0, 0.8 * pred_w))))
        sy1 = int(max(0, np.floor(float(ly1) - max(2.0, 0.20 * (ly2 - ly1)))))
        sy2 = int(min(img_h, np.ceil(float(ly2) + max(2.0, 0.20 * (ly2 - ly1)))))

        nx1 = pred_x1
        nx2 = pred_x2
        ny1 = float(ly1)
        ny2 = float(ly2)
        if sx2 > sx1 and sy2 > sy1:
            roi = ink[sy1:sy2, sx1:sx2]
            nlab, labels, stats, _ = cv2.connectedComponentsWithStats((roi > 0).astype(np.uint8), connectivity=8)
            comp_boxes: List[Tuple[float, float, float, float]] = []
            for i in range(1, int(nlab)):
                x = int(stats[i, cv2.CC_STAT_LEFT]) + sx1
                y = int(stats[i, cv2.CC_STAT_TOP]) + sy1
                w = int(stats[i, cv2.CC_STAT_WIDTH])
                h = int(stats[i, cv2.CC_STAT_HEIGHT])
                if w <= 0 or h <= 0:
                    continue
                ccx = x + (w / 2.0)
                # Keep glyphs near predicted token center.
                if abs(ccx - pred_cx) > max(7.0, 1.15 * pred_w):
                    continue
                comp_boxes.append((float(x), float(y), float(x + w), float(y + h)))
            if comp_boxes:
                nx1 = min(b[0] for b in comp_boxes) - 1.0
                nx2 = max(b[2] for b in comp_boxes) + 1.0
                ny1 = min(float(ly1), min(b[1] for b in comp_boxes) - 1.0)
                ny2 = max(float(ly2), max(b[3] for b in comp_boxes) + 1.0)

        # Clamp final width to prevent oversized bars and left/right drift.
        new_w = max(1.0, nx2 - nx1)
        min_w = max(1.0, 0.45 * bw)
        max_w = max(1.0, min(1.20 * bw, pred_w * 1.35))
        if new_w < min_w or new_w > max_w:
            nx1 = pred_x1
            nx2 = pred_x2
            ny1 = float(ly1)
            ny2 = float(ly2)

        # Month-name explicit second pass:
        # for 2-digit day values, search within the refined day box neighborhood and
        # pull in a missing second glyph if OCR geometry clipped it.
        if best_month_name and len(day_token) >= 2:
            char_w = max(1.0, float(lx2 - lx1) / max(1, line_len))
            tx1 = int(max(0, np.floor(nx1 - 0.20 * char_w)))
            tx2 = int(min(img_w, np.ceil(nx2 + 1.30 * char_w)))
            ty1 = int(max(0, np.floor(ny1 - max(1.0, 0.20 * (ly2 - ly1)))))
            ty2 = int(min(img_h, np.ceil(ny2 + max(1.0, 0.20 * (ly2 - ly1)))))
            if tx2 > tx1 and ty2 > ty1:
                roi2 = ink[ty1:ty2, tx1:tx2]
                nlab2, _labels2, stats2, _cent2 = cv2.connectedComponentsWithStats((roi2 > 0).astype(np.uint8), connectivity=8)
                ext_boxes: List[Tuple[float, float, float, float]] = []
                for i in range(1, int(nlab2)):
                    x = int(stats2[i, cv2.CC_STAT_LEFT]) + tx1
                    y = int(stats2[i, cv2.CC_STAT_TOP]) + ty1
                    w = int(stats2[i, cv2.CC_STAT_WIDTH])
                    h = int(stats2[i, cv2.CC_STAT_HEIGHT])
                    if w <= 0 or h <= 0:
                        continue
                    ccy = y + (h / 2.0)
                    ccx = x + (w / 2.0)
                    if ccy < (ny1 - 0.45 * bh) or ccy > (ny2 + 0.45 * bh):
                        continue
                    if ccx < (nx1 - 0.35 * char_w) or ccx > (nx2 + 1.80 * char_w):
                        continue
                    ext_boxes.append((float(x), float(y), float(x + w), float(y + h)))
                if ext_boxes:
                    nx1 = min(nx1, min(b[0] for b in ext_boxes) - 0.5)
                    nx2 = max(nx2, max(b[2] for b in ext_boxes) + 0.8)
                    ny1 = min(ny1, min(b[1] for b in ext_boxes) - 0.5)
                    ny2 = max(ny2, max(b[3] for b in ext_boxes) + 0.5)

        nx1 = max(0.0, min(float(img_w), nx1))
        nx2 = max(0.0, min(float(img_w), nx2))
        ny1 = max(0.0, min(float(img_h), ny1))
        ny2 = max(0.0, min(float(img_h), ny2))
        if nx2 <= nx1 or ny2 <= ny1:
            out.append(box)
            continue
        out.append(
            RedactionBox(
                page_index=box.page_index,
                tag=box.tag,
                text=box.text,
                bbox_xyxy=(nx1, ny1, nx2, ny2),
            )
        )
    return out


def save_pdf_from_images(images: Sequence[Image.Image], output_pdf: Path) -> None:
    if not images:
        raise ValueError("No images provided for PDF creation")

    doc = fitz.open()
    try:
        for image in images:
            rgb = image.convert("RGB")
            buf = io.BytesIO()
            rgb.save(buf, format="PNG")  # avoids JPEG encoder
            img_bytes = buf.getvalue()

            page = doc.new_page(width=rgb.width, height=rgb.height)
            page.insert_image(fitz.Rect(0, 0, rgb.width, rgb.height), stream=img_bytes)

        doc.save(str(output_pdf))
    finally:
        doc.close()



def _save_page_cache(page: OCRPage, path: Path) -> None:
    payload = {
        "page_index": page.page_index,
        "width": page.width,
        "height": page.height,
        "text": page.text,
        "lines": [asdict(line) for line in page.lines],
        "extraction_method": page.extraction_method,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_page_cache(path: Path) -> OCRPage:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return OCRPage(
        page_index=int(payload["page_index"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        text=payload["text"],
        lines=[
            OCRLine(
                text=line["text"],
                confidence=float(line["confidence"]),
                polygon=line.get("polygon", []),
                bbox_xyxy=tuple(line["bbox_xyxy"]),
                char_start=int(line["char_start"]),
                char_end=int(line["char_end"]),
            )
            for line in payload["lines"]
        ],
        extraction_method=payload.get("extraction_method", "ocr"),
    )


def _build_page(
    page_idx: int,
    image: Image.Image,
    cache_ocr_page_json: Path,
    dpi: int,
    pdf_doc: Optional[fitz.Document],
    overwrite: bool,
    ocr_backend: str,
    pdf_text_mode: str,
) -> OCRPage:
    if cache_ocr_page_json.exists() and not overwrite:
        return _load_page_cache(cache_ocr_page_json)

    page: Optional[OCRPage] = None
    allow_pdf_text = str(pdf_text_mode or "hybrid_pdf_text").strip().lower() != "ocr_only"
    if pdf_doc is not None and allow_pdf_text:
        try:
            page = _pdf_page_to_ocr_page(pdf_doc, page_idx, dpi=dpi)
            if len(page.text.strip()) < 8:
                page = None
        except Exception:
            page = None

    if page is None:
        temp_img = cache_ocr_page_json.with_suffix(".png")
        image.save(temp_img)
        page = run_ocr_file_to_page(temp_img, page_index=page_idx, ocr_backend=ocr_backend)
    elif page.extraction_method == "pdf_text":
        # Hybrid augment: merge OCR footer lines to catch stamped footer PHI not in PDF text layer.
        try:
            temp_img = cache_ocr_page_json.with_suffix(".png")
            image.save(temp_img)
            ocr_footer = run_ocr_file_to_page(temp_img, page_index=page_idx, ocr_backend=ocr_backend)
            page = _merge_bottom_footer_ocr(page, ocr_footer)
        except Exception:
            pass

    _save_page_cache(page, cache_ocr_page_json)
    return page


def _find_phrase_spans(text: str, phrases: Sequence[str]) -> List[PIISpan]:
    spans: List[PIISpan] = []
    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase:
            continue
        for match in re.compile(re.escape(phrase), flags=re.IGNORECASE).finditer(text):
            spans.append(
                PIISpan(
                    start=match.start(),
                    end=match.end(),
                    tag="manual",
                    text=match.group(0),
                )
            )
    return merge_overlapping_spans(spans)


def _detect_name_label_spans(page_text: str) -> List[PIISpan]:
    """
    Deterministic fallback for templates that present "Name: <value>" on header lines.
    """
    spans: List[PIISpan] = []
    if not page_text:
        return spans
    patt = re.compile(
        r"\b(?:patient\s*name|name)\s*[:\-]\s*(?P<val>[A-Za-z][A-Za-z ,.'`-]{2,80})",
        flags=re.IGNORECASE,
    )
    stop_pat = re.compile(
        r"\b(?:mrn|record|dob|date\s*of\s*birth|study\s*time|signed\s*on|age|gender|sex|height|weight|bsa)\b",
        flags=re.IGNORECASE,
    )
    for m in patt.finditer(page_text):
        raw = str(m.group("val") or "")
        raw = stop_pat.split(raw)[0].strip(" ,;:-")
        if not raw:
            continue
        # Must look like at least First Last.
        parts = [p for p in re.split(r"[\s,]+", raw) if p]
        if len(parts) < 2:
            continue
        if any(any(ch.isdigit() for ch in p) for p in parts):
            continue
        start = m.start("val")
        end = start + len(raw)
        spans.append(
            PIISpan(
                start=start,
                end=end,
                tag="patient_name",
                text=page_text[start:end],
                source="deterministic_header",
                reason="name_label_value",
            )
        )
    return merge_overlapping_spans(spans)


def _detect_demographic_summary_name_spans(page_text: str) -> List[PIISpan]:
    """
    Fallback for summary lines like:
    "Bridget Mairs - 26 y.o. female; born Dec. 1999"
    """
    spans: List[PIISpan] = []
    if not page_text:
        return spans
    patt = re.compile(
        r"(?im)^\s*(?P<val>[A-Za-z][A-Za-z ,.'`-]{1,80})\s*(?:,|-)?\s*"
        r"(?:is\s+a\s+)?\d{1,3}\s*y\.?\s*o\.?\s*(?:male|female)\b[^\n\r]{0,80}\b(?:born|dob|date\s*of\s*birth)\b"
    )
    inline_patt = re.compile(
        r"(?i)(?P<val>[A-Za-z][A-Za-z ,.'`-]{1,80})\s*(?:,|-)?\s*"
        r"(?:is\s+a\s+)?\d{1,3}\s*y\.?\s*o\.?\s*(?:male|female)\b.{0,80}?\b(?:born|dob|date\s*of\s*birth)\b"
    )
    age_gender_patt = re.compile(
        r"(?i)(?:assessment\s*/\s*plan|hpi|history\s+of\s+present\s+illness)?\s*[:\-]?\s*"
        r"(?P<val>[A-Za-z][A-Za-z ,.'`-]{1,80})\s+is\s+a\s+\d{1,3}\s*y\.?\s*o\.?\s*(?:male|female)\b"
    )
    stop_pat = re.compile(
        r"\b(?:mrn|record|study\s*time|signed\s*on|age|gender|sex|height|weight|bsa|"
        r"encounter\s*summary|op\s*visit|progress\s*notes)\b",
        flags=re.IGNORECASE,
    )
    for m in list(patt.finditer(page_text)) + list(inline_patt.finditer(page_text)) + list(age_gender_patt.finditer(page_text)):
        raw = str(m.group("val") or "")
        raw = stop_pat.split(raw)[0].strip(" ,;:-")
        if not raw:
            continue
        # Keep only person-like fragments.
        parts = [p for p in re.split(r"[\s,]+", raw) if p]
        if len(parts) < 1 or len(parts) > 4:
            continue
        if any(any(ch.isdigit() for ch in p) for p in parts):
            continue
        low = raw.lower()
        if low in {"patient", "name", "female", "male"}:
            continue
        if re.search(r"\b(?:hospital|clinic|department|service|program|team|study|exam|test)\b", low):
            continue
        start = m.start("val")
        end = start + len(raw)
        spans.append(
            PIISpan(
                start=start,
                end=end,
                tag="patient_name",
                text=page_text[start:end],
                source="deterministic_header",
                reason="demographic_summary_name",
            )
        )
    return merge_overlapping_spans(spans)


def _sweep_patient_name_leak_spans(page_text: str, spans: Sequence[PIISpan]) -> List[PIISpan]:
    """
    Post-detection sweep:
    - Re-find full patient-name literals anywhere on the page.
    - Re-find first/last tokens in patient-context lines.
    This catches OCR-template misses where a name appears multiple times.
    """
    if not page_text:
        return []
    existing = {(int(s.start), int(s.end)) for s in spans if s.tag == "patient_name"}
    additions: List[PIISpan] = []

    full_literals: set[str] = set()
    tokens: set[str] = set()
    for s in spans:
        if s.tag != "patient_name":
            continue
        lit = (s.text or page_text[s.start:s.end] or "").strip()
        if not lit or _is_patient_label_text(lit):
            continue
        norm = re.sub(r"\s+", " ", lit).strip(" ,;:-")
        if not norm:
            continue
        parts = re.findall(r"[A-Za-z][A-Za-z'`-]+", norm)
        if len(parts) >= 2:
            full_literals.add(norm)
            for tok in (parts[0], parts[-1]):
                t = tok.strip()
                if len(t) >= 4 and t.lower() not in {"patient", "name"}:
                    tokens.add(t)
        elif len(parts) == 1:
            t = parts[0].strip()
            if len(t) >= 4 and t.lower() not in {"patient", "name", "female", "male"}:
                # Single-token OCR names from demographic summary lines should still sweep.
                tokens.add(t)

    for lit in sorted(full_literals, key=len, reverse=True):
        patt = re.compile(re.escape(lit).replace(r"\ ", r"\s+"), flags=re.IGNORECASE)
        for m in patt.finditer(page_text):
            key = (int(m.start()), int(m.end()))
            if key in existing:
                continue
            existing.add(key)
            additions.append(
                PIISpan(
                    start=int(m.start()),
                    end=int(m.end()),
                    tag="patient_name",
                    text=page_text[m.start():m.end()],
                    source="cross_page_forced",
                    reason="patient_name_token_propagation",
                )
            )

    ctx_pat = re.compile(
        r"\b(?:patient\s*name|mrn|medical\s*record|record\s*(?:#|no\.?|number)|dob|born|"
        r"female|male|y\.?\s*o\.?|encounter\s*summary|demographics|op\s*visit|assessment\s*/\s*plan|hpi)\b",
        flags=re.IGNORECASE,
    )
    for tok in sorted(tokens, key=len, reverse=True):
        patt = re.compile(rf"\b{re.escape(tok)}\b", flags=re.IGNORECASE)
        for m in patt.finditer(page_text):
            key = (int(m.start()), int(m.end()))
            if key in existing:
                continue
            line_start = page_text.rfind("\n", 0, m.start()) + 1
            line_end = page_text.find("\n", m.end())
            if line_end < 0:
                line_end = len(page_text)
            line = page_text[line_start:line_end]
            if not ctx_pat.search(line):
                continue
            existing.add(key)
            additions.append(
                PIISpan(
                    start=int(m.start()),
                    end=int(m.end()),
                    tag="patient_name",
                    text=page_text[m.start():m.end()],
                    source="cross_page_forced",
                    reason="patient_name_token_propagation",
                )
            )
    return merge_overlapping_spans(additions)


def _detect_footer_name_spans(page_text: str) -> List[PIISpan]:
    """
    Footer fallback for templates that emit patient identity as:
    - "Last, First [Middle/Initial] - <MRN>"
    - "First Last - <MRN>"
    This seeds patient_name spans so later leak-sweep can propagate same literals
    to demographics/header regions when read-order is jumbled.
    """
    if not page_text:
        return []
    spans: List[PIISpan] = []
    existing: set[tuple[int, int]] = set()
    cursor = 0
    pat_comma = re.compile(
        r"(?i)^\s*(?P<name>[A-Za-z][A-Za-z'`\- ]{0,40},\s*[A-Za-z][A-Za-z'`\- ]{0,40})\s*-\s*(?P<mrn>\d{4,})\s*$"
    )
    pat_space = re.compile(
        r"(?i)^\s*(?P<name>[A-Za-z][A-Za-z'`\- ]{1,40}\s+[A-Za-z][A-Za-z'`\- ]{1,40})\s*-\s*(?P<mrn>\d{4,})\s*$"
    )
    for raw_line in page_text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        line_start = cursor
        cursor += len(raw_line)
        match = pat_comma.search(line) or pat_space.search(line)
        if not match:
            continue
        name_raw = str(match.group("name") or "").strip()
        if not name_raw or _is_patient_label_text(name_raw):
            continue
        start = int(line_start + match.start("name"))
        end = int(line_start + match.end("name"))
        key = (start, end)
        if key in existing:
            continue
        existing.add(key)
        spans.append(
            PIISpan(
                start=start,
                end=end,
                tag="patient_name",
                text=page_text[start:end],
                source="deterministic_footer",
                reason="footer_name_mrn_line",
            )
        )
    return merge_overlapping_spans(spans)


def _normalize_forced_literal(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _forced_name_match_patterns(literal: str) -> List[str]:
    """
    Build robust regex patterns for patient-name carry-forward matching.
    Handles flexible whitespace and common "Last, First" vs "First Last" variants.
    """
    patterns: List[str] = []
    norm = _normalize_forced_literal(literal)
    if not norm:
        return patterns

    # Base literal with OCR-tolerant whitespace.
    base = re.escape(norm).replace(r"\ ", r"\s+")
    patterns.append(rf"\b{base}\b")

    parts = re.findall(r"[A-Za-z][A-Za-z'`-]+", norm)
    if len(parts) >= 2:
        first = re.escape(parts[0])
        last = re.escape(parts[-1])
        # First Last (allow middle noise between).
        patterns.append(rf"\b{first}\b[\s,.;:/-]+(?:[A-Za-z][A-Za-z'`-]+\s+)?\b{last}\b")
        # Last, First
        patterns.append(rf"\b{last}\b[\s,.;:/-]+\b{first}\b")

    # Stable de-duplication.
    out: List[str] = []
    seen: set[str] = set()
    for p in patterns:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _collect_forced_literals_from_spans(spans: Sequence[PIISpan]) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {
        "patient_name": set(),
        "mrn": set(),
        "dob": set(),
        "date": set(),
    }
    for s in spans:
        if s.tag not in out:
            continue
        lit = _normalize_forced_literal(s.text or "")
        if not lit:
            continue
        # Ignore obvious label-like text.
        if s.tag == "patient_name" and _is_patient_label_text(lit):
            continue
        if len(lit) < 2:
            continue
        out[s.tag].add(lit)
    return out


def _merge_forced_literals(dst: Dict[str, set[str]], src: Dict[str, set[str]]) -> Dict[str, set[str]]:
    for k, vals in src.items():
        if k not in dst:
            dst[k] = set()
        dst[k].update(vals)
    return dst


def _forced_literal_spans_for_page(page_text: str, forced_literals: Dict[str, set[str]]) -> List[PIISpan]:
    spans: List[PIISpan] = []
    for tag in ("patient_name", "mrn", "dob", "date"):
        values = sorted(list(forced_literals.get(tag, set())), key=len, reverse=True)
        for lit in values:
            if not lit:
                continue
            if tag == "patient_name":
                patterns = _forced_name_match_patterns(lit)
            else:
                patterns = [re.escape(lit).replace(r"\ ", r"\s+")]
            for patt_str in patterns:
                patt = re.compile(patt_str, flags=re.IGNORECASE)
                for m in patt.finditer(page_text):
                    text_val = page_text[m.start():m.end()]
                    # Extra guard for patient label text.
                    if tag == "patient_name" and _is_patient_label_text(text_val):
                        continue
                    spans.append(
                        PIISpan(
                            start=m.start(),
                            end=m.end(),
                            tag=tag,
                            text=text_val,
                            source="cross_page_forced",
                            reason="forced_literal_carry_forward",
                        )
                    )
    return merge_overlapping_spans(spans)


def _capture_raw_dob_from_page_text(page_text: str) -> str:
    """
    Capture a raw DOB candidate from pre-redaction page text using DOB-label
    proximity (supports split/reversed table order). This is used as a durable
    raw anchor when later text-order inference is noisy.
    """
    text = str(page_text or "")
    if not text:
        return ""
    lines = [str(ln or "").strip() for ln in text.splitlines() if str(ln or "").strip()]
    if not lines:
        return ""

    dob_label_pat = re.compile(r"\b(?:d\.?\s*o\.?\s*b\.?|date\s*of\s*b(?:i)?rth|date\s*of\s*brth|born)\b", re.I)
    studyish_pat = re.compile(
        r"\b(?:study\s*time|signed\s*on|study\s*on|performed\s*on|exam\s*date|service\s*date)\b",
        re.I,
    )
    date_pat = re.compile(
        r"(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|"
        r"\d{1,2}-[A-Za-z]{3,9}-\d{2,4}|"
        r"[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}|"
        r"[A-Za-z]{3,9}\.?\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{4}|"
        r"[A-Za-z]{3,9}\.?\s+\d{4}|"
        r"[A-Za-z]{3,9}\.?\s*[-/]\s*\d{4})",
        re.I,
    )

    # 1) Prefer local window around DOB label lines.
    for i, ln in enumerate(lines):
        if not dob_label_pat.search(ln):
            continue
        for j in range(max(0, i - 2), min(len(lines), i + 3)):
            probe = lines[j]
            if studyish_pat.search(probe):
                continue
            for m in date_pat.finditer(probe):
                cand = str(m.group(1) or "").strip()
                if _parse_date_for_age_days(cand) is not None:
                    return cand

    # 2) Reversed order: date line adjacent to DOB label line.
    for i, ln in enumerate(lines):
        if studyish_pat.search(ln):
            continue
        prev_ln = lines[i - 1] if i > 0 else ""
        next_ln = lines[i + 1] if i + 1 < len(lines) else ""
        if not (dob_label_pat.search(prev_ln) or dob_label_pat.search(next_ln)):
            continue
        for m in date_pat.finditer(ln):
            cand = str(m.group(1) or "").strip()
            if _parse_date_for_age_days(cand) is not None:
                return cand
    return ""


def process_single_document(
    file_path: Path,
    output_root: Path,
    cache_root: Path,
    detector: BasePHIDetector,
    detector_device: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
    overwrite: bool = True,
    ocr_backend: str = "paddle",
    pdf_text_mode: str = "hybrid_pdf_text",
    eu_mode: bool = False,
    full_date_overlay_mode: bool = False,
) -> Dict[str, Any]:
    allowed_tags = {"patient_name", "mrn", "date", "dob"}
    doc_name = safe_stem(file_path)
    doc_dir = ensure_dir(output_root / doc_name)
    cache_dir = ensure_dir(cache_root / doc_name)
    review_pages_dir = ensure_dir(doc_dir / "review_pages")
    ocr_dir = ensure_dir(cache_dir / "ocr")
    ocr_backend = _normalize_ocr_backend(ocr_backend)
    pdf_text_mode = str(pdf_text_mode or "hybrid_pdf_text").strip().lower()
    if pdf_text_mode not in {"hybrid_pdf_text", "ocr_only"}:
        raise ValueError("pdf_text_mode must be one of: hybrid_pdf_text, ocr_only")

    logger.info(
        "Processing %s with detector=%s ocr=%s pdf_text_mode=%s",
        file_path.name,
        detector.detector_name,
        ocr_backend,
        pdf_text_mode,
    )
    pages = load_document_pages(file_path, dpi=dpi)
    redacted_pages: List[Image.Image] = []
    page_summaries: List[Dict[str, Any]] = []
    total_spans = 0
    total_boxes = 0
    start_time = time.perf_counter()
    footer_template_rel: Optional[Tuple[float, float, float, float]] = None
    doc_dob_value: str = ""
    doc_raw_dob_captured: str = ""
    forced_literals: Dict[str, set[str]] = {
        "patient_name": set(),
        "mrn": set(),
        "dob": set(),
        "date": set(),
    }

    pdf_doc = fitz.open(file_path) if file_path.suffix.lower() == ".pdf" else None
    for page_idx, image in enumerate(pages):
        ocr_json = ocr_dir / f"page_{page_idx + 1:04d}.json"
        page = _build_page(
            page_idx=page_idx,
            image=image,
            cache_ocr_page_json=ocr_dir / f"{ocr_backend}_{pdf_text_mode}_page_{page_idx + 1:04d}.json",
            dpi=dpi,
            pdf_doc=pdf_doc,
            overwrite=overwrite,
            ocr_backend=ocr_backend,
            pdf_text_mode=pdf_text_mode,
        )
        if not doc_raw_dob_captured:
            captured = _capture_raw_dob_from_page_text(page.text)
            if captured:
                doc_raw_dob_captured = captured
                if not doc_dob_value:
                    ts_cap = _parse_date_for_age_days(captured)
                    if ts_cap is not None:
                        doc_dob_value = ts_cap.strftime("%Y-%m-%d")

        pii_spans = detector.detect(page.text, device=detector_device)
        pii_spans = [span for span in pii_spans if span.tag in allowed_tags]
        # Deterministic header fallback for "Name: <value>" misses.
        label_name_spans = _detect_name_label_spans(page.text)
        demo_summary_name_spans = _detect_demographic_summary_name_spans(page.text)
        footer_name_spans = _detect_footer_name_spans(page.text)
        if label_name_spans or demo_summary_name_spans or footer_name_spans:
            pii_spans = merge_overlapping_spans(
                [*pii_spans, *label_name_spans, *demo_summary_name_spans, *footer_name_spans]
            )
        for span in pii_spans:
            if not span.text:
                span.text = page.text[span.start:span.end]
            if not span.source:
                span.source = detector.detector_name
            if not span.reason:
                span.reason = f"{span.tag}_detected"
        # Remove label-only false positives from persisted detections.
        pii_spans = [
            span
            for span in pii_spans
            if not (
                span.tag == "patient_name"
                and _is_patient_label_text(span.text or page.text[span.start:span.end])
            )
        ]
        pii_spans = expand_patient_name_token_spans(page.text, pii_spans)
        # Final pass to catch patient-name leakage after initial detection.
        leak_name_spans = _sweep_patient_name_leak_spans(page.text, pii_spans)
        if leak_name_spans:
            pii_spans = merge_overlapping_spans([*pii_spans, *leak_name_spans])
        # Cross-page safety pass:
        # Once key identifiers are found, force-check/redact them on later pages.
        forced_page_spans = _forced_literal_spans_for_page(page.text, forced_literals)
        if forced_page_spans:
            pii_spans = merge_overlapping_spans([*pii_spans, *forced_page_spans])

        stress_layout = _is_probable_stress_test_layout(page.text)
        redact_full_dates_mode = bool(eu_mode) or bool(full_date_overlay_mode)
        effective_spans = build_effective_redaction_spans(
            page.text,
            pii_spans,
            prefer_day_first_dates=stress_layout,
            redact_full_dates=redact_full_dates_mode,
        )
        # Update forced literals from effective spans (labels already filtered).
        _merge_forced_literals(forced_literals, _collect_forced_literals_from_spans(effective_spans))
        if not doc_dob_value:
            dob_candidates: List[str] = []
            for span in effective_spans:
                if span.tag != "dob":
                    continue
                val = (span.text or page.text[span.start:span.end] or "").strip()
                if val:
                    dob_candidates.append(val)
            for val in sorted(list(forced_literals.get("dob", set())), key=len, reverse=True):
                if val:
                    dob_candidates.append(str(val))
            for cand in dob_candidates:
                ts = _parse_date_for_age_days(cand)
                if ts is not None:
                    if not doc_raw_dob_captured:
                        doc_raw_dob_captured = str(cand or "").strip()
                    doc_dob_value = ts.strftime("%Y-%m-%d")
                    break
        if not doc_dob_value:
            # Fallback for layouts where DOB is detected as a generic date token:
            # infer from page text labels (e.g., "Date of birth: ...") so EU
            # age overlays can still be rendered.
            inferred_from_page = infer_raw_dates_from_pages(
                [
                    {
                        "source_text_for_dates": page.text,
                    }
                ]
            )
            inferred_dob = str(
                inferred_from_page.get("raw_dob", "")
                or inferred_from_page.get("dob", "")
                or ""
            ).strip()
            if inferred_dob:
                if not doc_raw_dob_captured:
                    doc_raw_dob_captured = inferred_dob
                doc_dob_value = inferred_dob
        fallback_age_days: Optional[int] = None
        dob_ts = _parse_date_for_age_days(doc_dob_value)
        if dob_ts is not None:
            parsed_days: List[int] = []
            for span in effective_spans:
                if span.tag not in {"date", "date_history"}:
                    continue
                raw_text = (span.text or page.text[span.start:span.end] or "").strip()
                dts = _parse_date_for_age_days(raw_text)
                if dts is None:
                    continue
                delta = int((dts - dob_ts).days)
                if delta >= 0:
                    parsed_days.append(delta)
            if parsed_days:
                fallback_age_days = max(parsed_days)
        redaction_boxes = pii_spans_to_redaction_boxes(page, effective_spans)
        try:
            redaction_boxes = refine_date_day_boxes_with_image(image, redaction_boxes)
        except Exception:
            # Keep pipeline resilient: if pass-2 refinement fails on a page,
            # fall back to pass-1 boxes instead of aborting the document.
            logger.exception("Pass-2 date refinement failed on %s page %s", file_path.name, page_idx + 1)
        try:
            redaction_boxes = refine_date_day_boxes_pass3(image, page, redaction_boxes)
        except Exception:
            # Keep pipeline resilient if pass-3 line/ink snapping fails.
            logger.exception("Pass-3 date refinement failed on %s page %s", file_path.name, page_idx + 1)
        signature_footer_like = _has_bottom_signature_line(page)
        # Legacy footer fallback: catches burned-in name/MRN stamp on bottom-left.
        footer_boxes: List[RedactionBox] = []
        if not signature_footer_like:
            footer_boxes = detect_footer_stamp_proxy_boxes(page)
            if not footer_boxes:
                footer_boxes = detect_footer_boxes_from_patient_tokens(page, pii_spans)
        redaction_boxes.extend(footer_boxes)
        # If current page misses footer but previous pages had reliable footer box, propagate template.
        if footer_boxes:
            best_footer = footer_boxes[0]
            fx1, fy1, fx2, fy2 = best_footer.bbox_xyxy
            pw = max(1.0, float(page.width))
            ph = max(1.0, float(page.height))
            footer_template_rel = (fx1 / pw, fy1 / ph, fx2 / pw, fy2 / ph)
        elif footer_template_rel and not signature_footer_like:
            rx1, ry1, rx2, ry2 = footer_template_rel
            pw = max(1.0, float(page.width))
            ph = max(1.0, float(page.height))
            redaction_boxes.append(
                RedactionBox(
                    page_index=page.page_index,
                    tag="footer_phi",
                    text="footer_template_propagated",
                    bbox_xyxy=(rx1 * pw, ry1 * ph, rx2 * pw, ry2 * ph),
                )
            )
        # Table fallback: capture value cell for "Patient Name" rows (not label text).
        redaction_boxes.extend(detect_patient_name_table_boxes(page))
        # Label/value fallback for older forms with explicit MRN/date labels.
        labeled_mrn_boxes = detect_mrn_label_value_boxes(page)
        labeled_date_boxes = detect_labeled_date_day_boxes(
            page,
            prefer_day_first=stress_layout,
        )
        redaction_boxes = _prefer_label_geometry_boxes(redaction_boxes, labeled_mrn_boxes)
        redaction_boxes = _prefer_label_geometry_boxes(redaction_boxes, labeled_date_boxes)
        redaction_boxes = _force_label_boxes_on_labeled_lines(page, redaction_boxes, labeled_mrn_boxes, "mrn")
        redaction_boxes = _force_label_boxes_on_labeled_lines(page, redaction_boxes, labeled_date_boxes, "date_day")
        if signature_footer_like:
            # Final guardrail: on signature lines only keep day/time redactions.
            filtered_boxes: List[RedactionBox] = []
            for box in redaction_boxes:
                if box.tag in {"date_day", "time"}:
                    filtered_boxes.append(box)
                    continue
                if _box_on_signature_line(page, box):
                    continue
                filtered_boxes.append(box)
            redaction_boxes = filtered_boxes
        overlay_items = [
            {
                "tag": str(box.tag or ""),
                "text": str(box.text or ""),
                "bbox_xyxy": [int(round(box.bbox_xyxy[0])), int(round(box.bbox_xyxy[1])), int(round(box.bbox_xyxy[2])), int(round(box.bbox_xyxy[3]))],
            }
            for box in redaction_boxes
        ]
        if bool(eu_mode):
            page_text_overlays = _build_age_days_overlays_from_items(
                overlay_items,
                doc_dob_value,
                fallback_age_days=fallback_age_days,
                exclude_bboxes=_dob_label_exclusion_regions(page),
            )
        elif bool(full_date_overlay_mode):
            page_text_overlays = date_spans_to_text_overlays(page, effective_spans)
        else:
            page_text_overlays = []
        redacted_img = apply_redaction_boxes(image, redaction_boxes, text_overlays=page_text_overlays)
        redacted_pages.append(redacted_img)

        total_spans += len(pii_spans)
        total_boxes += len(redaction_boxes)

        review_png = review_pages_dir / f"{doc_name}_page_{page_idx + 1:04d}.png"
        redacted_img.save(review_png)

        page_summaries.append(
            {
                "page_index": page_idx,
                "ocr_text_len": len(page.text),
                "extraction_method": page.extraction_method,
                # Preferred base text for downstream sanitized OCR export.
                "source_text_for_export": page.text,
                # Keep original page text for metadata extraction (raw DOB/study date).
                "source_text_for_dates": page.text,
                "normalized_text_for_export": normalize_dates_to_first_of_month(page.text),
                "pii_spans": [asdict(span) for span in pii_spans],
                "effective_redaction_spans": [asdict(span) for span in effective_spans],
                "redaction_boxes": [
                    {
                        "tag": box.tag,
                        "text": box.text,
                        "bbox_xyxy": list(box.bbox_xyxy),
                    }
                    for box in redaction_boxes
                ],
                "text_overlays": page_text_overlays,
                "review_png_path": str(review_png),
            }
        )

    if pdf_doc is not None:
        pdf_doc.close()

    ext = file_path.suffix.lower()
    if ext == ".pdf":
        redacted_doc_path = doc_dir / f"{doc_name}_redacted.pdf"
        save_pdf_from_images(redacted_pages, redacted_doc_path)
    else:
        redacted_doc_path = doc_dir / f"{doc_name}_redacted{ext}"
        redacted_pages[0].save(redacted_doc_path)

    duration_sec = time.perf_counter() - start_time
    manifest_json = doc_dir / f"{doc_name}_manifest.json"
    manifest_json.write_text(
        json.dumps(
            {
                "source_file": str(file_path),
                "redacted_file": str(redacted_doc_path),
                "detector_backend": detector.detector_name,
                "ocr_backend": ocr_backend,
                "pdf_text_mode": pdf_text_mode,
                "pages": page_summaries,
                "total_pages": len(page_summaries),
                "total_pii_spans": total_spans,
                "total_redaction_boxes": total_boxes,
                "original_total_redaction_boxes": total_boxes,
                "duration_sec": duration_sec,
                # Captured from pre-redaction text during processing; used as a
                # robust DOB anchor when later text-order inference is noisy.
                "captured_raw_dob_pre_redaction": str(doc_raw_dob_captured or ""),
                "captured_dob_pre_redaction": str(_normalize_optional_date(doc_raw_dob_captured) or ""),
                "manual_redaction": {
                    "enabled": False,
                    "phrases": [],
                    "total_manual_boxes": 0,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    inferred_dates = infer_raw_dates_from_pages(page_summaries)
    if not doc_raw_dob_captured:
        doc_raw_dob_captured = str(inferred_dates.get("raw_dob", "") or "").strip()
    raw_dob_out = str(doc_raw_dob_captured or inferred_dates.get("raw_dob", "") or "").strip()
    dob_out = str(_normalize_optional_date(raw_dob_out) or inferred_dates.get("dob", "") or "").strip()
    input_kind = classify_input_kind(file_path)

    return {
        "site_id": "",
        "source_file": str(file_path),
        "source_filename": file_path.name,
        "source_ext": ext,
        "input_kind": input_kind,
        "doc_id": doc_name,
        "detector_backend": detector.detector_name,
        "ocr_backend": ocr_backend,
        "pdf_text_mode": pdf_text_mode,
        "redacted_file": str(redacted_doc_path),
        "review_pages_dir": str(review_pages_dir),
        "manifest_json": str(manifest_json),
        "total_pages": len(page_summaries),
        "total_pii_spans": total_spans,
        "total_redaction_boxes": total_boxes,
        "duration_sec": round(duration_sec, 3),
        "phi_found": int(total_spans > 0),
        "review_status": "pending",
        "approved_to_send": np.nan,
        "reviewer": "",
        "review_notes": "",
        "force_id": "",
        "file_id": "",
        "modality_instance": 1,
        "first_name": "",
        "last_name": "",
        "mrn": "",
        "raw_dob": raw_dob_out,
        "dob": dob_out,
        "gender": "",
        "patient_id": "",
        "modality_type": "",
        "raw_study_date": inferred_dates.get("raw_study_date", ""),
        "study_date": inferred_dates.get("study_date", ""),
        "eu_mode": int(bool(eu_mode)),
        "full_date_overlay_mode": int(bool(full_date_overlay_mode)),
    }


def process_reports_local(
    input_dir: str,
    output_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    device: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
    overwrite: bool = True,
    detector_backend: str = "hybrid",
    ocr_backend: str = "paddle",
    pdf_text_mode: str = "hybrid_pdf_text",
    recursive_input_scan: bool = False,
    ai_model_path: Optional[str] = None,
    ai_prompt_template: Optional[str] = None,
    source_files: Optional[Sequence[str]] = None,
    append_to_tracker: bool = False,
    tracker_csv_path: Optional[str] = None,
    skip_files_in_tracker: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    eu_mode: bool = False,
    full_date_overlay_mode: bool = False,
) -> pd.DataFrame:
    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    output_path = Path(output_dir).expanduser().resolve() if output_dir else input_path / "redaction_output"
    cache_path = Path(cache_dir).expanduser().resolve() if cache_dir else output_path / ".cache"

    ensure_dir(output_path)
    ensure_dir(cache_path)
    tracker_csv = (
        Path(str(tracker_csv_path)).expanduser().resolve()
        if tracker_csv_path and str(tracker_csv_path).strip()
        else output_path / "redaction_tracker.csv"
    )

    if source_files:
        files: List[Path] = []
        for raw in source_files:
            path = Path(str(raw)).expanduser().resolve()
            if not path.exists() or not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_DOC_EXTS:
                continue
            files.append(path)
    else:
        files = list_input_files(input_path, recursive=recursive_input_scan)

    if skip_files_in_tracker and tracker_csv.exists():
        try:
            tracked_df = load_tracker(str(tracker_csv), create_if_missing=False)
            tracked_paths: set[str] = set()
            if "source_file" in tracked_df.columns:
                for raw in tracked_df["source_file"].fillna("").astype(str).tolist():
                    value = raw.strip()
                    if not value:
                        continue
                    try:
                        tracked_path = Path(value).expanduser().resolve()
                    except Exception:
                        continue
                    tracked_paths.add(os.path.normcase(os.path.normpath(str(tracked_path))))

            if tracked_paths:
                before_count = len(files)
                files = [
                    file_path
                    for file_path in files
                    if os.path.normcase(os.path.normpath(str(file_path.resolve()))) not in tracked_paths
                ]
                skipped_count = before_count - len(files)
                if skipped_count > 0:
                    logger.info(
                        "Skipping %s file(s) already present in tracker: %s",
                        skipped_count,
                        tracker_csv,
                    )
        except Exception as exc:
            logger.warning("Could not apply tracker-based skipping from %s: %s", tracker_csv, exc)

    if not files:
        if skip_files_in_tracker and tracker_csv.exists():
            logger.info("All discovered files are already present in tracker; nothing to process.")
            return load_tracker(str(tracker_csv), create_if_missing=True)
        raise ValueError(f"No supported files found under: {input_path}")

    detector = build_detector(
        detector_backend,
        ai_model_path=ai_model_path,
        ai_prompt_template=ai_prompt_template,
    )
    ocr_backend = _normalize_ocr_backend(ocr_backend)
    pdf_text_mode = str(pdf_text_mode or "hybrid_pdf_text").strip().lower()
    if pdf_text_mode not in {"hybrid_pdf_text", "ocr_only"}:
        raise ValueError("pdf_text_mode must be one of: hybrid_pdf_text, ocr_only")
    rows: List[Dict[str, Any]] = []
    logger.info("Found %s supported file(s)", len(files))
    total_files = len(files)
    for idx, file_path in enumerate(files, start=1):
        try:
            rows.append(
                process_single_document(
                    file_path=file_path,
                    output_root=output_path,
                    cache_root=cache_path,
                    detector=detector,
                    detector_device=device,
                    dpi=dpi,
                    overwrite=overwrite,
                    ocr_backend=ocr_backend,
                    pdf_text_mode=pdf_text_mode,
                    eu_mode=bool(eu_mode),
                    full_date_overlay_mode=bool(full_date_overlay_mode),
                )
            )
        except Exception as exc:
            logger.exception("Failed processing %s", file_path)
            rows.append(
                {
                    "site_id": "",
                    "source_file": str(file_path),
                    "source_filename": file_path.name,
                    "source_ext": file_path.suffix.lower(),
                    "input_kind": classify_input_kind(file_path),
                    "doc_id": safe_stem(file_path),
                    "detector_backend": detector_backend,
                    "ocr_backend": ocr_backend,
                    "pdf_text_mode": pdf_text_mode,
                    "redacted_file": "",
                    "review_pages_dir": "",
                    "manifest_json": "",
                    "total_pages": np.nan,
                    "total_pii_spans": np.nan,
                    "total_redaction_boxes": np.nan,
                    "duration_sec": np.nan,
                    "phi_found": np.nan,
                    "review_status": "error",
                    "approved_to_send": np.nan,
                    "reviewer": "",
                    "review_notes": str(exc),
                    "force_id": "",
                    "file_id": "",
                    "modality_instance": 1,
                    "first_name": "",
                    "last_name": "",
                    "mrn": "",
                    "raw_dob": "",
                    "dob": "",
                    "gender": "",
                    "patient_id": "",
                    "modality_type": "",
                    "raw_study_date": "",
                    "study_date": "",
                    "full_date_overlay_mode": int(bool(full_date_overlay_mode)),
                }
            )
        finally:
            if progress_callback is not None:
                try:
                    progress_callback(idx, total_files, file_path.name)
                except Exception:
                    pass

    ensure_dir(tracker_csv.parent)
    tracker_df = pd.DataFrame(rows)
    if append_to_tracker and tracker_csv.exists():
        existing = load_tracker(str(tracker_csv), create_if_missing=True)
        merged = pd.concat([existing, tracker_df], ignore_index=True)
        if "doc_id" in merged.columns:
            merged = merged.drop_duplicates(subset=["doc_id"], keep="last")
        tracker_df = merged
    save_tracker(tracker_df, str(tracker_csv))
    tracker_df = load_tracker(str(tracker_csv), create_if_missing=True)
    logger.info("Tracker written to %s", tracker_csv)
    return tracker_df


def _empty_tracker_df() -> pd.DataFrame:
    return pd.DataFrame(columns=TRACKER_COLUMNS)


def _ensure_tracker_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in TRACKER_COLUMNS:
        if col not in out.columns:
            if col in {
                "approved_to_send",
                "sent_to_aws",
                "phi_found",
                "total_pages",
                "total_pii_spans",
                "auto_redaction_elements",
                "total_redaction_boxes",
                "user_deleted_elements",
                "user_added_elements",
                "modality_instance",
                "duration_sec",
                "dup",
                "eu_mode",
                "full_date_overlay_mode",
            }:
                out[col] = np.nan
            else:
                out[col] = ""
    # Legacy mapping: if force_id missing but patient_id exists, hydrate force_id.
    if "force_id" in out.columns and "patient_id" in out.columns:
        force_blank = out["force_id"].fillna("").astype(str).str.strip() == ""
        out.loc[force_blank, "force_id"] = out.loc[force_blank, "patient_id"].fillna("").astype(str)
    # Keep only known columns in stable order.
    # Backfill new redaction metrics when loading older trackers.
    if "auto_redaction_elements" in out.columns and "total_redaction_boxes" in out.columns:
        mask_blank = pd.to_numeric(out["auto_redaction_elements"], errors="coerce").isna()
        out.loc[mask_blank, "auto_redaction_elements"] = out.loc[mask_blank, "total_redaction_boxes"]
    if "user_deleted_elements" in out.columns:
        out["user_deleted_elements"] = pd.to_numeric(out["user_deleted_elements"], errors="coerce").fillna(0).astype(int)
    if "user_added_elements" in out.columns:
        out["user_added_elements"] = pd.to_numeric(out["user_added_elements"], errors="coerce").fillna(0).astype(int)
    if "modality_instance" in out.columns:
        out["modality_instance"] = pd.to_numeric(out["modality_instance"], errors="coerce").fillna(1).astype(int)
    if "dup" in out.columns:
        out["dup"] = pd.to_numeric(out["dup"], errors="coerce").fillna(0).astype(int)
    if "eu_mode" in out.columns:
        out["eu_mode"] = pd.to_numeric(out["eu_mode"], errors="coerce").fillna(0).astype(int)
    if "full_date_overlay_mode" in out.columns:
        out["full_date_overlay_mode"] = pd.to_numeric(out["full_date_overlay_mode"], errors="coerce").fillna(0).astype(int)
    # Backfill age_at_event for any existing rows that have DOB + study date.
    if {"age_at_event", "dob", "study_date"}.issubset(out.columns):
        age_blank = out["age_at_event"].fillna("").astype(str).str.strip().eq("")
        if bool(age_blank.any()):
            calc_cols = [c for c in ["raw_dob", "raw_study_date", "dob", "study_date"] if c in out.columns]
            out.loc[age_blank, "age_at_event"] = out.loc[age_blank, calc_cols].apply(
                lambda r: _compute_age_at_event(
                    str(r.get("raw_dob", "") or r.get("dob", "") or ""),
                    str(r.get("raw_study_date", "") or r.get("study_date", "") or ""),
                ),
                axis=1,
            )
        out["age_at_event"] = out["age_at_event"].fillna("").astype(str).apply(_normalize_age_at_event)
    out = out[[c for c in TRACKER_COLUMNS if c in out.columns]]
    return out


def load_tracker(tracker_csv: str, create_if_missing: bool = False) -> pd.DataFrame:
    raw = str(tracker_csv or "").strip()
    if raw == "":
        raise ValueError("Tracker path is empty. Provide a CSV path or folder path.")

    tracker_path = Path(raw).expanduser().resolve()
    if tracker_path.is_dir():
        tracker_path = tracker_path / "redaction_tracker.csv"

    if not tracker_path.exists():
        if not create_if_missing:
            raise FileNotFoundError(f"Tracker not found: {tracker_path}")
        ensure_dir(tracker_path.parent)
        empty_df = _empty_tracker_df()
        empty_df.to_csv(tracker_path, index=False)
        return empty_df

    df = pd.read_csv(tracker_path)
    df = _ensure_tracker_schema(df)
    # Persist schema upgrades in place.
    save_tracker(df, str(tracker_path))
    return df


def save_tracker(df: pd.DataFrame, tracker_csv: str) -> None:
    tracker_path = Path(tracker_csv).expanduser().resolve()
    normalized = _ensure_tracker_schema(df)
    # Duplicate marker (DUP=1): same MRN + same de-identified study date.
    if {"mrn", "study_date", "dup"}.issubset(normalized.columns):
        work = normalized.copy()
        work["_k_mrn"] = work["mrn"].fillna("").astype(str).str.strip().str.lower()
        work["_k_date"] = work["study_date"].fillna("").astype(str).str.strip()
        work["dup"] = 0
        valid = work["_k_mrn"].ne("") & work["_k_date"].ne("")
        if bool(valid.any()):
            grouped = work[valid].groupby(["_k_mrn", "_k_date"], dropna=False)
            for _, idxs in grouped.groups.items():
                idx_list = list(idxs)
                if len(idx_list) <= 1:
                    continue
                work.loc[idx_list, "dup"] = 1
            normalized["dup"] = pd.to_numeric(work["dup"], errors="coerce").fillna(0).astype(int)
    # Stable display order for users.
    sort_cols = [c for c in ["last_name", "first_name", "study_date", "doc_id"] if c in normalized.columns]
    if sort_cols:
        normalized = normalized.sort_values(by=sort_cols, kind="stable", na_position="last")
    # Write CSV directly to avoid Windows newline double-translation (\r\r\n),
    # which can appear as blank rows between records in spreadsheet viewers.
    # Pandas arg name differs by version: "lineterminator" vs "line_terminator".
    try:
        normalized.to_csv(tracker_path, index=False, encoding="utf-8", lineterminator="\n")
    except TypeError:
        try:
            normalized.to_csv(tracker_path, index=False, encoding="utf-8", line_terminator="\n")
        except TypeError:
            normalized.to_csv(tracker_path, index=False, encoding="utf-8")


def _normalize_study_date(value: str) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ValueError("study_date must be a valid date")
    ts = ts.replace(day=1)
    return ts.strftime("%Y-%m-%d")


def validate_force_id(force_id: str) -> str:
    normalized = (force_id or "").strip().upper()
    pattern = re.compile(r"^[A-Z]{3}-[A-Z0-9]{6}-\d+$")
    if not pattern.match(normalized):
        raise ValueError(
            "force_id must match XXX-LLLLLL-i or XXX-00000i-i (example: ABC-SMIJOH-1)"
        )
    return normalized


def validate_file_id(file_id: str) -> str:
    normalized = (file_id or "").strip().upper()
    pattern = re.compile(r"^[A-Z]{3}-[A-Z0-9]{6}-\d+_\d{8}_\d+$")
    if not pattern.match(normalized):
        raise ValueError(
            "file_id must match XXX-LLLLLL-i_YYYYMMDD-j (example: ABC-SMIJOH-1_20260501_1)"
        )
    return normalized


def normalize_modality_instance(value: object) -> int:
    try:
        inst = int(float(value))
    except Exception as exc:
        raise ValueError("modality_instance must be an integer >= 1") from exc
    if inst < 1:
        raise ValueError("modality_instance must be >= 1")
    return inst


def _normalize_optional_date(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    ts = pd.to_datetime(raw, errors="coerce")
    if pd.isna(ts):
        return ""
    ts = ts.replace(day=1)
    return ts.strftime("%Y-%m-%d")


def infer_raw_dates_from_pages(pages: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    raw_dob = ""
    raw_study_date = ""
    dob_candidates: List[str] = []
    generic_date_candidates: List[str] = []
    study_label_date_candidates: List[str] = []

    label_pat = re.compile(
        r"\b(?:date\s*of\s*test|test\s*date|study\s*date|date\s*of\s*study|exam\s*date|service\s*date|performed\s*on)\b",
        flags=re.IGNORECASE,
    )
    date_pat = re.compile(
        r"\b(?:"
        r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|"
        r"\d{4}[./-]\d{1,2}[./-]\d{1,2}|"
        r"\d{8}|"
        r"\d{1,2}-[A-Za-z]{3,9}-\d{2,4}|"
        r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}|"
        r"[A-Za-z]{3,9}\s+\d{4}|"
        r"[A-Za-z]{3,9}\s*[-/]\s*\d{4}|"
        r"[-xX#]{1,4}\s*[-/]\s*[A-Za-z]{3,9}\s*[-/]\s*\d{4}"
        r")\b"
    )
    dob_label_pat = re.compile(
        r"\b(?:d\.?\s*o\.?\s*b\.?|date\s*of\s*birth|birth\s*date|born)\b",
        flags=re.IGNORECASE,
    )
    dob_pat = re.compile(
        r"\b(?:d\.?\s*o\.?\s*b\.?|date\s*of\s*birth|birth\s*date|born)\b\s*[:\-]?\s*("
        r"[0-9]{1,8}[./-][0-9]{1,2}[./-][0-9]{1,8}|"
        r"[0-9]{8}|"
        r"[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}|"
        r"[A-Za-z]{3,9}\.?\s+\d{4}|"
        r"[A-Za-z]{3,9}\.?\s*[-/]\s*\d{4}|"
        r"[-xX#]{1,4}\s*[-/]\s*[A-Za-z]{3,9}\.?\s*[-/]\s*\d{4})",
        flags=re.IGNORECASE,
    )
    signed_label_pat = re.compile(
        r"\b(?:signed\s*on|study\s*time|performed\s*on)\b",
        flags=re.IGNORECASE,
    )

    def _expand_two_digit_year(year_2: int) -> int:
        return 2000 + year_2 if year_2 <= 30 else 1900 + year_2

    def _coerce_raw_date(value: str) -> tuple[str, str]:
        v = str(value or "").strip()
        if not v:
            return "", ""
        # Common OCR substitutions in numeric fragments.
        v = re.sub(r"(?<=\d)[Oo](?=\d)", "0", v)
        v = re.sub(r"(?<=\d)[Il](?=\d)", "1", v)
        # Keep only the most date-like fragment.
        patt = re.compile(
            r"("
            r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|"
            r"\d{4}[./-]\d{1,2}[./-]\d{1,2}|"
            r"\d{8}|"
            r"\d{1,2}[/-]\d{1,2}\d{2}|"
            r"\d{1,2}[/-]\d{1}\d{3}|"
            r"[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{2,4}|"
            r"\d{1,2}[-\s][A-Za-z]{3,9}[-\s]\d{2,4}|"
            r"[A-Za-z]{3,9}\.?\s+\d{4}|"
            r"[A-Za-z]{3,9}\.?\s*[-/]\s*\d{4}|"
            r"[-xX#]{1,4}\s*[-/]\s*[A-Za-z]{3,9}\.?\s*[-/]\s*\d{4}|"
            r"\d{1,2}[/-]\d{4}"
            r")",
            flags=re.IGNORECASE,
        )
        m = patt.search(v)
        if not m:
            return "", ""
        frag = m.group(1).strip().replace(".", "")
        if frag.isdigit() and len(frag) == 8:
            # Try YYYYMMDD first.
            y1, mo1, d1 = int(frag[:4]), int(frag[4:6]), int(frag[6:8])
            if 1900 <= y1 <= 2100 and 1 <= mo1 <= 12 and 1 <= d1 <= 31:
                raw = f"{mo1}/{d1}/{y1}"
                norm = _normalize_optional_date(raw)
                return (raw, norm) if norm else ("", "")
            # Try MMDDYYYY next.
            mo2, d2, y2 = int(frag[:2]), int(frag[2:4]), int(frag[4:8])
            if 1900 <= y2 <= 2100 and 1 <= mo2 <= 12 and 1 <= d2 <= 31:
                raw = f"{mo2}/{d2}/{y2}"
                norm = _normalize_optional_date(raw)
                return (raw, norm) if norm else ("", "")
        # Handle OCR-collapsed numeric dates like "8/1196" => "8/1/1996".
        mm = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})(\d{2})", frag)
        if mm:
            month = int(mm.group(1))
            day = int(mm.group(2))
            year = _expand_two_digit_year(int(mm.group(3)))
            if 1 <= month <= 12 and 1 <= day <= 31:
                raw = f"{month}/{day}/{year}"
                norm = _normalize_optional_date(raw)
                return (raw, norm) if norm else ("", "")
        mm2 = re.fullmatch(r"(\d{1,2})[/-](\d)(\d{3})", frag)
        if mm2:
            month = int(mm2.group(1))
            day = int(mm2.group(2))
            year = _expand_two_digit_year(int(mm2.group(3)[-2:]))
            if 1 <= month <= 12 and 1 <= day <= 31:
                raw = f"{month}/{day}/{year}"
                norm = _normalize_optional_date(raw)
                return (raw, norm) if norm else ("", "")

        # Generic parse for standard numeric/month formats.
        ts = pd.to_datetime(frag, errors="coerce")
        if pd.isna(ts):
            return "", ""
        year = int(ts.year)
        if year < 1900 or year > 2100:
            return "", ""
        # Preserve day for raw, but normalize redacted date to first-of-month.
        raw = f"{int(ts.month)}/{int(ts.day)}/{year}"
        norm = _normalize_optional_date(raw)
        return (raw, norm) if norm else ("", "")

    dob_ranked_candidates: List[tuple[int, str]] = []
    study_ranked_candidates: List[tuple[int, str]] = []

    for page in pages:
        page_text = str(
            page.get("source_text_for_dates", "")
            or page.get("text", "")
            or page.get("normalized_text_for_export", "")
            or ""
        )
        if page_text:
            lines = page_text.splitlines()

            def _line_dates(line_value: str) -> List[str]:
                out_dates: List[str] = []
                for dm in date_pat.finditer(str(line_value or "")):
                    candidate_raw, candidate_norm = _coerce_raw_date(str(dm.group(0)).strip())
                    if candidate_raw and candidate_norm:
                        out_dates.append(candidate_raw)
                return out_dates

            for idx, line in enumerate(lines):
                line_match = label_pat.search(line)
                if not line_match:
                    pass
                else:
                    if not re.search(r"\b(?:dob|date\s*of\s*birth|born)\b", line, flags=re.IGNORECASE):
                        tail = line[line_match.end() :]
                        dm = date_pat.search(tail) or date_pat.search(line)
                        if dm:
                            candidate_raw, candidate_norm = _coerce_raw_date(str(dm.group(0)).strip())
                            if candidate_raw and candidate_norm:
                                study_label_date_candidates.append(candidate_raw)
                                study_ranked_candidates.append((100, candidate_raw))

                has_dob_label = bool(dob_label_pat.search(line))
                has_signed_label = bool(signed_label_pat.search(line))
                cur_dates = _line_dates(line)
                prev_line = lines[idx - 1] if idx > 0 else ""
                next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
                prev_dates = _line_dates(prev_line)
                next_dates = _line_dates(next_line)
                prev_has_signed = bool(signed_label_pat.search(prev_line))
                next_has_signed = bool(signed_label_pat.search(next_line))

                if has_dob_label:
                    # Same-line DOB label/value is highest confidence.
                    for cand in cur_dates:
                        if has_signed_label:
                            continue
                        dob_ranked_candidates.append((120, cand))
                    # Handle OCR row order flips: value on preceding/following line.
                    for cand in prev_dates:
                        if prev_has_signed:
                            continue
                        dob_ranked_candidates.append((110, cand))
                    for cand in next_dates:
                        if next_has_signed:
                            continue
                        dob_ranked_candidates.append((110, cand))

                # Reverse-order form: date line immediately followed by DOB label line.
                if cur_dates and bool(dob_label_pat.search(next_line)):
                    if not has_signed_label:
                        for cand in cur_dates:
                            dob_ranked_candidates.append((115, cand))

                # Capture study-date neighbors conservatively.
                if line_match:
                    for cand in cur_dates:
                        study_ranked_candidates.append((95, cand))
                    for cand in next_dates:
                        if not next_has_signed:
                            study_ranked_candidates.append((85, cand))

            if not raw_dob:
                dm = dob_pat.search(page_text)
                if dm:
                    candidate_raw, candidate_norm = _coerce_raw_date(str(dm.group(1)).strip())
                    if candidate_raw and candidate_norm:
                        raw_dob = candidate_raw
                        dob_ranked_candidates.append((105, candidate_raw))
            if not raw_dob:
                # Fallback: any line with DOB label plus a date-like token.
                for line in lines:
                    if not dob_label_pat.search(line):
                        continue
                    dm = date_pat.search(line)
                    if not dm:
                        continue
                    candidate_raw, candidate_norm = _coerce_raw_date(str(dm.group(0)).strip())
                    if candidate_raw and candidate_norm:
                        raw_dob = candidate_raw
                        dob_ranked_candidates.append((100, candidate_raw))
                        break

        # For DOB candidates, trust only raw detector spans (not de-id effective spans).
        for span in list(page.get("pii_spans", []) or []):
            tag = str(span.get("tag", "") or "").strip().lower()
            value = str(span.get("text", "") or "").strip()
            if not value:
                continue
            candidate_raw, candidate_norm = _coerce_raw_date(value)
            if not candidate_raw or not candidate_norm:
                continue
            if tag == "dob":
                dob_candidates.append(candidate_raw)
        for span in list(page.get("effective_redaction_spans", []) or []) + list(page.get("pii_spans", []) or []):
            tag = str(span.get("tag", "") or "").strip().lower()
            value = str(span.get("text", "") or "").strip()
            if not value:
                continue
            candidate_raw, candidate_norm = _coerce_raw_date(value)
            if not candidate_raw or not candidate_norm:
                continue
            if tag in {"date", "date_history"}:
                generic_date_candidates.append(candidate_raw)
                study_ranked_candidates.append((60, candidate_raw))

    if not raw_dob and dob_ranked_candidates:
        raw_dob = sorted(dob_ranked_candidates, key=lambda t: t[0], reverse=True)[0][1]
    if not raw_dob and dob_candidates:
        raw_dob = dob_candidates[0]

    if study_ranked_candidates:
        raw_study_date = sorted(study_ranked_candidates, key=lambda t: t[0], reverse=True)[0][1]
    elif study_label_date_candidates:
        raw_study_date = study_label_date_candidates[0]
    elif generic_date_candidates:
        raw_study_date = generic_date_candidates[0]
    elif dob_candidates:
        raw_study_date = dob_candidates[0]

    def _to_dt(raw_value: str) -> Optional[pd.Timestamp]:
        norm = _normalize_optional_date(raw_value)
        if not norm:
            return None
        ts = pd.to_datetime(norm, errors="coerce")
        if pd.isna(ts):
            return None
        return ts

    dob_dt = _to_dt(raw_dob)
    study_dt = _to_dt(raw_study_date)
    if dob_dt is not None and study_dt is not None and dob_dt >= study_dt:
        # Enforce DOB before study date; fall back to next strongest valid DOB.
        best_valid_dob = ""
        for _score, cand in sorted(dob_ranked_candidates, key=lambda t: t[0], reverse=True):
            cand_dt = _to_dt(cand)
            if cand_dt is not None and cand_dt < study_dt:
                best_valid_dob = cand
                break
        if best_valid_dob:
            raw_dob = best_valid_dob
        else:
            raw_dob = ""

    dob = _normalize_optional_date(raw_dob)
    study_date = _normalize_optional_date(raw_study_date)
    return {
        "raw_dob": raw_dob,
        "dob": dob,
        "raw_study_date": raw_study_date,
        "study_date": study_date,
    }


def infer_raw_dates_from_manifest(manifest_path: Path) -> Dict[str, str]:
    if not manifest_path.exists():
        return {
            "raw_dob": "",
            "dob": "",
            "raw_study_date": "",
            "study_date": "",
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "raw_dob": "",
            "dob": "",
            "raw_study_date": "",
            "study_date": "",
        }
    pages = list(manifest.get("pages", []) or [])
    return infer_raw_dates_from_pages(pages)


def _compute_age_at_event(dob_value: str, study_date_value: str) -> str:
    tdob = _parse_date_for_age_days(str(dob_value or ""))
    tstudy = _parse_date_for_age_days(str(study_date_value or ""))
    if tdob is None or tstudy is None:
        return ""
    delta_days = int(max(0, int((tstudy - tdob).days)))
    return str(delta_days)


def _normalize_age_at_event(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    txt = raw.lower().strip()

    # Explicit day-unit inputs.
    m_day = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(?:d|day|days)\s*$", txt)
    if m_day:
        try:
            return str(max(0, int(round(float(m_day.group(1))))))
        except Exception:
            return ""

    # Explicit year-unit inputs (legacy/manual); convert to days.
    m_year = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(?:y|yr|yrs|year|years)\s*$", txt)
    if m_year:
        try:
            years = max(0.0, float(m_year.group(1)))
            return str(int(round(years * 365.25)))
        except Exception:
            return ""

    # Bare numeric:
    # - decimals in plausible human-age range are treated as legacy years
    # - all other numerics are treated as days
    try:
        num = max(0.0, float(txt))
        if "." in txt:
            if num <= 130.0:
                return str(int(round(num * 365.25)))
            return str(int(round(num)))
        return str(max(0, int(round(num))))
    except Exception:
        return ""


def _queue_rows_for_resend(df: pd.DataFrame, mask: pd.Series) -> None:
    """
    Mark rows as needing resend to AWS after metadata/content changes.
    """
    if "sent_to_aws" not in df.columns:
        df["sent_to_aws"] = 0
    if "sent_to_aws_at_utc" not in df.columns:
        df["sent_to_aws_at_utc"] = ""
    df.loc[mask, "sent_to_aws"] = 0
    df.loc[mask, "sent_to_aws_at_utc"] = ""


def set_review_metadata(
    tracker_csv: str,
    doc_id: str,
    force_id: str,
    modality_type: str,
    study_date: str,
    modality_instance: int = 1,
    file_id: str = "",
    first_name: str = "",
    last_name: str = "",
    mrn: str = "",
    dob: str = "",
    gender: str = "",
    age_at_event: str = "",
    eu_mode: bool = False,
    site_id: str = "",
) -> pd.DataFrame:
    df = load_tracker(tracker_csv)
    mask = df["doc_id"] == doc_id
    if not mask.any():
        raise ValueError(f"doc_id not found in tracker: {doc_id}")

    normalized_force_id = validate_force_id(force_id)
    normalized_modality = (modality_type or "").strip()
    if not normalized_modality:
        raise ValueError("modality_type is required")

    normalized_study_date = _normalize_study_date(study_date)
    normalized_instance = normalize_modality_instance(modality_instance)
    # Auto-increment instance when another row already exists for same patient+month,
    # but from a different source file.
    current_src = ""
    try:
        current_src = str(df.loc[mask, "source_filename"].iloc[-1])
    except Exception:
        current_src = ""
    if "source_filename" in df.columns:
        other = df[~mask].copy()
        same_key = (
            other["force_id"].fillna("").astype(str).str.upper().eq(normalized_force_id)
            & other["study_date"].fillna("").astype(str).eq(normalized_study_date)
        )
        if bool(same_key.any()):
            if current_src:
                same_key = same_key & other["source_filename"].fillna("").astype(str).ne(current_src)
            prior_instances = pd.to_numeric(other.loc[same_key, "modality_instance"], errors="coerce").fillna(1).astype(int)
            if not prior_instances.empty:
                min_next = int(prior_instances.max()) + 1
                if normalized_instance < min_next:
                    normalized_instance = min_next
    default_file_id = f"{normalized_force_id}_{normalized_study_date.replace('-', '')}_{normalized_instance}"
    normalized_file_id = validate_file_id(file_id or default_file_id)
    input_raw_study_date = (study_date or "").strip()
    existing_raw_study_date = str(df.loc[mask, "raw_study_date"].fillna("").iloc[-1]).strip()
    existing_norm_study_date = _normalize_optional_date(existing_raw_study_date)
    input_raw_dob = (dob or "").strip()
    existing_raw_dob = str(df.loc[mask, "raw_dob"].fillna("").iloc[-1]).strip()
    existing_norm_dob = _normalize_optional_date(existing_raw_dob)
    normalized_dob = _normalize_optional_date(dob)
    # Preserve previously captured raw study-date text when caller passes normalized form.
    if not input_raw_study_date:
        raw_study_date = existing_raw_study_date
    elif existing_raw_study_date and existing_norm_study_date and existing_norm_study_date == normalized_study_date:
        raw_study_date = existing_raw_study_date
    else:
        raw_study_date = input_raw_study_date
    # Preserve previously captured raw DOB text when caller only passes normalized DOB.
    if not input_raw_dob:
        raw_dob = existing_raw_dob
    elif existing_raw_dob and existing_norm_dob and existing_norm_dob == normalized_dob:
        raw_dob = existing_raw_dob
    else:
        raw_dob = input_raw_dob
    normalized_age = _normalize_age_at_event(age_at_event)
    if not normalized_age:
        normalized_age = _compute_age_at_event(
            raw_dob or normalized_dob,
            raw_study_date or normalized_study_date,
        )

    df.loc[mask, "force_id"] = normalized_force_id
    df.loc[mask, "site_id"] = (site_id or "").strip()
    df.loc[mask, "file_id"] = normalized_file_id
    df.loc[mask, "modality_instance"] = int(normalized_instance)
    # keep legacy mirror for compatibility
    df.loc[mask, "patient_id"] = normalized_force_id
    df.loc[mask, "first_name"] = (first_name or "").strip()
    df.loc[mask, "last_name"] = (last_name or "").strip()
    df.loc[mask, "mrn"] = (mrn or "").strip()
    df.loc[mask, "raw_dob"] = raw_dob
    df.loc[mask, "dob"] = normalized_dob
    df.loc[mask, "age_at_event"] = normalized_age
    df.loc[mask, "gender"] = (gender or "").strip()
    df.loc[mask, "modality_type"] = normalized_modality
    df.loc[mask, "raw_study_date"] = raw_study_date
    df.loc[mask, "study_date"] = normalized_study_date
    df.loc[mask, "eu_mode"] = int(bool(eu_mode))
    _queue_rows_for_resend(df, mask)
    save_tracker(df, tracker_csv)
    return df


def set_review_decision(
    tracker_csv: str,
    doc_id: str,
    approved_to_send: bool,
    reviewer: str = "",
    review_notes: str = "",
    review_status: str = "reviewed",
    approved_session_id: str = "",
) -> pd.DataFrame:
    df = load_tracker(tracker_csv)
    mask = df["doc_id"] == doc_id
    if not mask.any():
        raise ValueError(f"doc_id not found in tracker: {doc_id}")

    df.loc[mask, "approved_to_send"] = int(bool(approved_to_send))
    if bool(approved_to_send):
        df.loc[mask, "approved_session_id"] = (approved_session_id or "").strip()
        df.loc[mask, "approved_at_utc"] = datetime.now(timezone.utc).isoformat()
        # Re-approval should queue document for a fresh send.
        df.loc[mask, "sent_to_aws"] = 0
        df.loc[mask, "sent_to_aws_at_utc"] = ""
    else:
        df.loc[mask, "approved_session_id"] = ""
        df.loc[mask, "approved_at_utc"] = ""
        df.loc[mask, "sent_to_aws"] = 0
        df.loc[mask, "sent_to_aws_at_utc"] = ""
    df.loc[mask, "review_status"] = review_status
    df.loc[mask, "reviewer"] = reviewer
    df.loc[mask, "review_notes"] = review_notes
    save_tracker(df, tracker_csv)
    return df


def mark_case_reviewed(
    tracker_csv: str,
    doc_id: str,
    reviewer: str = "",
    review_notes: str = "",
) -> pd.DataFrame:
    """
    Explicitly mark a document as reviewed.
    This is intentionally separate from manual edit statuses.
    """
    df = load_tracker(tracker_csv)
    mask = df["doc_id"] == doc_id
    if not mask.any():
        raise ValueError(f"doc_id not found in tracker: {doc_id}")

    df.loc[mask, "review_status"] = "reviewed"
    if reviewer:
        df.loc[mask, "reviewer"] = reviewer
    if review_notes:
        existing_notes = str(df.loc[mask, "review_notes"].fillna("").iloc[-1]).strip()
        df.loc[mask, "review_notes"] = f"{existing_notes} | {review_notes}".strip(" |")
    save_tracker(df, tracker_csv)
    return df


def _rebuild_redacted_document_from_review_pages(row: pd.Series) -> None:
    source_ext = str(row["source_ext"]).lower()
    review_dir = Path(row["review_pages_dir"])
    redacted_file = Path(row["redacted_file"])
    images = [Image.open(path).convert("RGB") for path in sorted(review_dir.glob("*.png"))]
    if not images:
        return
    if source_ext == ".pdf":
        save_pdf_from_images(images, redacted_file)
    else:
        images[0].save(redacted_file)


def get_page_redaction_boxes(tracker_csv: str, doc_id: str, page_number: int) -> List[Tuple[int, int, int, int]]:
    items = get_page_redaction_items(tracker_csv, doc_id, page_number)
    return [tuple(item["bbox_xyxy"]) for item in items]


def get_page_redaction_items(tracker_csv: str, doc_id: str, page_number: int) -> List[Dict[str, Any]]:
    if page_number < 1:
        raise ValueError("page_number must be >= 1")
    df = load_tracker(tracker_csv)
    mask = df["doc_id"] == doc_id
    if not mask.any():
        raise ValueError(f"doc_id not found in tracker: {doc_id}")
    row = df[mask].iloc[-1]
    manifest_path = Path(str(row["manifest_json"]))
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    page_idx = page_number - 1
    for page in manifest.get("pages", []):
        if int(page.get("page_index", -1)) != page_idx:
            continue
        out: List[Dict[str, Any]] = []
        for box in page.get("redaction_boxes", []):
            coords = box.get("bbox_xyxy", [])
            if not isinstance(coords, list) or len(coords) != 4:
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in coords]
            xa, xb = sorted([x1, x2])
            ya, yb = sorted([y1, y2])
            if xa == xb or ya == yb:
                continue
            out.append(
                {
                    "bbox_xyxy": [xa, ya, xb, yb],
                    "tag": str(box.get("tag", "") or ""),
                    "text": str(box.get("text", "") or ""),
                }
            )
        return out
    return []


def _load_source_page_image(
    source_file: Path,
    source_ext: str,
    page_number: int,
    target_size: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    if source_ext == ".pdf":
        with fitz.open(source_file) as doc:
            page_idx = page_number - 1
            if page_idx < 0 or page_idx >= len(doc):
                raise IndexError(f"page_number out of range for PDF: {page_number}")
            page = doc.load_page(page_idx)
            if target_size:
                page_w = max(1.0, float(page.rect.width))
                page_h = max(1.0, float(page.rect.height))
                sx = float(target_size[0]) / page_w
                sy = float(target_size[1]) / page_h
                pix = page.get_pixmap(matrix=fitz.Matrix(sx, sy), alpha=False)
            else:
                # Match pipeline default render quality when no explicit target exists.
                scale = float(DEFAULT_DPI) / 72.0
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    else:
        image = Image.open(source_file).convert("RGB")

    if target_size and image.size != target_size:
        resampling = getattr(Image, "Resampling", Image)
        image = image.resize(target_size, resampling.LANCZOS)
    return image


def override_page_redactions(
    tracker_csv: str,
    doc_id: str,
    page_number: int,
    boxes_xyxy: Sequence[Tuple[int, int, int, int]],
    redaction_items: Optional[Sequence[Dict[str, Any]]] = None,
    text_overlays: Optional[Sequence[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Rebuild a specific page from original source and apply exactly the provided boxes.
    This enables deleting/overriding AI/NLP redactions.
    """
    if page_number < 1:
        raise ValueError("page_number must be >= 1")

    df = load_tracker(tracker_csv)
    mask = df["doc_id"] == doc_id
    if not mask.any():
        raise ValueError(f"doc_id not found in tracker: {doc_id}")
    row = df[mask].iloc[-1]

    def _clean_path_value(value: Any) -> str:
        text = str(value or "").strip()
        return "" if text.lower() in {"", "nan", "none", "<na>", "nat"} else text

    manifest_raw = _clean_path_value(row.get("manifest_json", ""))
    source_raw = _clean_path_value(row.get("source_file", ""))
    review_raw = _clean_path_value(row.get("review_pages_dir", ""))
    redacted_raw = _clean_path_value(row.get("redacted_file", ""))
    manifest_path = Path(manifest_raw) if manifest_raw else Path("")
    source_file = Path(source_raw) if source_raw else Path("")
    source_ext = str(row["source_ext"]).lower()
    review_dir = Path(review_raw) if review_raw else Path("")

    # Fallbacks for older/broken tracker rows where review_pages_dir is missing/NaN.
    if not review_raw:
        if manifest_raw:
            review_dir = manifest_path.parent / "review_pages"
        elif redacted_raw:
            review_dir = Path(redacted_raw).parent / "review_pages"
        else:
            review_dir = Path(tracker_csv).resolve().parent / str(doc_id) / "review_pages"

    if not review_dir.exists():
        raise FileNotFoundError(f"review_pages_dir not found: {review_dir}")
    if not source_file.exists():
        raise FileNotFoundError(f"source file not found: {source_file}")

    page_images = sorted(review_dir.glob("*.png"))
    page_idx = page_number - 1
    if page_idx < 0 or page_idx >= len(page_images):
        raise IndexError(f"page_number out of range; max is {len(page_images)}")

    page_path = page_images[page_idx]
    current = Image.open(page_path).convert("RGB")
    base = _load_source_page_image(
        source_file=source_file,
        source_ext=source_ext,
        page_number=page_number,
        target_size=current.size,
    )

    draw = ImageDraw.Draw(base)
    applied: List[Tuple[int, int, int, int]] = []
    applied_items: List[Dict[str, Any]] = []
    applied_text: List[Dict[str, Any]] = []
    width, height = base.size
    item_inputs = list(redaction_items or [])
    if item_inputs:
        normalized_inputs: List[Dict[str, Any]] = []
        for item in item_inputs:
            bbox = item.get("bbox_xyxy", [0, 0, 0, 0])
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            normalized_inputs.append(
                {
                    "bbox_xyxy": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    "tag": str(item.get("tag", "") or ""),
                    "text": str(item.get("text", "") or ""),
                }
            )
    else:
        normalized_inputs = [
            {"bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)], "tag": "manual_override", "text": ""}
            for (x1, y1, x2, y2) in boxes_xyxy
        ]

    for item in normalized_inputs:
        x1, y1, x2, y2 = item["bbox_xyxy"]
        xa, xb = sorted([int(x1), int(x2)])
        ya, yb = sorted([int(y1), int(y2)])
        xa = max(0, min(xa, width))
        xb = max(0, min(xb, width))
        ya = max(0, min(ya, height))
        yb = max(0, min(yb, height))
        if xa == xb or ya == yb:
            continue
        draw.rectangle([xa, ya, xb, yb], fill="black")
        applied.append((xa, ya, xb, yb))
        applied_items.append(
            {
                "tag": str(item.get("tag", "") or "manual_override"),
                "text": str(item.get("text", "") or ""),
                "bbox_xyxy": [xa, ya, xb, yb],
            }
        )

    if text_overlays is None:
        eu_raw = pd.to_numeric(row.get("eu_mode", 0), errors="coerce")
        is_eu_mode = bool(0 if pd.isna(eu_raw) else int(eu_raw))
        if is_eu_mode:
            dob_anchor = str(row.get("raw_dob", "") or row.get("dob", "") or "").strip()
            if not _parse_date_for_age_days(dob_anchor):
                for item in normalized_inputs:
                    if str(item.get("tag", "") or "").strip().lower() == "dob":
                        fallback_dob = str(item.get("text", "") or "").strip()
                        if _parse_date_for_age_days(fallback_dob):
                            dob_anchor = fallback_dob
                            break
            fallback_days: Optional[int] = None
            try:
                norm_age = _normalize_age_at_event(str(row.get("age_at_event", "") or "").strip())
                fallback_days = int(norm_age) if norm_age else None
            except Exception:
                fallback_days = None
            overlays = _build_age_days_overlays_from_items(
                normalized_inputs,
                dob_anchor,
                fallback_age_days=fallback_days,
            )
        else:
            overlays = []
    else:
        overlays = list(text_overlays or [])
    for overlay in overlays:
        layout = _resolve_overlay_text_layout(
            draw,
            overlay,
            width,
            height,
            default_size=18,
            min_size=8,
            max_size=96,
        )
        if not layout:
            continue
        text = str(layout["text"])
        x = int(layout["x"])
        y = int(layout["y"])
        size = int(layout["size"])
        font = layout["font"]
        xa, ya, xb, yb = [int(v) for v in layout["bbox_xyxy"]]
        has_bbox = bool(layout["has_bbox"])
        color = str(overlay.get("color", "white") or "white")
        bg = str(overlay.get("bg", "") or "").strip().lower()
        stroke_fill = str(overlay.get("stroke_fill", "") or "").strip()
        stroke_width_raw = overlay.get("stroke_width", 0)
        try:
            stroke_width = max(0, int(float(stroke_width_raw)))
        except Exception:
            stroke_width = 0
        if bg:
            try:
                tx1, ty1, tx2, ty2 = draw.textbbox((x, y), text, font=font)
            except Exception:
                tx1, ty1, tx2, ty2 = (
                    x,
                    y,
                    x + max(1, int(size * max(1, len(text)) * 0.55)),
                    y + max(1, int(size * 1.2)),
                )
            pad = max(1, int(round(size * 0.12)))
            draw.rectangle([tx1 - pad, ty1 - pad, tx2 + pad, ty2 + pad], fill=bg)
        draw.text(
            (x, y),
            text,
            fill=color,
            font=font,
            stroke_width=stroke_width,
            stroke_fill=(stroke_fill or None),
        )
        applied_text.append(
            {
                "text": text,
                "x": x,
                "y": y,
                "size": size,
                "color": color,
                "stroke_fill": stroke_fill,
                "stroke_width": int(stroke_width),
                "bbox_xyxy": [int(xa), int(ya), int(xb), int(yb)] if has_bbox else [int(x), int(y), int(x), int(y)],
            }
        )

    base.save(page_path)
    _rebuild_redacted_document_from_review_pages(row)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        page_idx_manifest = page_number - 1
        for page in manifest.get("pages", []):
            if int(page.get("page_index", -1)) == page_idx_manifest:
                page["redaction_boxes"] = applied_items
                page["text_overlays"] = applied_text
                break
        current_total_boxes = int(
            sum(len(p.get("redaction_boxes", []) or []) for p in manifest.get("pages", []))
        )
        original_total_boxes = int(
            manifest.get("original_total_redaction_boxes", manifest.get("total_redaction_boxes", current_total_boxes))
        )
        user_deleted = max(0, original_total_boxes - current_total_boxes)
        user_added = max(0, current_total_boxes - original_total_boxes)
        manual = manifest.get("manual_redaction", {})
        manual["enabled"] = True
        manual["last_override_page"] = int(page_number)
        manual["override_box_count"] = int(len(applied))
        manual["text_overlay_count"] = int(len(applied_text))
        manual["user_deleted_box_count"] = int(user_deleted)
        manual["user_added_box_count"] = int(user_added)
        manifest["manual_redaction"] = manual
        manifest["total_redaction_boxes"] = int(current_total_boxes)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    existing_notes = str(df.loc[mask, "review_notes"].fillna("").iloc[-1]).strip()
    note = f"Override page {page_number} redactions with {len(applied)} box(es)"
    df.loc[mask, "review_notes"] = f"{existing_notes} | {note}".strip(" |")
    df.loc[mask, "review_status"] = "manual_redaction_applied"
    _queue_rows_for_resend(df, mask)
    save_tracker(df, tracker_csv)
    return df


def get_page_text_overlays(
    tracker_csv: str,
    doc_id: str,
    page_number: int,
) -> List[Dict[str, Any]]:
    df = load_tracker(tracker_csv)
    mask = df["doc_id"] == doc_id
    if not mask.any():
        return []
    row = df[mask].iloc[-1]
    manifest_path = Path(str(row["manifest_json"]))
    if not manifest_path.exists():
        return []

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    page_idx = page_number - 1
    for page in manifest.get("pages", []):
        if int(page.get("page_index", -1)) != page_idx:
            continue
        overlays = page.get("text_overlays", [])
        if not isinstance(overlays, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for item in overlays:
            try:
                text = str(item.get("text", "") or "").strip()
                if not text:
                    continue
                bbox = item.get("bbox_xyxy", [0, 0, 0, 0])
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    bbox = [int(float(item.get("x", 0) or 0)), int(float(item.get("y", 0) or 0)), int(float(item.get("x", 0) or 0)), int(float(item.get("y", 0) or 0))]
                normalized.append(
                    {
                        "text": text,
                        "x": int(float(item.get("x", 0) or 0)),
                        "y": int(float(item.get("y", 0) or 0)),
                        "size": int(float(item.get("size", 18) or 18)),
                        "color": str(item.get("color", "white") or "white"),
                        "bg": str(item.get("bg", "") or ""),
                        "stroke_fill": str(item.get("stroke_fill", "") or ""),
                        "stroke_width": int(float(item.get("stroke_width", 0) or 0)),
                        "bbox_xyxy": [int(float(bbox[0])), int(float(bbox[1])), int(float(bbox[2])), int(float(bbox[3]))],
                    }
                )
            except Exception:
                continue
        return normalized
    return []


def compile_redacted_document(tracker_csv: str, doc_id: str) -> pd.DataFrame:
    """
    Rebuild final redacted document from review page PNGs for one doc.
    Useful as an explicit "compile" action after manual edits.
    """
    df = load_tracker(tracker_csv)
    mask = df["doc_id"] == doc_id
    if not mask.any():
        raise ValueError(f"doc_id not found in tracker: {doc_id}")
    row = df[mask].iloc[-1]
    _rebuild_redacted_document_from_review_pages(row)
    existing_notes = str(df.loc[mask, "review_notes"].fillna("").iloc[-1]).strip()
    note = "Compiled redacted document from edited review pages"
    df.loc[mask, "review_notes"] = f"{existing_notes} | {note}".strip(" |")
    _queue_rows_for_resend(df, mask)
    save_tracker(df, tracker_csv)
    return df


def apply_manual_box_redaction(
    tracker_csv: str,
    doc_id: str,
    page_number: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> pd.DataFrame:
    if page_number < 1:
        raise ValueError("page_number must be >= 1")

    df = load_tracker(tracker_csv)
    mask = df["doc_id"] == doc_id
    if not mask.any():
        raise ValueError(f"doc_id not found in tracker: {doc_id}")

    row = df[mask].iloc[-1]
    review_dir = Path(str(row["review_pages_dir"]))
    manifest_path = Path(str(row["manifest_json"]))
    if not review_dir.exists():
        raise FileNotFoundError(f"review_pages_dir not found: {review_dir}")

    page_images = sorted(review_dir.glob("*.png"))
    page_idx = page_number - 1
    if page_idx >= len(page_images):
        raise IndexError(f"page_number out of range; max is {len(page_images)}")

    page_path = page_images[page_idx]
    image = Image.open(page_path).convert("RGB")
    width, height = image.size

    xa, xb = sorted([int(x1), int(x2)])
    ya, yb = sorted([int(y1), int(y2)])
    xa = max(0, min(xa, width))
    xb = max(0, min(xb, width))
    ya = max(0, min(ya, height))
    yb = max(0, min(yb, height))
    if xa == xb or ya == yb:
        raise ValueError("Manual box has zero area")

    draw = ImageDraw.Draw(image)
    draw.rectangle([xa, ya, xb, yb], fill="black")
    image.save(page_path)

    _rebuild_redacted_document_from_review_pages(row)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manual = manifest.get("manual_redaction", {})
        manual_boxes = manual.get("manual_boxes", [])
        manual_boxes.append(
            {
                "page_number": page_number,
                "bbox_xyxy": [xa, ya, xb, yb],
                "source": "manual_box",
            }
        )
        manifest["manual_redaction"] = {
            "enabled": True,
            "phrases": manual.get("phrases", []),
            "total_manual_boxes": int(manual.get("total_manual_boxes", 0)) + 1,
            "manual_boxes": manual_boxes,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    existing_notes = str(df.loc[mask, "review_notes"].fillna("").iloc[-1]).strip()
    note = f"Manual box page {page_number}: [{xa},{ya},{xb},{yb}]"
    df.loc[mask, "review_notes"] = f"{existing_notes} | {note}".strip(" |")
    df.loc[mask, "review_status"] = "manual_redaction_applied"
    _queue_rows_for_resend(df, mask)
    save_tracker(df, tracker_csv)
    return df


def apply_manual_boxes_redaction(
    tracker_csv: str,
    doc_id: str,
    page_number: int,
    boxes_xyxy: Sequence[Tuple[int, int, int, int]],
) -> pd.DataFrame:
    updated_df: Optional[pd.DataFrame] = None
    for x1, y1, x2, y2 in boxes_xyxy:
        updated_df = apply_manual_box_redaction(
            tracker_csv=tracker_csv,
            doc_id=doc_id,
            page_number=page_number,
            x1=int(x1),
            y1=int(y1),
            x2=int(x2),
            y2=int(y2),
        )
    if updated_df is None:
        raise ValueError("No manual boxes provided")
    return updated_df


def apply_manual_phrase_redaction(
    tracker_csv: str,
    doc_id: str,
    phrases: Sequence[str],
    pad_ratio_x: float = DEFAULT_PAD_RATIO_X,
    pad_ratio_y: float = DEFAULT_PAD_RATIO_Y,
) -> Tuple[pd.DataFrame, int]:
    normalized = [value.strip() for value in phrases if value and value.strip()]
    if not normalized:
        raise ValueError("Provide at least one phrase for manual redaction")

    df = load_tracker(tracker_csv)
    mask = df["doc_id"] == doc_id
    if not mask.any():
        raise ValueError(f"doc_id not found in tracker: {doc_id}")

    row = df[mask].iloc[-1]
    review_dir = Path(row["review_pages_dir"])
    manifest_path = Path(row["manifest_json"])
    cache_ocr_dir = Path(tracker_csv).resolve().parent / ".cache" / doc_id / "ocr"
    if not review_dir.exists():
        raise FileNotFoundError(f"review_pages_dir not found: {review_dir}")
    if not cache_ocr_dir.exists():
        raise FileNotFoundError(f"OCR cache not found: {cache_ocr_dir}")

    total_manual_boxes = 0
    review_images = sorted(review_dir.glob("*.png"))
    for index, image_path in enumerate(review_images):
        page_json = cache_ocr_dir / f"page_{index + 1:04d}.json"
        if not page_json.exists():
            continue
        page = _load_page_cache(page_json)
        manual_spans = _find_phrase_spans(page.text, normalized)
        manual_boxes = pii_spans_to_redaction_boxes(page, manual_spans)
        total_manual_boxes += len(manual_boxes)
        if manual_boxes:
            image = Image.open(image_path).convert("RGB")
            image = apply_redaction_boxes(
                image,
                manual_boxes,
                pad_ratio_x=pad_ratio_x,
                pad_ratio_y=pad_ratio_y,
            )
            image.save(image_path)

    _rebuild_redacted_document_from_review_pages(row)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing_manual = manifest.get("manual_redaction", {})
    merged_phrases = list(dict.fromkeys([*existing_manual.get("phrases", []), *normalized]))
    manifest["manual_redaction"] = {
        "enabled": True,
        "phrases": merged_phrases,
        "total_manual_boxes": int(existing_manual.get("total_manual_boxes", 0)) + total_manual_boxes,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    df.loc[mask, "review_status"] = "manual_redaction_applied"
    df.loc[mask, "review_notes"] = (
        df.loc[mask, "review_notes"].fillna("") + f" | Manual phrases: {', '.join(normalized)}"
    ).str.strip(" |")
    _queue_rows_for_resend(df, mask)
    save_tracker(df, tracker_csv)
    return df, total_manual_boxes


def collect_approved_files(
    tracker_csv: str,
    approved_output_dir: Optional[str] = None,
    copy_review_pngs: bool = False,
) -> List[str]:
    df = load_tracker(tracker_csv)
    approved_df = df[df["approved_to_send"] == 1].copy()
    if approved_df.empty:
        logger.info("No approved files found")
        return []

    tracker_dir = Path(tracker_csv).resolve().parent
    if approved_output_dir:
        provided = Path(approved_output_dir).resolve()
        if provided.name.lower() == "pdf":
            root_dir = provided.parent
            out_dir = ensure_dir(provided)
        else:
            root_dir = provided
            out_dir = ensure_dir(root_dir / "pdf")
    else:
        root_dir = tracker_dir / "approved_for_transfer"
        out_dir = ensure_dir(root_dir / "pdf")

    copied: List[str] = []
    for _, row in approved_df.iterrows():
        src = Path(row["redacted_file"])
        if src.exists():
            dest = out_dir / src.name
            shutil.copy2(src, dest)
            copied.append(str(dest))

        if copy_review_pngs:
            review_dir = Path(row["review_pages_dir"])
            if review_dir.exists():
                review_out = ensure_dir(root_dir / "review_pages" / review_dir.name)
                for png in review_dir.glob("*.png"):
                    shutil.copy2(png, review_out / png.name)
    return copied


def run_ablation(
    input_dir: str,
    detector_backends: Sequence[str] = ("nlp", "hybrid", "openpipe"),
    device: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
    overwrite: bool = False,
) -> pd.DataFrame:
    runs: List[pd.DataFrame] = []
    root = Path(input_dir).expanduser().resolve()
    for backend in detector_backends:
        logger.info("Running ablation backend=%s", backend)
        run_df = process_reports_local(
            input_dir=input_dir,
            output_dir=str(root / f"redaction_output_{backend}"),
            cache_dir=str(root / "redaction_ablation_cache"),
            device=device,
            dpi=dpi,
            overwrite=overwrite,
            detector_backend=backend,
        ).copy()
        run_df["ablation_backend"] = backend
        runs.append(run_df)

    combined = pd.concat(runs, ignore_index=True)
    summary = (
        combined.groupby("ablation_backend", dropna=False)
        .agg(
            docs=("doc_id", "count"),
            phi_found_docs=("phi_found", "sum"),
            total_pii_spans=("total_pii_spans", "sum"),
            total_boxes=("total_redaction_boxes", "sum"),
            mean_runtime_sec=("duration_sec", "mean"),
        )
        .reset_index()
    )
    out_csv = root / "ablation_summary.csv"
    summary.to_csv(out_csv, index=False)
    logger.info("Ablation summary written to %s", out_csv)
    return summary


def process_images_local(input_dir: str) -> List[Dict[str, Any]]:
    df = process_reports_local(input_dir=input_dir)
    return df.to_dict(orient="records")

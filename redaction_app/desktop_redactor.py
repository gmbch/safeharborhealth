from __future__ import annotations

import json
import html
import re
import shutil
import sys
import time
import uuid
import ctypes
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Sequence

import pandas as pd
import numpy as np
from PIL import Image

# Ensure local src imports work even when launched from another cwd.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from PySide6.QtCore import QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QPen, QPixmap, QShortcut, QKeySequence, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QTextBrowser,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolBox,
    QVBoxLayout,
    QWidget,
)

from aws_transfer import (
    generate_local_redacted_ocr_for_doc,
    request_presigned_upload_via_api,
    upload_approved_files_via_api,
)
from cognito_auth import CognitoToken, load_cognito_config, login_with_cognito
from site_catalog import load_site_ids
from utils import (
    apply_manual_phrase_redaction,
    compile_redacted_document,
    collect_approved_files,
    infer_raw_dates_from_manifest,
    list_input_files,
    get_page_redaction_items,
    get_page_text_overlays,
    process_reports_local,
    save_pdf_from_images,
    load_tracker,
    save_tracker,
    override_page_redactions,
    set_review_decision,
    set_review_metadata,
)


Box = Tuple[int, int, int, int]
SCREENSHOT_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
DEFAULT_PAGE_ZOOM = 0.75
DEFAULT_SAFEHARBOR_API_BASE_URL = "https://mn8uzpl56b.execute-api.us-east-1.amazonaws.com/dev"


def _asset_candidates(*names: str) -> list[Path]:
    paths: list[Path] = []
    base_dirs = [THIS_DIR, THIS_DIR.parent]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        base_dirs.insert(0, Path(str(meipass)))
    for base in base_dirs:
        for name in names:
            paths.append(base / name)
    return paths


def _resolve_app_icon_path() -> Path | None:
    # Prefer ICO for Windows taskbar / packaged EXE.
    for p in _asset_candidates("lighthouse-logo.png", "app_icon.ico", "force-logo.ico", "force-logo.png", "force-log.png"):
        if p.exists() and p.is_file():
            return p
    return None


class BatchProcessWorker(QThread):
    batch_done = Signal(object, int, int, object)
    batch_error = Signal(str)

    def __init__(
        self,
        *,
        input_dir: str,
        output_dir: str | None,
        overwrite: bool,
        detector_backend: str,
        ocr_backend: str,
        pdf_text_mode: str,
        device: str,
        batch_files: List[str],
        batch_index: int,
        total_batches: int,
        eu_mode: bool = False,
        full_date_overlay_mode: bool = False,
    ) -> None:
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.detector_backend = detector_backend
        self.ocr_backend = ocr_backend
        self.pdf_text_mode = pdf_text_mode
        self.device = device
        self.batch_files = batch_files
        self.batch_index = batch_index
        self.total_batches = total_batches
        self.eu_mode = bool(eu_mode)
        self.full_date_overlay_mode = bool(full_date_overlay_mode)

    def run(self) -> None:
        try:
            df = process_reports_local(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                detector_backend=self.detector_backend,
                ocr_backend=self.ocr_backend,
                pdf_text_mode=self.pdf_text_mode,
                device=self.device,
                overwrite=self.overwrite,
                source_files=self.batch_files,
                append_to_tracker=True,
                eu_mode=self.eu_mode,
                full_date_overlay_mode=self.full_date_overlay_mode,
            )
            self.batch_done.emit(df, self.batch_index, self.total_batches, self.batch_files)
        except Exception as exc:
            self.batch_error.emit(str(exc))


class ProcessReportsWorker(QThread):
    process_done = Signal(object, object)
    process_progress = Signal(int, int, str, float)
    process_error = Signal(str)

    def __init__(
        self,
        *,
        input_dir: str,
        output_dir: str | None,
        tracker_csv_path: str | None,
        append_to_tracker: bool,
        overwrite: bool,
        detector_backend: str,
        ocr_backend: str,
        pdf_text_mode: str,
        device: str,
        source_files: List[str],
        eu_mode: bool = False,
        full_date_overlay_mode: bool = False,
    ) -> None:
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.tracker_csv_path = tracker_csv_path
        self.append_to_tracker = append_to_tracker
        self.overwrite = overwrite
        self.detector_backend = detector_backend
        self.ocr_backend = ocr_backend
        self.pdf_text_mode = pdf_text_mode
        self.device = device
        self.source_files = source_files
        self.eu_mode = bool(eu_mode)
        self.full_date_overlay_mode = bool(full_date_overlay_mode)

    def run(self) -> None:
        try:
            started_at = time.perf_counter()

            def _on_progress(done: int, total: int, filename: str) -> None:
                elapsed = max(0.0, time.perf_counter() - started_at)
                self.process_progress.emit(int(done), int(total), str(filename), float(elapsed))

            df = process_reports_local(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                detector_backend=self.detector_backend,
                ocr_backend=self.ocr_backend,
                pdf_text_mode=self.pdf_text_mode,
                device=self.device,
                overwrite=self.overwrite,
                source_files=self.source_files,
                append_to_tracker=self.append_to_tracker,
                tracker_csv_path=self.tracker_csv_path,
                progress_callback=_on_progress,
                eu_mode=self.eu_mode,
                full_date_overlay_mode=self.full_date_overlay_mode,
            )
            self.process_done.emit(df, self.source_files)
        except Exception as exc:
            self.process_error.emit(str(exc))

APP_QSS = """
QMainWindow, QWidget {
    background-color: #111111;
    color: #f5f5f5;
    font-family: "Segoe UI", "Inter", "Arial";
    font-size: 13px;
}

QTabWidget::pane {
    border: 1px solid #3a3a3a;
    border-radius: 10px;
    background: #1a1a1a;
    top: -1px;
}

QTabBar::tab {
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    color: #f2f2f2;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 8px 14px;
    margin-right: 6px;
    min-width: 110px;
}

QTabBar::tab:selected {
    background: #a10016;
    color: #ffffff;
    border-bottom-color: #a10016;
    font-weight: 600;
}

QLabel {
    color: #f5f5f5;
}

QLineEdit, QComboBox, QSpinBox, QTextEdit, QListWidget, QTableWidget, QGraphicsView {
    background: #1f1f1f;
    border: 1px solid #3a3a3a;
    border-radius: 8px;
    padding: 6px 8px;
    color: #f5f5f5;
    selection-background-color: #7f0011;
    selection-color: #ffffff;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QTextEdit:focus, QListWidget:focus, QTableWidget:focus, QGraphicsView:focus {
    border: 1px solid #d71920;
}

QPushButton {
    background: #b30018;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 8px 12px;
    font-weight: 600;
}

QPushButton:hover {
    background: #d71920;
}

QPushButton:pressed {
    background: #8f0015;
}

QPushButton:disabled {
    background: #5f5f5f;
    color: #d0d0d0;
}

QHeaderView::section {
    background: #2a2a2a;
    color: #f5f5f5;
    border: none;
    border-right: 1px solid #3a3a3a;
    border-bottom: 1px solid #3a3a3a;
    padding: 6px;
    font-weight: 600;
}

QScrollBar:vertical {
    background: #222222;
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: #a10016;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #d71920;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""
BUILD_STAMP = "manual-edits-r3-2026-05-15"


class RedactionGraphicsView(QGraphicsView):
    box_drawn = Signal(tuple)
    box_selected = Signal(str, int)
    box_moved = Signal(str, int, tuple)
    resize_mode_changed = Signal(str, int)

    def __init__(self, scene: QGraphicsScene) -> None:
        super().__init__(scene)
        self._drawing = False
        self._start = QPointF()
        self._temp_item: QGraphicsRectItem | None = None
        self._drag_item: QGraphicsRectItem | None = None
        self._drag_kind: str | None = None
        self._drag_idx: int = -1
        self._drag_start = QPointF()
        self._drag_orig_rect = QRectF()
        self._resize_item: QGraphicsRectItem | None = None
        self._resize_kind: str | None = None
        self._resize_idx: int = -1
        self._resize_orig_rect = QRectF()
        self._resize_handle: str | None = None
        self._resize_mode_kind: str | None = None
        self._resize_mode_idx: int = -1
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMouseTracking(True)

    def _resize_hit(self, rect: QRectF, p: QPointF) -> str | None:
        # Generous hitbox so users can reliably resize.
        tol = 16.0
        l = abs(p.x() - rect.left()) <= tol
        r = abs(p.x() - rect.right()) <= tol
        t = abs(p.y() - rect.top()) <= tol
        b = abs(p.y() - rect.bottom()) <= tol
        if l and t:
            return "tl"
        if r and t:
            return "tr"
        if l and b:
            return "bl"
        if r and b:
            return "br"
        if l:
            return "l"
        if r:
            return "r"
        if t:
            return "t"
        if b:
            return "b"
        return None

    def _resize_hit_view(self, rect_scene: QRectF, pos_view) -> str | None:
        # Screen-pixel hit testing so resize works consistently at any zoom.
        tl = self.mapFromScene(rect_scene.topLeft())
        br = self.mapFromScene(rect_scene.bottomRight())
        left = min(tl.x(), br.x())
        right = max(tl.x(), br.x())
        top = min(tl.y(), br.y())
        bottom = max(tl.y(), br.y())
        tol = 12
        l = abs(int(pos_view.x()) - int(left)) <= tol
        r = abs(int(pos_view.x()) - int(right)) <= tol
        t = abs(int(pos_view.y()) - int(top)) <= tol
        b = abs(int(pos_view.y()) - int(bottom)) <= tol
        if l and t:
            return "tl"
        if r and t:
            return "tr"
        if l and b:
            return "bl"
        if r and b:
            return "br"
        if l:
            return "l"
        if r:
            return "r"
        if t:
            return "t"
        if b:
            return "b"
        return None

    def _resize_rect(self, rect: QRectF, cur: QPointF, handle: str) -> QRectF:
        out = QRectF(rect)
        if "l" in handle:
            out.setLeft(cur.x())
        if "r" in handle:
            out.setRight(cur.x())
        if "t" in handle:
            out.setTop(cur.y())
        if "b" in handle:
            out.setBottom(cur.y())
        out = out.normalized()
        if out.width() < 3.0:
            out.setRight(out.left() + 3.0)
        if out.height() < 3.0:
            out.setBottom(out.top() + 3.0)
        return out

    def clear_resize_mode(self) -> None:
        self._resize_mode_kind = None
        self._resize_mode_idx = -1
        self.resize_mode_changed.emit("", -1)

    def mouseDoubleClickEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            clicked_item = self.itemAt(event.pos())
            if isinstance(clicked_item, QGraphicsRectItem):
                payload = clicked_item.data(0)
                if isinstance(payload, tuple) and len(payload) == 2:
                    kind = str(payload[0])
                    try:
                        idx = int(payload[1])
                    except Exception:
                        idx = -1
                    if idx >= 0:
                        if self._resize_mode_kind == kind and self._resize_mode_idx == idx:
                            self.clear_resize_mode()
                        else:
                            self._resize_mode_kind = kind
                            self._resize_mode_idx = idx
                            self.resize_mode_changed.emit(kind, idx)
                        event.accept()
                        return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            clicked_item = self.itemAt(event.pos())
            if clicked_item is not None:
                payload = clicked_item.data(0)
                if isinstance(payload, tuple) and len(payload) == 2:
                    kind = str(payload[0])
                    try:
                        idx = int(payload[1])
                    except Exception:
                        idx = -1
                    if idx >= 0:
                        # Start drag or resize for existing box.
                        if isinstance(clicked_item, QGraphicsRectItem):
                            scene_pt = self.mapToScene(event.pos())
                            if self._resize_mode_kind == kind and self._resize_mode_idx == idx:
                                handle = self._resize_hit_view(clicked_item.sceneBoundingRect(), event.pos())
                                if handle:
                                    self._resize_item = clicked_item
                                    self._resize_kind = kind
                                    self._resize_idx = idx
                                    self._resize_orig_rect = QRectF(clicked_item.rect())
                                    self._resize_handle = handle
                                    # Active resize visual cue: dark-blue state while handle drag is engaged.
                                    clicked_item.setPen(QPen(QColor(0, 55, 170), 3))
                                    clicked_item.setBrush(QColor(0, 55, 170, 90))
                            else:
                                self._drag_item = clicked_item
                                self._drag_kind = kind
                                self._drag_idx = idx
                                self._drag_start = scene_pt
                                self._drag_orig_rect = QRectF(clicked_item.rect())
                        self.box_selected.emit(kind, idx)
                        event.accept()
                        return
            self._drawing = True
            self._start = self.mapToScene(event.pos())
            self._temp_item = QGraphicsRectItem()
            self._temp_item.setPen(QPen(QColor(0, 102, 255), 2))
            self.scene().addItem(self._temp_item)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # type: ignore[override]
        if (
            self._resize_item is not None
            and self._resize_kind is not None
            and self._resize_idx >= 0
            and self._resize_handle
        ):
            cur = self.mapToScene(event.pos())
            self._resize_item.setRect(self._resize_rect(self._resize_orig_rect, cur, self._resize_handle))
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if self._drag_item is not None and self._drag_kind is not None and self._drag_idx >= 0:
            cur = self.mapToScene(event.pos())
            delta = cur - self._drag_start
            self._drag_item.setRect(self._drag_orig_rect.translated(delta))
            event.accept()
            return
        if self._drawing and self._temp_item is not None:
            end = self.mapToScene(event.pos())
            self._temp_item.setRect(QRectF(self._start, end).normalized())
            event.accept()
            return
        # Hover affordance: show resize/move cursor for existing boxes.
        hovered = self.itemAt(event.pos())
        cursor = Qt.ArrowCursor
        if isinstance(hovered, QGraphicsRectItem):
            payload = hovered.data(0)
            if isinstance(payload, tuple) and len(payload) == 2:
                kind = str(payload[0])
                try:
                    idx = int(payload[1])
                except Exception:
                    idx = -1
                if idx >= 0 and self._resize_mode_kind == kind and self._resize_mode_idx == idx:
                    handle = self._resize_hit_view(hovered.sceneBoundingRect(), event.pos())
                    if handle in {"tl", "br", "tr", "bl", "l", "r", "t", "b"}:
                        # Grab-style cursor for resize interaction affordance.
                        cursor = Qt.OpenHandCursor
                    else:
                        cursor = Qt.ArrowCursor
                else:
                    cursor = Qt.SizeAllCursor
        self.setCursor(cursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # type: ignore[override]
        if self._resize_item is not None and event.button() == Qt.LeftButton:
            rect = self._resize_item.rect().normalized()
            kind = self._resize_kind or ""
            idx = int(self._resize_idx)
            self._resize_item = None
            self._resize_kind = None
            self._resize_idx = -1
            self._resize_handle = None
            if idx >= 0 and kind:
                x1, y1, x2, y2 = (
                    int(rect.left()),
                    int(rect.top()),
                    int(rect.right()),
                    int(rect.bottom()),
                )
                if x2 > x1 and y2 > y1:
                    self.box_moved.emit(kind, idx, (x1, y1, x2, y2))
            self.setCursor(Qt.OpenHandCursor)
            event.accept()
            return
        if self._drag_item is not None and event.button() == Qt.LeftButton:
            rect = self._drag_item.rect().normalized()
            kind = self._drag_kind or ""
            idx = int(self._drag_idx)
            self._drag_item = None
            self._drag_kind = None
            self._drag_idx = -1
            if idx >= 0 and kind:
                x1, y1, x2, y2 = (
                    int(rect.left()),
                    int(rect.top()),
                    int(rect.right()),
                    int(rect.bottom()),
                )
                if x2 > x1 and y2 > y1:
                    self.box_moved.emit(kind, idx, (x1, y1, x2, y2))
            event.accept()
            return
        if self._drawing and event.button() == Qt.LeftButton and self._temp_item is not None:
            rect = self._temp_item.rect().normalized()
            self.scene().removeItem(self._temp_item)
            self._temp_item = None
            self._drawing = False
            x1, y1, x2, y2 = int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom())
            if x2 > x1 and y2 > y1:
                self.box_drawn.emit((x1, y1, x2, y2))
            event.accept()
            return
        super().mouseReleaseEvent(event)


class HoverTextBrowser(QTextBrowser):
    token_hovered = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.setMouseTracking(True)
        # Qt bindings differ by version: some expose `highlighted`, others don't.
        highlighted = getattr(self, "highlighted", None)
        if highlighted is not None:
            try:
                highlighted.connect(self._emit_anchor)
            except Exception:
                pass

    def _emit_anchor(self, link: str) -> None:
        self.token_hovered.emit(str(link or ""))

    def mouseMoveEvent(self, event):  # type: ignore[override]
        try:
            pos = event.position().toPoint()  # Qt6
        except Exception:
            pos = event.pos()  # Fallback
        link = self.anchorAt(pos)
        self.token_hovered.emit(str(link or ""))
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):  # type: ignore[override]
        self.token_hovered.emit("")
        super().leaveEvent(event)


class DesktopRedactor(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"SafeHarborAI ({BUILD_STAMP})")
        self.resize(1450, 900)
        self.current_session_id = str(uuid.uuid4())
        self.cognito_token: CognitoToken | None = None

        self.tracker_path: str = ""
        self.doc_id: str = ""
        self.page_number = 1
        self.page_paths: List[Path] = []
        self.base_boxes: List[Box] = []
        self.base_box_items: List[dict] = []
        self.page_text_overlays: List[dict] = []
        self.manual_boxes: List[Box] = []
        self.selected_base_row: int = -1
        self.selected_manual_row: int = -1
        self.resize_mode_kind: str | None = None
        self.resize_mode_idx: int = -1
        self.current_zoom = DEFAULT_PAGE_ZOOM
        self._view_state: dict | None = None
        self.batch_files_all: List[str] = []
        self.batch_size = 10
        self.batch_cursor = 0
        self.batch_signature: tuple[str, str, bool] | None = None
        self.batch_sections: List[tuple[int, int, List[str]]] = []
        self.batch_worker: BatchProcessWorker | None = None
        self.process_worker: ProcessReportsWorker | None = None
        self._current_run_append_to_tracker = False
        self._current_run_tracker_path: str = ""
        self._doc_combo_doc_ids: List[str] = []
        self.undo_stack: List[dict] = []
        self._updating_review_table = False
        self._deleted_base_candidates: List[dict] = []
        self._deleted_candidates_scope: tuple[str, int] | None = None
        self._free_text_token_map: dict[str, str] = {}
        self._last_auto_age_at_event: str = ""
        self.site_id_options: List[str] = load_site_ids()
        if not self.site_id_options:
            self.site_id_options = ["BCH"]

        root = QWidget()
        self.setCentralWidget(root)
        shell = QVBoxLayout(root)
        header = QHBoxLayout()
        self.logo_label = QLabel()
        self.logo_label.setFixedHeight(48)
        self.logo_label.setMinimumWidth(160)
        self.logo_label.setStyleSheet(
            "border: 1px solid #cccccc; border-radius: 8px; padding: 4px; background-color: #ffffff;"
        )
        self.header_title = QLabel("SafeHarborAI")
        self.header_title.setStyleSheet("font-size: 18px; font-weight: 700; color: #f5f5f5;")
        header.addWidget(self.logo_label)
        header.addWidget(self.header_title, 1)
        shell.addLayout(header)
        self.tabs = QTabWidget()
        shell.addWidget(self.tabs)

        self.tab_link = QWidget()
        self.tab_review = QWidget()
        self.tab_manual = QWidget()
        self.tab_send = QWidget()
        self.tabs.addTab(self.tab_link, "1) Home")
        self.tabs.addTab(self.tab_manual, "2) Manual Edits")
        self.tabs.addTab(self.tab_review, "3) Review")
        self.tabs.addTab(self.tab_send, "4) Send to AWS")
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self._build_link_tab()
        self._build_manual_tab()
        self._build_review_tab()
        self._build_send_tab()
        self._try_autoload_logo()
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.setContext(Qt.ApplicationShortcut)
        self.undo_shortcut.activated.connect(self._on_ctrl_z)
        self.left_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.left_shortcut.setContext(Qt.ApplicationShortcut)
        self.left_shortcut.activated.connect(lambda: self._nudge_from_shortcut(-1, 0))
        self.right_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.right_shortcut.setContext(Qt.ApplicationShortcut)
        self.right_shortcut.activated.connect(lambda: self._nudge_from_shortcut(1, 0))
        self.up_shortcut = QShortcut(QKeySequence(Qt.Key_Up), self)
        self.up_shortcut.setContext(Qt.ApplicationShortcut)
        self.up_shortcut.activated.connect(lambda: self._nudge_from_shortcut(0, -1))
        self.down_shortcut = QShortcut(QKeySequence(Qt.Key_Down), self)
        self.down_shortcut.setContext(Qt.ApplicationShortcut)
        self.down_shortcut.activated.connect(lambda: self._nudge_from_shortcut(0, 1))

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._overlay_items: List[QGraphicsRectItem] = []

    def _build_link_tab(self) -> None:
        lay = QVBoxLayout(self.tab_link)
        top_row = QHBoxLayout()
        req_label = QLabel("Reviewer Name (Required)")
        req_label.setStyleSheet("color: #ff6b6b; font-weight: 700;")
        top_row.addWidget(req_label)
        self.user_name_edit = QLineEdit()
        self.user_name_edit.setPlaceholderText("Enter reviewer name")
        self.user_name_edit.setStyleSheet(
            "border: 2px solid #d71920; border-radius: 8px; background-color: #1f1f1f; color: #ffffff;"
        )
        self.user_name_edit.textChanged.connect(self._sync_identity_to_send)
        top_row.addWidget(self.user_name_edit, 2)
        site_label = QLabel("Site ID (Required)")
        site_label.setStyleSheet("color: #ff6b6b; font-weight: 700;")
        top_row.addWidget(site_label)
        self.site_id_home_combo = QComboBox()
        self.site_id_home_combo.addItems(self.site_id_options)
        self.site_id_home_combo.setStyleSheet(
            "border: 2px solid #d71920; border-radius: 8px; background-color: #1f1f1f; color: #ffffff;"
        )
        self.site_id_home_combo.currentTextChanged.connect(self._sync_identity_to_send)
        top_row.addWidget(self.site_id_home_combo, 1)
        top_row.addWidget(QLabel("Default Event Type (Optional)"))
        self.home_default_modality_combo = QComboBox()
        self.home_default_modality_combo.addItems(["", "CMR", "CT", "Echo", "StressTest", "Cath", "Other"])
        self.home_default_modality_combo.setMinimumWidth(180)
        top_row.addWidget(self.home_default_modality_combo, 1)
        top_row.addWidget(QLabel("Input Type"))
        self.home_input_type_combo = QComboBox()
        self.home_input_type_combo.addItems(["Report", "Free text"])
        self.home_input_type_combo.setMinimumWidth(140)
        self.home_input_type_combo.currentTextChanged.connect(self._on_home_input_type_changed)
        top_row.addWidget(self.home_input_type_combo, 1)
        lay.addLayout(top_row)

        title1 = QLabel("Report Folder Path")
        title1.setStyleSheet("font-weight: 600;")
        lay.addWidget(title1)
        self.report_folder_desc_label = QLabel(
            "Select a report folder. Report mode supports PDFs and screenshot event folders."
        )
        self.report_folder_desc_label.setWordWrap(True)
        lay.addWidget(self.report_folder_desc_label)
        self.report_guidance_label = QLabel(
            "Report mode auto-reads root-level PDFs and child folders that contain screenshots.\n"
            "For screenshots, put all files from the same patient event in one folder.\n"
            "Naming is optional, but a helpful format is fid_study-date (for example: FID12345_2026-06-08)."
        )
        self.report_guidance_label.setWordWrap(True)
        self.report_guidance_label.setStyleSheet(
            "border: 1px solid #d71920; border-radius: 8px; padding: 8px; "
            "background-color: rgba(215, 25, 32, 0.12); color: #ffdede;"
        )
        self.report_guidance_label.setVisible(False)
        lay.addWidget(self.report_guidance_label)
        row1 = QHBoxLayout()
        self.input_dir_edit = QLineEdit()
        self.input_dir_edit.textChanged.connect(self._update_default_output_hint)
        row1.addWidget(self.input_dir_edit, 1)
        browse_in = QPushButton("Browse")
        browse_in.clicked.connect(self._pick_input_dir)
        row1.addWidget(browse_in)
        lay.addLayout(row1)

        single_title = QLabel("Single Report File (Optional)")
        single_title.setStyleSheet("font-weight: 600;")
        lay.addWidget(single_title)
        single_desc = QLabel(
            "If provided, only this file is processed. Otherwise all files in Report Folder Path are processed."
        )
        single_desc.setWordWrap(True)
        lay.addWidget(single_desc)
        row1b = QHBoxLayout()
        self.single_file_edit = QLineEdit()
        row1b.addWidget(self.single_file_edit, 1)
        browse_single = QPushButton("Browse File")
        browse_single.clicked.connect(self._pick_single_file)
        row1b.addWidget(browse_single)
        lay.addLayout(row1b)

        title2 = QLabel("Output Folder Path (Optional)")
        title2.setStyleSheet("font-weight: 600;")
        lay.addWidget(title2)
        self.output_hint_label = QLabel("Default output path: <report_folder>\\redaction_output")
        self.output_hint_label.setWordWrap(True)
        lay.addWidget(self.output_hint_label)
        row2 = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        row2.addWidget(self.output_dir_edit, 1)
        browse_out = QPushButton("Browse")
        browse_out.clicked.connect(self._pick_output_dir)
        row2.addWidget(browse_out)
        lay.addLayout(row2)

        title3 = QLabel("Tracker CSV Path (Optional)")
        title3.setStyleSheet("font-weight: 600;")
        lay.addWidget(title3)
        desc3 = QLabel(
            "Use an existing tracker CSV, or provide a new file path. "
            "If blank, tracker defaults to <report_folder>\\redaction_tracker.csv. "
            "If the CSV does not exist yet, one will be created automatically."
        )
        desc3.setWordWrap(True)
        lay.addWidget(desc3)
        row3 = QHBoxLayout()
        self.tracker_edit = QLineEdit()
        row3.addWidget(self.tracker_edit, 1)
        pick_tracker = QPushButton("Open Tracker")
        pick_tracker.clicked.connect(self._pick_tracker_file)
        row3.addWidget(pick_tracker)
        lay.addLayout(row3)

        limits_row = QHBoxLayout()
        limits_row.addWidget(QLabel("Processing Limit"))
        self.batch_state_label = QLabel("Up to 1,000 reports per run.")
        limits_row.addWidget(self.batch_state_label, 1)
        lay.addLayout(limits_row)

        self.home_numeric_force_id_chk = QCheckBox("Numeric-based FORCE ID")
        self.home_numeric_force_id_chk.setToolTip("Use iterative numeric IDs (XXX-00000i-1) with MKF reuse checks.")
        self.home_numeric_force_id_chk.setStyleSheet(
            """
            QCheckBox {
                color: #ffffff;
                border: 2px solid #d71920;
                border-radius: 8px;
                padding: 6px 10px;
                background-color: rgba(215, 25, 32, 0.12);
                font-weight: 700;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #d71920;
                border-radius: 3px;
                background: #111111;
            }
            QCheckBox::indicator:checked {
                background: #d71920;
            }
            """
        )
        self.home_numeric_force_id_chk.stateChanged.connect(lambda _v: self._auto_generate_force_id())

        self.home_eu_mode_chk = QCheckBox("European Union Mode")
        self.home_eu_mode_chk.setToolTip("EU mode: redact full dates and use numeric FORCE IDs.")
        self.home_eu_mode_chk.setStyleSheet(
            """
            QCheckBox {
                color: #ffffff;
                border: 2px solid #d71920;
                border-radius: 8px;
                padding: 6px 10px;
                background-color: rgba(215, 25, 32, 0.12);
                font-weight: 700;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #d71920;
                border-radius: 3px;
                background: #111111;
            }
            QCheckBox::indicator:checked {
                background: #d71920;
            }
            """
        )
        self.home_eu_mode_chk.stateChanged.connect(self._on_home_eu_mode_changed)

        self.home_full_date_overlay_chk = QCheckBox("Full-Date DeID Overlay (Non-EU)")
        self.home_full_date_overlay_chk.setToolTip(
            "Redact full date text and overlay normalized de-identified date (MM/01/YYYY)."
        )
        self.home_full_date_overlay_chk.setStyleSheet(
            """
            QCheckBox {
                color: #ffffff;
                border: 2px solid #d71920;
                border-radius: 8px;
                padding: 6px 10px;
                background-color: rgba(215, 25, 32, 0.12);
                font-weight: 700;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #d71920;
                border-radius: 3px;
                background: #111111;
            }
            QCheckBox::indicator:checked {
                background: #d71920;
            }
            """
        )
        self.home_full_date_overlay_chk.setChecked(True)
        self.home_full_date_overlay_chk.stateChanged.connect(self._on_home_full_date_overlay_changed)

        self.start_btn = QPushButton("Run Redaction")
        self.start_btn.clicked.connect(self.run_redaction_next)
        lay.addWidget(self.start_btn)
        self.overwrite_cache_chk = QCheckBox("Overwrite OCR cache")
        self.overwrite_cache_chk.setStyleSheet(
            """
            QCheckBox {
                color: #ffffff;
                border: 1px solid #d71920;
                border-radius: 6px;
                padding: 6px 10px;
                background-color: #1f1f1f;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #d71920;
                border-radius: 3px;
                background: #111111;
            }
            QCheckBox::indicator:checked {
                background: #d71920;
            }
            """
        )
        checks_row = QHBoxLayout()
        checks_row.addWidget(self.home_full_date_overlay_chk)
        checks_row.addWidget(self.home_numeric_force_id_chk)
        checks_row.addWidget(self.home_eu_mode_chk)
        checks_row.addWidget(self.overwrite_cache_chk)
        checks_row.addStretch()
        lay.addLayout(checks_row)
        self.home_progress_label = QLabel("")
        lay.addWidget(self.home_progress_label)
        self.home_progress = QProgressBar()
        self.home_progress.setVisible(False)
        lay.addWidget(self.home_progress)
        lay.addWidget(QLabel("Home Log"))
        self.home_log = QTextEdit()
        self.home_log.setReadOnly(True)
        lay.addWidget(self.home_log, 1)

        self._on_home_input_type_changed(self.home_input_type_combo.currentText())

    def _on_home_input_type_changed(self, input_type: str) -> None:
        normalized_input_type = str(input_type or "").strip().lower()
        is_free = normalized_input_type == "free text"
        is_report = not is_free
        if hasattr(self, "start_btn"):
            self.start_btn.setText("Open Manual Edits" if is_free else "Run Redaction")
        if hasattr(self, "report_folder_desc_label"):
            self.report_folder_desc_label.setText(
                "Select a report folder. Report mode supports PDFs and screenshot event folders."
            )
        if hasattr(self, "report_guidance_label"):
            self.report_guidance_label.setVisible(is_report)
        if hasattr(self, "output_hint_label"):
            if is_free:
                self.output_hint_label.setText("Free text mode does not use output folder files.")
            else:
                self._update_default_output_hint()
        if is_free:
            self._reset_free_text_metadata()
        self._update_manual_input_mode()

    def _build_event_pdf_from_folder(self, folder_path: Path, aggregate_root: Path) -> Path:
        image_files = sorted(
            [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in SCREENSHOT_EXTS],
            key=lambda p: p.name.lower(),
        )
        if not image_files:
            raise ValueError(f"No screenshots found in folder: {folder_path}")

        aggregate_root.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", folder_path.name).strip("._") or "event_group"
        out_pdf = aggregate_root / f"{safe_name}.pdf"
        if out_pdf.exists():
            try:
                out_pdf.unlink()
            except Exception:
                pass

        pil_pages: List[Image.Image] = []
        for image_path in image_files:
            with Image.open(image_path) as img:
                pil_pages.append(img.convert("RGB").copy())
        if not pil_pages:
            raise ValueError(f"No readable screenshot files in folder: {folder_path}")

        # Use the same lossless helper as the core pipeline to preserve OCR fidelity.
        save_pdf_from_images(pil_pages, out_pdf)
        for page in pil_pages:
            try:
                page.close()
            except Exception:
                pass
        return out_pdf

    def _collect_non_free_text_sources(
        self,
        input_dir: str,
        single_file_raw: str,
    ) -> tuple[List[Path], str]:
        selected_files: List[Path] = []

        if single_file_raw:
            single_path = Path(single_file_raw).expanduser().resolve()
            if not single_path.exists() or not single_path.is_file():
                raise FileNotFoundError(f"Single report not found: {single_path}")
            ext = single_path.suffix.lower()
            if ext not in {".pdf", *SCREENSHOT_EXTS}:
                raise ValueError(
                    "Single report file must be a PDF or screenshot image file "
                    "(.png, .jpg, .jpeg, .tif, .tiff, .bmp, .webp)."
                )
            selected_files = [single_path]
            if not input_dir:
                input_dir = str(single_path.parent)
                self.input_dir_edit.setText(input_dir)
            return selected_files, input_dir

        input_path = Path(input_dir).expanduser().resolve()
        if not input_path.exists() or not input_path.is_dir():
            raise FileNotFoundError(f"Report folder not found: {input_path}")

        direct_files = sorted([p for p in input_path.iterdir() if p.is_file()], key=lambda p: p.name.lower())
        direct_pdfs = [p for p in direct_files if p.suffix.lower() == ".pdf"]
        direct_images = [p for p in direct_files if p.suffix.lower() in SCREENSHOT_EXTS]

        output_path, _ = self._resolve_output_tracker_paths()
        aggregate_root = output_path / ".event_group_inputs"

        child_dirs = sorted([p for p in input_path.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
        grouped_folder_pdfs: List[Path] = []
        for folder in child_dirs:
            has_images = any(
                p.is_file() and p.suffix.lower() in SCREENSHOT_EXTS for p in folder.iterdir()
            )
            if not has_images:
                continue
            grouped_folder_pdfs.append(self._build_event_pdf_from_folder(folder, aggregate_root))

        root_group_pdf: Path | None = None
        if not grouped_folder_pdfs and direct_images:
            # Singular event-folder mode: direct screenshots in selected folder become one event doc.
            root_group_pdf = self._build_event_pdf_from_folder(input_path, aggregate_root)
            self.home_log.append(
                "Screenshot aggregation: selected folder contains screenshots directly. "
                "Processing this folder as one multi-page event document."
            )

        if grouped_folder_pdfs:
            self.home_log.append(
                f"Screenshot aggregation: found {len(grouped_folder_pdfs)} folder(s) with screenshots. "
                "Processing each folder as one multi-page event document."
            )
        if direct_pdfs:
            self.home_log.append(f"PDF auto-detect: found {len(direct_pdfs)} PDF file(s) at folder root.")
        if grouped_folder_pdfs and direct_images:
            self.home_log.append(
                f"Found {len(direct_images)} stray root-level screenshot file(s). "
                "Ignored them; organize screenshots into event folders."
            )
            QMessageBox.warning(
                self,
                "Stray screenshot files ignored",
                f"Found {len(direct_images)} image file(s) directly under the selected folder.\n\n"
                "Please organize screenshots into singular event folders. "
                "Root-level image files were ignored.",
            )

        selected_files = [*direct_pdfs, *grouped_folder_pdfs]
        if root_group_pdf is not None:
            selected_files.append(root_group_pdf)
        if not selected_files:
            raise ValueError(
                "No processable files found. Auto-read supports root-level PDFs and child folders "
                "that contain screenshots. Please ensure screenshots from the same event are in a singular folder."
            )
        return selected_files, input_dir

    def _update_default_output_hint(self) -> None:
        raw = self.input_dir_edit.text().strip() if hasattr(self, "input_dir_edit") else ""
        if not raw:
            hint = "<report_folder>\\redaction_output"
        else:
            try:
                hint = str(Path(raw).expanduser().resolve() / "redaction_output")
            except Exception:
                hint = f"{raw}\\redaction_output"
        if hasattr(self, "output_hint_label"):
            self.output_hint_label.setText(f"Default output path: {hint}")

    def _build_review_tab(self) -> None:
        lay = QVBoxLayout(self.tab_review)
        site_row = QHBoxLayout()
        site_row.addWidget(QLabel("Site ID (used for AWS)"))
        self.site_id_review_combo = QComboBox()
        self.site_id_review_combo.addItems(self.site_id_options)
        self.site_id_review_combo.currentTextChanged.connect(self._on_review_site_id_changed)
        site_row.addWidget(self.site_id_review_combo, 1)
        lay.addLayout(site_row)
        self.review_summary = QLabel("Load tracker to see review summary.")
        lay.addWidget(self.review_summary)
        self.review_batch_btn = QPushButton("Process Next Batch")
        self.review_batch_btn.clicked.connect(self.run_redaction_next)
        self.review_batch_btn.setVisible(False)
        lay.addWidget(self.review_batch_btn)
        self.batch_results_label = QLabel("Batch Results")
        self.batch_results_label.setVisible(False)
        lay.addWidget(self.batch_results_label)
        self.batch_toolbox = QToolBox()
        self.batch_toolbox.setVisible(False)
        lay.addWidget(self.batch_toolbox, 1)
        lay.addWidget(QLabel("All Processed Cases"))
        self.review_table = QTableWidget()
        self.review_table.itemChanged.connect(self.on_review_table_item_changed)
        lay.addWidget(self.review_table, 1)

    def _build_manual_tab(self) -> None:
        main = QHBoxLayout(self.tab_manual)
        left = QVBoxLayout()
        self.manual_left_layout = left
        right_panel = QWidget()
        self.manual_right_panel = right_panel
        right_panel.setMaximumWidth(320)
        right = QVBoxLayout(right_panel)
        main.addLayout(left, 5)
        main.addWidget(right_panel, 1)

        self.pending_doc_combo = QComboBox()
        self.pending_doc_combo.setMinimumWidth(280)
        self.pending_doc_combo.currentIndexChanged.connect(self.on_pending_doc_selected)
        self.reviewed_doc_combo = QComboBox()
        self.reviewed_doc_combo.setMinimumWidth(280)
        self.reviewed_doc_combo.currentIndexChanged.connect(self.on_reviewed_doc_selected)
        left.addWidget(QLabel("Required Metadata"))
        manual_site_row = QHBoxLayout()
        manual_site_row.addWidget(QLabel("Site ID (used for AWS)"))
        self.site_id_manual_combo = QComboBox()
        self.site_id_manual_combo.addItems(self.site_id_options)
        self.site_id_manual_combo.currentTextChanged.connect(self._on_manual_site_id_changed)
        manual_site_row.addWidget(self.site_id_manual_combo, 1)
        left.addLayout(manual_site_row)
        meta_row = QHBoxLayout()
        meta_row.addWidget(QLabel("FORCE ID"))
        self.force_id_edit = QLineEdit("XXX-LLLFFF-1")
        self.force_id_edit.setMinimumWidth(170)
        meta_row.addWidget(self.force_id_edit, 2)
        meta_row.addWidget(QLabel("File ID"))
        self.file_id_edit = QLineEdit("")
        self.file_id_edit.setPlaceholderText("XXX-LLLFFF-1_YYYYMMDD_1")
        self.file_id_edit.setMinimumWidth(210)
        self.file_id_edit.setReadOnly(True)
        meta_row.addWidget(self.file_id_edit, 2)
        meta_row.addWidget(QLabel("Instance #"))
        self.instance_spin = QSpinBox()
        self.instance_spin.setRange(1, 999)
        self.instance_spin.setValue(1)
        meta_row.addWidget(self.instance_spin, 1)
        meta_row.addWidget(QLabel("Modality"))
        self.modality_combo = QComboBox()
        self.modality_combo.addItems(["", "CMR", "CT", "Echo", "StressTest", "Cath", "Other"])
        self.modality_combo.setMinimumWidth(120)
        meta_row.addWidget(self.modality_combo, 1)
        self.study_date_label = QLabel("Study Date (YYYY-MM-DD)")
        meta_row.addWidget(self.study_date_label)
        self.study_date_edit = QLineEdit("")
        self.study_date_edit.setMinimumWidth(130)
        meta_row.addWidget(self.study_date_edit, 1)
        self.age_at_event_label = QLabel("Age at Event (days)")
        self.age_at_event_edit = QLineEdit("")
        self.age_at_event_edit.setReadOnly(False)
        self.age_at_event_edit.setMinimumWidth(110)
        self.age_at_event_edit.setPlaceholderText("e.g., 3869")
        self.age_at_event_label.setVisible(False)
        self.age_at_event_edit.setVisible(False)
        meta_row.addWidget(self.age_at_event_label)
        meta_row.addWidget(self.age_at_event_edit, 1)
        left.addLayout(meta_row)
        meta_row2 = QHBoxLayout()
        meta_row2.addWidget(QLabel("First Name"))
        self.first_name_edit = QLineEdit("")
        meta_row2.addWidget(self.first_name_edit, 1)
        meta_row2.addWidget(QLabel("Last Name"))
        self.last_name_edit = QLineEdit("")
        meta_row2.addWidget(self.last_name_edit, 1)
        meta_row2.addWidget(QLabel("MRN"))
        self.mrn_edit = QLineEdit("")
        meta_row2.addWidget(self.mrn_edit, 1)
        meta_row2.addWidget(QLabel("DOB"))
        self.dob_edit = QLineEdit("")
        self.dob_edit.setPlaceholderText("YYYY-MM-DD")
        meta_row2.addWidget(self.dob_edit, 1)
        meta_row2.addWidget(QLabel("Gender"))
        self.gender_edit = QComboBox()
        self.gender_edit.addItems(["", "M", "F"])
        meta_row2.addWidget(self.gender_edit, 1)
        left.addLayout(meta_row2)
        self.force_id_edit.textChanged.connect(self._refresh_file_id_preview)
        self.first_name_edit.textChanged.connect(self._auto_generate_force_id)
        self.last_name_edit.textChanged.connect(self._auto_generate_force_id)
        self.study_date_edit.textChanged.connect(self._refresh_file_id_preview)
        self.study_date_edit.textChanged.connect(self._refresh_age_at_event)
        self.instance_spin.valueChanged.connect(self._refresh_file_id_preview)
        self.dob_edit.textChanged.connect(self._refresh_age_at_event)
        self.age_at_event_edit.textChanged.connect(self._sync_age_to_tracker_current_doc)
        decision_row = QHBoxLayout()
        self.ok_to_send_btn = QPushButton("OK to Send")
        self.ok_to_send_btn.setStyleSheet(
            "background: #1f8f3a; color: #ffffff; font-weight: 700; border-radius: 8px; padding: 8px 12px;"
        )
        self.ok_to_send_btn.clicked.connect(self.mark_ok_to_send)
        decision_row.addWidget(self.ok_to_send_btn, 1)
        self.not_ok_btn = QPushButton("Not OK to Send")
        self.not_ok_btn.setStyleSheet(
            "background: #b30018; color: #ffffff; font-weight: 700; border-radius: 8px; padding: 8px 12px;"
        )
        self.not_ok_btn.clicked.connect(self.mark_not_ok_to_send)
        decision_row.addWidget(self.not_ok_btn, 1)
        left.addLayout(decision_row)
        self.manual_pending_label = QLabel("Needs Review")
        left.addWidget(self.manual_pending_label)
        left.addWidget(self.pending_doc_combo)
        self.manual_reviewed_label = QLabel("Reviewed / Approved")
        left.addWidget(self.manual_reviewed_label)
        left.addWidget(self.reviewed_doc_combo)

        self.manual_page_controls = QWidget()
        page_row = QHBoxLayout(self.manual_page_controls)
        page_row.setContentsMargins(0, 0, 0, 0)
        page_row.addWidget(QLabel("Page"))
        self.page_spin = QSpinBox()
        self.page_spin.setMinimum(1)
        self.page_spin.valueChanged.connect(self.on_page_changed)
        page_row.addWidget(self.page_spin)
        self.page_indicator_label = QLabel("1/1")
        page_row.addWidget(self.page_indicator_label)
        self.prev_btn = QPushButton("← Prev Page")
        self.prev_btn.clicked.connect(self.prev_page)
        page_row.addWidget(self.prev_btn)
        self.next_btn = QPushButton("Next Page →")
        self.next_btn.clicked.connect(self.next_page)
        page_row.addWidget(self.next_btn)
        self.zoom_out_btn = QPushButton("Zoom -")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        page_row.addWidget(self.zoom_out_btn)
        self.zoom_in_btn = QPushButton("Zoom +")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        page_row.addWidget(self.zoom_in_btn)
        self.zoom_fit_btn = QPushButton("Fit")
        self.zoom_fit_btn.clicked.connect(self.fit_to_view)
        page_row.addWidget(self.zoom_fit_btn)
        self.zoom_reset_btn = QPushButton("100%")
        self.zoom_reset_btn.clicked.connect(self.reset_zoom)
        page_row.addWidget(self.zoom_reset_btn)
        self.zoom_label = QLabel(f"Zoom: {int(round(DEFAULT_PAGE_ZOOM * 100))}%")
        page_row.addWidget(self.zoom_label)
        left.addWidget(self.manual_page_controls)

        self.scene = QGraphicsScene()
        self.view = RedactionGraphicsView(self.scene)
        self.view.box_drawn.connect(self.on_box_drawn)
        self.view.box_selected.connect(self.on_box_selected)
        self.view.box_moved.connect(self.on_box_moved)
        self.view.resize_mode_changed.connect(self.on_resize_mode_changed)
        left.addWidget(self.view)

        right.addWidget(QLabel("AI/NLP Boxes (editable)"))
        self.base_list = QListWidget()
        self.base_list.currentRowChanged.connect(self.on_base_row_changed)
        right.addWidget(self.base_list)
        self.delete_base_btn = QPushButton("Delete Selected AI/NLP Box")
        self.delete_base_btn.clicked.connect(self.delete_selected_base_box)
        right.addWidget(self.delete_base_btn)
        right.addWidget(QLabel("Selected Box Inscription"))
        self.inscription_edit = QLineEdit("")
        self.inscription_edit.setPlaceholderText("Enter replacement text (e.g., 3869d)")
        right.addWidget(self.inscription_edit)
        self.set_inscription_btn = QPushButton("Set/Update Inscription")
        self.set_inscription_btn.clicked.connect(self.set_selected_box_inscription)
        right.addWidget(self.set_inscription_btn)
        self.clear_inscription_btn = QPushButton("Delete Inscription")
        self.clear_inscription_btn.clicked.connect(self.clear_selected_box_inscription)
        right.addWidget(self.clear_inscription_btn)
        self.undo_btn = QPushButton("Undo Last Move")
        self.undo_btn.clicked.connect(self.undo_last)
        right.addWidget(self.undo_btn)
        self.clear_btn = QPushButton("Remove All Moves")
        self.clear_btn.clicked.connect(self.clear_manual)
        right.addWidget(self.clear_btn)
        self.apply_btn = QPushButton("Run Box Redaction")
        self.apply_btn.clicked.connect(self.apply_page)
        right.addWidget(self.apply_btn)
        self.status = QLabel("Load tracker to begin.")
        right.addWidget(self.status)
        right.addStretch()

        self.manual_free_text_section = QWidget()
        free_lay = QVBoxLayout(self.manual_free_text_section)
        free_lay.addWidget(QLabel("Free Text Input"))
        self.free_text_input = QTextEdit()
        self.free_text_input.setPlaceholderText("Paste source text here...")
        free_lay.addWidget(self.free_text_input, 1)
        self.free_text_apply_btn = QPushButton("Run Free-text Redaction")
        self.free_text_apply_btn.clicked.connect(self._run_free_text_redaction)
        free_lay.addWidget(self.free_text_apply_btn)
        free_lay.addWidget(QLabel("Redacted Text"))
        self.free_text_output = HoverTextBrowser()
        self.free_text_output.setReadOnly(True)
        self.free_text_output.token_hovered.connect(self._on_free_text_token_hovered)
        free_lay.addWidget(self.free_text_output, 1)
        self.free_text_hover_value = QLabel("Hovered Original Value: -")
        self.free_text_hover_value.setWordWrap(True)
        free_lay.addWidget(self.free_text_hover_value)
        left.addWidget(self.manual_free_text_section, 2)
        self.manual_free_text_section.setVisible(False)
        self._update_eu_metadata_ui()

    def _build_send_tab(self) -> None:
        lay = QVBoxLayout(self.tab_send)
        intro = QLabel("Final confirmation before secure transfer to FORCE.")
        intro.setStyleSheet("font-weight: 600;")
        lay.addWidget(intro)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Send Source (folder or tracker CSV)"))
        self.send_source_edit = QLineEdit("")
        self.send_source_edit.setPlaceholderText("Choose report folder or redaction_tracker.csv")
        source_row.addWidget(self.send_source_edit, 1)
        self.send_source_browse_btn = QPushButton("Browse...")
        self.send_source_browse_btn.clicked.connect(self._pick_send_source)
        source_row.addWidget(self.send_source_browse_btn)
        self.send_source_load_btn = QPushButton("Load for Send")
        self.send_source_load_btn.clicked.connect(self._load_send_source)
        source_row.addWidget(self.send_source_load_btn)
        lay.addLayout(source_row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Site ID (Required)"))
        self.site_id_send_combo = QComboBox()
        self.site_id_send_combo.addItems(self.site_id_options)
        self.site_id_send_combo.currentTextChanged.connect(self._on_send_site_id_changed)
        row.addWidget(self.site_id_send_combo, 1)
        lay.addLayout(row)

        reviewer_row = QHBoxLayout()
        reviewer_row.addWidget(QLabel("Reviewer Name"))
        self.send_reviewer_edit = QLineEdit("")
        self.send_reviewer_edit.setPlaceholderText("Enter reviewer name")
        reviewer_row.addWidget(self.send_reviewer_edit, 1)
        lay.addLayout(reviewer_row)

        auth_row = QHBoxLayout()
        self.cognito_login_btn = QPushButton("Sign in to send")
        self.cognito_login_btn.clicked.connect(self.sign_in_for_send)
        auth_row.addWidget(self.cognito_login_btn)
        self.cognito_status = QLabel("Not signed in")
        auth_row.addWidget(self.cognito_status, 1)
        lay.addLayout(auth_row)

        self.send_summary = QLabel("Load tracker to view send summary.")
        self.send_summary.setWordWrap(True)
        lay.addWidget(self.send_summary)

        self.send_source_path = QLabel("Source path: -")
        self.send_source_path.setWordWrap(True)
        lay.addWidget(self.send_source_path)

        self.send_confirm_chk = QCheckBox(
            "Please confirm you have reviewed the redacted data thoroughly before sending to FORCE. "
            "This tool is intended as a first pass, not a final pass."
        )
        self.send_confirm_chk.setStyleSheet(
            "QCheckBox { "
            "padding: 10px 12px; "
            "border: 2px solid #d71920; "
            "border-radius: 8px; "
            "background-color: rgba(215, 25, 32, 0.12); "
            "font-weight: 700; "
            "}"
        )
        lay.addWidget(self.send_confirm_chk)
        self.test_api_btn = QPushButton("Test API Connection")
        self.test_api_btn.clicked.connect(self.test_api_connection)
        lay.addWidget(self.test_api_btn)
        self.send_btn = QPushButton("Send Approved Reports")
        self.send_btn.clicked.connect(self.send_approved)
        lay.addWidget(self.send_btn)
        self.send_progress = QProgressBar()
        self.send_progress.setVisible(False)
        lay.addWidget(self.send_progress)
        self.send_log = QTextEdit()
        self.send_log.setReadOnly(True)
        lay.addWidget(self.send_log, 1)

    def _pick_send_source(self) -> None:
        # Let user select either tracker CSV directly or a folder containing it.
        path_file, _ = QFileDialog.getOpenFileName(self, "Select Tracker CSV", "", "CSV Files (*.csv)")
        if path_file:
            self.send_source_edit.setText(path_file)
            return
        path_dir = QFileDialog.getExistingDirectory(self, "Select Folder Containing Tracker")
        if path_dir:
            self.send_source_edit.setText(path_dir)

    def _resolve_send_tracker_path(self, raw: str) -> str:
        candidate = str(raw or "").strip()
        if not candidate:
            return ""
        p = Path(candidate).expanduser().resolve()
        if p.is_dir():
            p = p / "redaction_tracker.csv"
        return str(p)

    def _load_send_source(self) -> None:
        raw = self.send_source_edit.text().strip() if hasattr(self, "send_source_edit") else ""
        tracker_path = self._resolve_send_tracker_path(raw)
        if not tracker_path:
            self.send_log.append("Enter a send source folder or tracker CSV.")
            return
        p = Path(tracker_path)
        if not p.exists():
            self.send_log.append(f"Tracker not found: {tracker_path}")
            return
        self.tracker_path = str(p)
        # Keep Home tracker field in sync for consistency.
        if hasattr(self, "tracker_edit"):
            self.tracker_edit.setText(self.tracker_path)
        try:
            self.open_tracker()
            self._refresh_send_summary()
            self.send_log.append(f"Loaded tracker for send: {self.tracker_path}")
        except Exception as exc:
            self.send_log.append(f"Failed to load tracker: {exc}")

    def _get_api_config_from_env(self) -> tuple[str, str, str, str, str | None]:
        import os

        config = load_cognito_config()
        base_url = (os.getenv("SAFEHARBOR_API_BASE_URL") or config.get("api_base_url", "")).strip().rstrip("/")
        if not base_url:
            base_url = DEFAULT_SAFEHARBOR_API_BASE_URL.strip().rstrip("/")
        if not base_url:
            return "", "", "", "", None
        presign_api = f"{base_url}/uploads/presign"
        complete_api = f"{base_url}/uploads/complete"
        session_start_api = f"{base_url}/uploads/session/start"
        review_api = f"{base_url}/review-events"
        return presign_api, complete_api, session_start_api, review_api, None

    def sign_in_for_send(self) -> None:
        try:
            self.cognito_token = login_with_cognito()
            self.cognito_status.setText("Signed in; site permissions checked by service")
            self.cognito_login_btn.setText("Sign in again")
            self.send_log.append("Cognito sign-in succeeded.")
        except Exception as exc:
            self.cognito_token = None
            self.cognito_status.setText("Not signed in")
            self.send_log.append(f"Cognito sign-in failed: {exc}")

    def _format_api_exception(self, exc: Exception) -> str:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        raw_text = str(exc)
        body_text = ""
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                parsed = response.json()
                if isinstance(parsed, dict):
                    detail = str(parsed.get("detail", "") or "").strip()
                    err = str(parsed.get("error", "") or "").strip()
                    if err and detail:
                        body_text = f"{err} ({detail})"
                    elif err:
                        body_text = err
                    else:
                        body_text = json.dumps(parsed)
                else:
                    body_text = str(parsed)
            except Exception:
                try:
                    body_text = str(getattr(response, "text", "") or "").strip()
                except Exception:
                    body_text = ""
        if not body_text:
            body_text = raw_text
        body_text = body_text.replace("\n", " ").strip()
        if len(body_text) > 500:
            body_text = body_text[:500] + "..."
        if status is not None:
            return f"HTTP {status}: {body_text}"
        return body_text

    def _sync_identity_to_send(self) -> None:
        site_id = self._get_selected_site_id()
        reviewer = self.user_name_edit.text().strip()
        self._set_site_id_across_tabs(site_id)
        # Seed Home -> Send only when the Send field is empty. Loading a
        # tracker refreshes the surrounding UI and can call this method; an
        # explicitly entered reviewer must survive that refresh.
        if reviewer and not self.send_reviewer_edit.hasFocus() and not self.send_reviewer_edit.text().strip():
            self.send_reviewer_edit.setText(reviewer)
        self._auto_generate_force_id()
        self._autofill_tracker_site_id_from_home()

    def _autofill_tracker_site_id_from_home(self) -> None:
        site_id = self._get_selected_site_id().strip()
        if not site_id or not self.tracker_path:
            return
        try:
            df = load_tracker(self.tracker_path)
            if "site_id" not in df.columns or df.empty:
                return
            needs_update = df["site_id"].fillna("").astype(str).str.strip().ne(site_id)
            if not bool(needs_update.any()):
                return
            df.loc[:, "site_id"] = site_id
            save_tracker(df, self.tracker_path)
            self._refresh_review_tab()
        except Exception:
            # Best-effort auto-fill; UI remains usable even if this fails.
            return

    def _on_review_site_id_changed(self, value: str) -> None:
        site_id = str(value or "").strip()
        self._set_site_id_across_tabs(site_id, source="review")
        self._autofill_tracker_site_id_from_home()

    def _on_manual_site_id_changed(self, value: str) -> None:
        site_id = str(value or "").strip()
        self._set_site_id_across_tabs(site_id, source="manual")
        self._autofill_tracker_site_id_from_home()

    def _on_send_site_id_changed(self, value: str) -> None:
        site_id = str(value or "").strip()
        self._set_site_id_across_tabs(site_id, source="send")
        self._autofill_tracker_site_id_from_home()

    def _get_selected_site_id(self) -> str:
        if hasattr(self, "site_id_home_combo"):
            return str(self.site_id_home_combo.currentText() or "").strip()
        return ""

    def _set_site_id_combo_value(self, combo: QComboBox, site_id: str) -> None:
        desired = str(site_id or "").strip()
        if not desired:
            return
        if str(combo.currentText() or "").strip() == desired:
            return
        idx = combo.findText(desired, Qt.MatchFixedString)
        if idx < 0:
            return
        blocked = combo.blockSignals(True)
        try:
            combo.setCurrentIndex(idx)
        finally:
            combo.blockSignals(blocked)

    def _set_site_id_across_tabs(self, site_id: str, source: str = "home") -> None:
        desired = str(site_id or "").strip()
        if not desired:
            return
        if hasattr(self, "site_id_home_combo") and source != "home":
            self._set_site_id_combo_value(self.site_id_home_combo, desired)
        if hasattr(self, "site_id_review_combo") and source != "review":
            self._set_site_id_combo_value(self.site_id_review_combo, desired)
        if hasattr(self, "site_id_manual_combo") and source != "manual":
            self._set_site_id_combo_value(self.site_id_manual_combo, desired)
        if hasattr(self, "site_id_send_combo") and source != "send":
            self._set_site_id_combo_value(self.site_id_send_combo, desired)

    def _refresh_send_summary(self) -> None:
        if not self.tracker_path:
            self.send_summary.setText("Load tracker to view send summary.")
            self.send_source_path.setText("Source path: -")
            return
        df = load_tracker(self.tracker_path)
        if "first_name" in df.columns:
            df = df.sort_values(by=["first_name", "doc_id"], kind="stable", na_position="last")
        processed = len(df)
        if "review_status" in df.columns:
            reviewed = int(df["review_status"].fillna("").astype(str).str.lower().eq("reviewed").sum())
        else:
            reviewed = 0
        if "approved_to_send" in df.columns:
            to_send = int(pd.to_numeric(df["approved_to_send"], errors="coerce").fillna(0).eq(1).sum())
        else:
            to_send = 0
        if "approved_to_send" in df.columns and "sent_to_aws" in df.columns:
            unsent_mask = (
                pd.to_numeric(df["approved_to_send"], errors="coerce").fillna(0).eq(1)
                & pd.to_numeric(df["sent_to_aws"], errors="coerce").fillna(0).ne(1)
            )
            unsent = int(
                unsent_mask.sum()
            )
        else:
            unsent = to_send
        self.send_summary.setText(
            f"Cases processed: {processed} | Cases reviewed: {reviewed} | Cases marked to send: {to_send} | Unsent approved: {unsent}"
        )
        tracker_dir = Path(self.tracker_path).resolve().parent
        staged_root = tracker_dir / "approved_for_transfer"
        self.send_source_path.setText(
            f"Transfer folders: PDF={staged_root / 'pdf'} | OCR={staged_root / 'ocr'}"
        )

    def _count_uploadable_approved_cases(self) -> int:
        if not self.tracker_path:
            return 0
        try:
            df = load_tracker(self.tracker_path)
        except Exception:
            return 0
        if "approved_to_send" not in df.columns:
            return 0
        mask = pd.to_numeric(df["approved_to_send"], errors="coerce").fillna(0).eq(1)
        if "sent_to_aws" in df.columns:
            mask = mask & pd.to_numeric(df["sent_to_aws"], errors="coerce").fillna(0).ne(1)
        approved = df.loc[mask].copy()
        if approved.empty:
            return 0

        valid_count = 0
        for _, row in approved.iterrows():
            redacted_file = Path(str(row.get("redacted_file", "")))
            if not redacted_file.exists():
                continue
            ocr_path = Path(str(row.get("redacted_ocr_json", "") or "").strip())
            if not ocr_path.exists():
                continue
            try:
                payload = json.loads(ocr_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    valid_count += 1
            except Exception:
                continue
        return valid_count

    def _pick_input_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select report folder")
        if path:
            self.input_dir_edit.setText(path)

    def _pick_single_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select single report",
            "",
            "Reports (*.pdf *.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)",
        )
        if path:
            self.single_file_edit.setText(path)
            if not self.input_dir_edit.text().strip():
                try:
                    self.input_dir_edit.setText(str(Path(path).expanduser().resolve().parent))
                except Exception:
                    pass

    def _pick_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self.output_dir_edit.setText(path)

    def _pick_tracker_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Tracker CSV", "", "CSV Files (*.csv)")
        if path:
            self.tracker_edit.setText(path)

    def _set_logo(self, path: str) -> None:
        pix = QPixmap(path)
        if pix.isNull():
            return
        self.logo_label.setPixmap(
            pix.scaledToHeight(40, Qt.SmoothTransformation)
        )

    def _try_autoload_logo(self) -> None:
        # Auto-load repo logo when present.
        candidates = _asset_candidates("force-logo.png", "force-log.png")
        for path in candidates:
            if path.exists():
                self._set_logo(str(path))
                break

    def _resolve_tracker_path_from_inputs(self) -> str:
        explicit = self.tracker_edit.text().strip()
        if explicit:
            return str(Path(explicit).expanduser().resolve())
        input_dir = self.input_dir_edit.text().strip()
        if not input_dir:
            return ""
        # Default tracker location is alongside the source report folder.
        return str(Path(input_dir).expanduser().resolve() / "redaction_tracker.csv")

    def _next_tracker_version_path(self, tracker_path: Path) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = tracker_path.with_name(f"{tracker_path.stem}_{ts}{tracker_path.suffix}")
        i = 1
        while candidate.exists():
            candidate = tracker_path.with_name(f"{tracker_path.stem}_{ts}_{i}{tracker_path.suffix}")
            i += 1
        return candidate

    def _get_approved_source_set(self, tracker_csv: str) -> set[str]:
        approved_sources: set[str] = set()
        try:
            df = load_tracker(tracker_csv)
        except Exception:
            return approved_sources
        if "approved_to_send" not in df.columns or "source_file" not in df.columns:
            return approved_sources
        mask = pd.to_numeric(df["approved_to_send"], errors="coerce").fillna(0).eq(1)
        for raw in df.loc[mask, "source_file"].fillna("").astype(str):
            try:
                approved_sources.add(str(Path(raw).expanduser().resolve()))
            except Exception:
                continue
        return approved_sources

    def _apply_default_modality_to_tracker(self, tracker_csv: str, source_files: Sequence[str]) -> None:
        modality = self.home_default_modality_combo.currentText().strip() if hasattr(self, "home_default_modality_combo") else ""
        if not modality:
            return
        if not tracker_csv or not Path(tracker_csv).exists():
            return
        df = load_tracker(tracker_csv)
        if "source_file" not in df.columns:
            return
        file_set: set[str] = set()
        for raw in source_files:
            try:
                file_set.add(str(Path(str(raw)).expanduser().resolve()))
            except Exception:
                continue
        if not file_set:
            return
        mask_files = df["source_file"].fillna("").astype(str).apply(
            lambda x: str(Path(x).expanduser().resolve()) in file_set
        )
        if "modality_type" not in df.columns:
            df["modality_type"] = ""
        blank_mask = df["modality_type"].fillna("").astype(str).str.strip().eq("")
        mask = mask_files & blank_mask
        if not bool(mask.any()):
            return
        df.loc[mask, "modality_type"] = modality
        save_tracker(df, tracker_csv)

    def _find_free_text_name_spans(self, text: str) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        if not text:
            return spans

        label_patterns = [
            re.compile(
                r"(?im)\b(?:patient\s*name|name)\s*[:\-]?\s*"
                r"(?P<name>[A-Z][A-Za-z'`-]+(?:,\s*[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+)*)"
                r"|[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+){1,3})"
            ),
        ]
        for pat in label_patterns:
            for m in pat.finditer(text):
                s = int(m.start("name"))
                e = int(m.end("name"))
                if e > s:
                    spans.append((s, e))

        first = self.first_name_edit.text().strip() if hasattr(self, "first_name_edit") else ""
        last = self.last_name_edit.text().strip() if hasattr(self, "last_name_edit") else ""
        full_name = " ".join([p for p in [first, last] if p]).strip()
        if full_name and len(full_name) >= 3:
            for m in re.finditer(re.escape(full_name), text, flags=re.IGNORECASE):
                spans.append((int(m.start()), int(m.end())))
        return spans

    def _find_free_text_day_spans(self, text: str) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        if not text:
            return spans
        for m in re.finditer(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", text):
            spans.append((int(m.start(3)), int(m.end(3))))
        for m in re.finditer(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})\b", text):
            spans.append((int(m.start(2)), int(m.end(2))))
        for m in re.finditer(r"\b([A-Za-z]{3,9})[\s,./:-]*(\d{1,2})[\s,./:-]*,?[\s,./:-]*(\d{4})\b", text):
            spans.append((int(m.start(2)), int(m.end(2))))
        for m in re.finditer(r"\b(\d{1,2})-([A-Za-z]{3,9})-(\d{4})\b", text):
            spans.append((int(m.start(1)), int(m.end(1))))
        return spans

    def _find_free_text_mrn_spans(self, text: str) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        if not text:
            return spans

        label_patterns = [
            re.compile(
                r"(?im)\b(?:mrn|medical\s*record\s*number|hospital\s*record\s*(?:number|#)?|record\s*#)\b"
                r"\s*[:#-]?\s*(?P<val>[A-Za-z0-9][A-Za-z0-9\-]{3,24})"
            ),
        ]
        for pat in label_patterns:
            for m in pat.finditer(text):
                s = int(m.start("val"))
                e = int(m.end("val"))
                if e > s:
                    spans.append((s, e))

        mrn_value = self.mrn_edit.text().strip() if hasattr(self, "mrn_edit") else ""
        if mrn_value and len(mrn_value) >= 4:
            for m in re.finditer(re.escape(mrn_value), text, flags=re.IGNORECASE):
                spans.append((int(m.start()), int(m.end())))
        return spans

    def _find_free_text_full_date_spans(self, text: str) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        if not text:
            return spans
        pats = [
            r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
            r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
            r"\b[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}\b",
            r"\b\d{1,2}-[A-Za-z]{3,9}-\d{2,4}\b",
        ]
        for pat in pats:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                spans.append((int(m.start()), int(m.end())))
        return spans

    def _compute_age_at_study(self) -> str:
        dob = self._normalize_to_first_of_month(self.dob_edit.text().strip() if hasattr(self, "dob_edit") else "")
        study = self._normalize_to_first_of_month(
            self.study_date_edit.text().strip() if hasattr(self, "study_date_edit") else ""
        )
        if not dob or not study:
            return ""
        tdob = pd.to_datetime(dob, errors="coerce")
        tstudy = pd.to_datetime(study, errors="coerce")
        if pd.isna(tdob) or pd.isna(tstudy):
            return ""
        return str(max(0, int((tstudy - tdob).days)))

    def _compute_age_at_event_from_row(self, row: pd.Series) -> str:
        """
        Prefer raw date fields so metadata age matches report overlays, which use
        true event-day values rather than month-normalized dates.
        """
        dob_raw = self._clean_tracker_value(row.get("raw_dob", "")) or self._clean_tracker_value(row.get("dob", ""))
        study_raw = self._clean_tracker_value(row.get("raw_study_date", "")) or self._clean_tracker_value(row.get("study_date", ""))
        if not dob_raw or not study_raw:
            return ""
        tdob = pd.to_datetime(dob_raw, errors="coerce")
        tstudy = pd.to_datetime(study_raw, errors="coerce")
        if pd.isna(tdob) or pd.isna(tstudy):
            return ""
        return str(max(0, int((tstudy - tdob).days)))

    def _refresh_age_at_event(self) -> None:
        if hasattr(self, "age_at_event_edit"):
            auto_value = self._compute_age_at_study()
            current = self.age_at_event_edit.text().strip()
            # Keep field editable: only overwrite when blank or still at prior auto value.
            if (not current) or (current == self._last_auto_age_at_event):
                self.age_at_event_edit.setText(auto_value)
            self._last_auto_age_at_event = auto_value

    def _sync_age_to_tracker_current_doc(self) -> None:
        if not self.tracker_path or not self.doc_id or not hasattr(self, "age_at_event_edit"):
            return
        try:
            raw_age = self.age_at_event_edit.text().strip()
            normalized_age = raw_age
            if raw_age:
                try:
                    text = raw_age.strip().lower()
                    if text.endswith(("d", "day", "days")):
                        text = re.sub(r"[^\d.]+$", "", text).strip()
                    num = max(0.0, float(text))
                    if "." in text:
                        # Legacy decimal-year inputs should convert to days, but
                        # day values serialized as "3866.0" must remain days.
                        if num <= 130.0:
                            normalized_age = str(int(round(num * 365.25)))
                        else:
                            normalized_age = str(int(round(num)))
                    else:
                        normalized_age = str(max(0, int(round(num))))
                except Exception:
                    normalized_age = ""
            df = load_tracker(self.tracker_path)
            mask = df["doc_id"].astype(str) == str(self.doc_id)
            if not bool(mask.any()):
                return
            df.loc[mask, "age_at_event"] = normalized_age
            save_tracker(df, self.tracker_path)
        except Exception:
            return

    def _update_eu_metadata_ui(self) -> None:
        is_eu = self._is_eu_mode()
        if hasattr(self, "study_date_label"):
            self.study_date_label.setVisible(not is_eu)
        if hasattr(self, "study_date_edit"):
            self.study_date_edit.setVisible(not is_eu)
        # Age-at-event is available for all modes; EU mode relies on this as the only outbound date-derived field.
        if hasattr(self, "age_at_event_label"):
            self.age_at_event_label.setVisible(True)
        if hasattr(self, "age_at_event_edit"):
            self.age_at_event_edit.setVisible(True)
        self._refresh_age_at_event()

    def _merge_free_text_spans(
        self,
        text: str,
        name_spans: Sequence[Tuple[int, int]],
        mrn_spans: Sequence[Tuple[int, int]],
        day_spans: Sequence[Tuple[int, int]],
    ) -> List[dict]:
        candidates: List[dict] = []
        for s, e in name_spans:
            if e <= s:
                continue
            candidates.append(
                {
                    "start": int(s),
                    "end": int(e),
                    "replacement": "##########",
                    "original": text[s:e],
                    "kind": "name",
                    "priority": 0,
                }
            )
        for s, e in mrn_spans:
            if e <= s:
                continue
            candidates.append(
                {
                    "start": int(s),
                    "end": int(e),
                    "replacement": "##########",
                    "original": text[s:e],
                    "kind": "mrn",
                    "priority": 0,
                }
            )
        for s, e in day_spans:
            if e <= s:
                continue
            candidates.append(
                {
                    "start": int(s),
                    "end": int(e),
                    "replacement": "xx",
                    "original": text[s:e],
                    "kind": "day",
                    "priority": 1,
                }
            )
        candidates.sort(key=lambda x: (x["start"], x["priority"], -(x["end"] - x["start"])))

        merged: List[dict] = []
        last_end = -1
        for c in candidates:
            if int(c["start"]) < last_end:
                continue
            merged.append(c)
            last_end = int(c["end"])
        return merged

    def _build_free_text_redacted_html(self, text: str, spans: Sequence[dict]) -> str:
        out: List[str] = []
        token_map: dict[str, str] = {}
        cursor = 0
        token_i = 0
        for span in spans:
            s = int(span["start"])
            e = int(span["end"])
            repl = str(span["replacement"])
            orig = str(span["original"])
            if s > cursor:
                out.append(html.escape(text[cursor:s]))
            token = f"t{token_i}"
            token_map[token] = orig
            out.append(
                f'<a href="token:{token}" title="{html.escape(orig)}" '
                f'style="color:#f5f5f5; text-decoration: underline dotted #d71920;">'
                f"{html.escape(repl)}</a>"
            )
            cursor = e
            token_i += 1
        if cursor < len(text):
            out.append(html.escape(text[cursor:]))
        self._free_text_token_map = token_map
        return "<pre style=\"white-space: pre-wrap; font-family: Consolas, 'Courier New', monospace;\">" + "".join(out) + "</pre>"

    def _run_free_text_redaction(self) -> None:
        if not hasattr(self, "free_text_input") or not hasattr(self, "free_text_output"):
            return
        raw_text = self.free_text_input.toPlainText()
        if not raw_text.strip():
            QMessageBox.information(self, "Free text", "Paste text first, then run redaction.")
            return
        name_spans = self._find_free_text_name_spans(raw_text)
        mrn_spans = self._find_free_text_mrn_spans(raw_text)
        eu_mode = self._is_eu_mode()
        if eu_mode:
            full_date_spans = self._find_free_text_full_date_spans(raw_text)
            spans = self._merge_free_text_spans(
                raw_text,
                name_spans=name_spans,
                mrn_spans=mrn_spans,
                day_spans=[],
            )
            age_val = self._compute_age_at_study()
            date_repl = f"age:{age_val}" if age_val else "xx"
            for s, e in full_date_spans:
                spans.append(
                    {
                        "start": int(s),
                        "end": int(e),
                        "replacement": date_repl,
                        "original": raw_text[s:e],
                        "kind": "date",
                        "priority": 1,
                    }
                )
            spans.sort(key=lambda x: (int(x["start"]), int(x["priority"]), -(int(x["end"]) - int(x["start"]))))
            merged: List[dict] = []
            last_end = -1
            for c in spans:
                if int(c["start"]) < last_end:
                    continue
                merged.append(c)
                last_end = int(c["end"])
            spans = merged
            day_spans = full_date_spans
        else:
            day_spans = self._find_free_text_day_spans(raw_text)
            spans = self._merge_free_text_spans(
                raw_text,
                name_spans=name_spans,
                mrn_spans=mrn_spans,
                day_spans=day_spans,
            )
        html_text = self._build_free_text_redacted_html(raw_text, spans)
        self.free_text_output.setHtml(html_text)
        self.free_text_hover_value.setText("Hovered Original Value: -")
        self.home_log.append(
            f"Free text redaction complete: {len(name_spans)} name span(s), {len(mrn_spans)} MRN span(s), "
            f"{len(day_spans)} {'date' if eu_mode else 'day'} span(s)."
        )

    def _on_free_text_token_hovered(self, link: str) -> None:
        token = str(link or "")
        if token.startswith("token:"):
            key = token.split(":", 1)[1]
            original = self._free_text_token_map.get(key, "")
            if original:
                self.free_text_hover_value.setText(f"Hovered Original Value: {original}")
                return
        self.free_text_hover_value.setText("Hovered Original Value: -")

    def _reset_free_text_metadata(self) -> None:
        fields = [
            getattr(self, "force_id_edit", None),
            getattr(self, "file_id_edit", None),
            getattr(self, "study_date_edit", None),
            getattr(self, "first_name_edit", None),
            getattr(self, "last_name_edit", None),
            getattr(self, "mrn_edit", None),
            getattr(self, "dob_edit", None),
            getattr(self, "age_at_event_edit", None),
        ]
        for widget in fields:
            if widget is None:
                continue
            prev = widget.blockSignals(True)
            widget.setText("")
            widget.blockSignals(prev)

        if hasattr(self, "instance_spin"):
            prev = self.instance_spin.blockSignals(True)
            self.instance_spin.setValue(1)
            self.instance_spin.blockSignals(prev)
        if hasattr(self, "modality_combo"):
            prev = self.modality_combo.blockSignals(True)
            self.modality_combo.setCurrentIndex(0)
            self.modality_combo.blockSignals(prev)
        if hasattr(self, "gender_edit"):
            prev = self.gender_edit.blockSignals(True)
            self.gender_edit.setCurrentText("")
            self.gender_edit.blockSignals(prev)

        self._free_text_token_map = {}
        if hasattr(self, "free_text_hover_value"):
            self.free_text_hover_value.setText("Hovered Original Value: -")

    def _update_manual_input_mode(self) -> None:
        input_mode = (
            self.home_input_type_combo.currentText().strip().lower()
            if hasattr(self, "home_input_type_combo")
            else "report"
        )
        is_free = input_mode == "free text"
        if hasattr(self, "manual_free_text_section"):
            self.manual_free_text_section.setVisible(is_free)
        if hasattr(self, "manual_page_controls"):
            self.manual_page_controls.setVisible(not is_free)
        if hasattr(self, "view"):
            self.view.setVisible(not is_free)
        if hasattr(self, "manual_pending_label"):
            self.manual_pending_label.setVisible(not is_free)
        if hasattr(self, "pending_doc_combo"):
            self.pending_doc_combo.setVisible(not is_free)
        if hasattr(self, "manual_reviewed_label"):
            self.manual_reviewed_label.setVisible(not is_free)
        if hasattr(self, "reviewed_doc_combo"):
            self.reviewed_doc_combo.setVisible(not is_free)
        if hasattr(self, "manual_right_panel"):
            self.manual_right_panel.setVisible(not is_free)

    def run_redaction_next(self) -> None:
        if not self.user_name_edit.text().strip():
            QMessageBox.warning(self, "Required", "Please enter reviewer name on Home tab.")
            return
        if not self._get_selected_site_id().strip():
            QMessageBox.warning(self, "Required", "Please enter Site ID on Home tab.")
            return
        input_mode = (
            self.home_input_type_combo.currentText().strip().lower()
            if hasattr(self, "home_input_type_combo")
            else "report"
        )
        if input_mode == "free text":
            self.tabs.setCurrentIndex(1)
            self._update_manual_input_mode()
            return
        self._sync_identity_to_send()
        input_dir = self.input_dir_edit.text().strip()
        single_file_raw = self.single_file_edit.text().strip() if hasattr(self, "single_file_edit") else ""

        if self.process_worker is not None and self.process_worker.isRunning():
            self.home_log.append("Redaction is already running. Please wait.")
            return

        # Process one file or whole folder in one run (max 1,000 files).
        if input_dir or single_file_raw:
            try:
                selected_files, input_dir = self._collect_non_free_text_sources(
                    input_dir=input_dir,
                    single_file_raw=single_file_raw,
                )

                output_dir = self.output_dir_edit.text().strip() or None
                _output_path, default_tracker_path = self._resolve_output_tracker_paths()
                tracker_input = self.tracker_edit.text().strip()
                tracker_path = (
                    Path(tracker_input).expanduser().resolve()
                    if tracker_input
                    else default_tracker_path
                )
                append_to_tracker = False

                if tracker_path.exists():
                    prompt = QMessageBox(self)
                    prompt.setIcon(QMessageBox.Question)
                    prompt.setWindowTitle("Existing tracker found")
                    prompt.setText(
                        f"Existing tracker found:\n{tracker_path}\n\n"
                        "Reuse it and skip reports already marked OK to Send?"
                    )
                    prompt.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
                    prompt.setDefaultButton(QMessageBox.Yes)
                    choice = prompt.exec()
                    if choice == QMessageBox.Cancel:
                        return
                    if choice == QMessageBox.Yes:
                        append_to_tracker = True
                        approved_sources = self._get_approved_source_set(str(tracker_path))
                        before = len(selected_files)
                        selected_files = [
                            p for p in selected_files if str(p.expanduser().resolve()) not in approved_sources
                        ]
                        skipped = before - len(selected_files)
                        if skipped > 0:
                            self.home_log.append(
                                f"Skipped {skipped} previously approved report(s) based on existing tracker."
                            )
                        if not selected_files:
                            self.tracker_path = str(tracker_path)
                            self.tracker_edit.setText(self.tracker_path)
                            self.open_tracker()
                            self.home_log.append("No new reports to process. Loaded existing tracker.")
                            self.tabs.setCurrentIndex(1)
                            return
                    else:
                        tracker_path = self._next_tracker_version_path(tracker_path)
                        self.home_log.append(f"Creating new tracker file: {tracker_path.name}")

                max_files = 1000
                if len(selected_files) > max_files:
                    selected_files = selected_files[:max_files]
                self.tracker_path = str(tracker_path)
                self.tracker_edit.setText(self.tracker_path)
                self._current_run_tracker_path = self.tracker_path
                self._current_run_append_to_tracker = append_to_tracker
                self.home_progress_label.setText(
                    f"Processing {len(selected_files)} report(s). This can take a few minutes..."
                )
                self.home_progress.setVisible(True)
                self.home_progress.setRange(0, len(selected_files))
                self.home_progress.setValue(0)
                self.start_btn.setEnabled(False)
                self.batch_state_label.setText("Processing in progress...")
                self.home_log.append(f"Started processing {len(selected_files)} report(s).")

                self.process_worker = ProcessReportsWorker(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    tracker_csv_path=self.tracker_path,
                    append_to_tracker=append_to_tracker,
                    overwrite=bool(self.overwrite_cache_chk.isChecked()),
                    detector_backend="nlp",
                    ocr_backend="glmocr",
                    pdf_text_mode="hybrid_pdf_text",
                    device="cpu",
                    source_files=[str(p) for p in selected_files],
                    eu_mode=self._is_eu_mode(),
                    full_date_overlay_mode=self._is_full_date_overlay_mode(),
                )
                self.process_worker.process_progress.connect(self._on_process_progress)
                self.process_worker.process_done.connect(self._on_process_done)
                self.process_worker.process_error.connect(self._on_process_error)
                self.process_worker.finished.connect(self._on_process_finished)
                self.process_worker.start()
                self.tabs.setCurrentIndex(0)  # Stay on Home while running.
            except Exception as exc:
                self.home_log.append(f"Processing setup error: {exc}")
                QMessageBox.warning(self, "Processing setup error", str(exc))
            return

        # Otherwise, fallback to loading an existing tracker.
        tracker_path = self._resolve_tracker_path_from_inputs()
        if tracker_path and Path(tracker_path).exists():
            self.tracker_path = tracker_path
            self.open_tracker()
            self.tabs.setCurrentIndex(1)  # 2) Manual Edits
            return

        QMessageBox.information(
            self,
            "Missing input",
            "Enter a report folder or single report file to run redaction, or provide an existing tracker CSV.",
        )

    def _on_process_done(self, df: object, source_files: object) -> None:
        try:
            _ = df
            files = list(source_files) if source_files is not None else []
            if self._current_run_tracker_path:
                self._apply_default_modality_to_tracker(self._current_run_tracker_path, files)
            self.open_tracker()
            total_docs = self.pending_doc_combo.count() + self.reviewed_doc_combo.count()
            if total_docs == 0:
                self.home_log.append(
                    f"No documents available in tracker: {self.tracker_path}. "
                    "Check Home Log/processing errors."
                )
                self.batch_state_label.setText("Processed, but no renderable docs found.")
                return
            self.home_log.append(f"Processing complete: {len(files)} file(s).")
            self.home_log.append(f"Tracker loaded with {total_docs} document option(s).")
            self.batch_state_label.setText("Processing complete.")
            # After processing finishes, move directly to Manual Edits.
            self.tabs.setCurrentIndex(1)
        except Exception as exc:
            self.home_log.append(f"Processing completion error: {exc}")

    def _on_process_progress(self, done: int, total: int, filename: str, elapsed_sec: float) -> None:
        total = max(1, int(total))
        done = max(0, min(int(done), total))
        self.home_progress.setVisible(True)
        self.home_progress.setRange(0, total)
        self.home_progress.setValue(done)
        eta_text = ""
        if done > 0 and done < total and elapsed_sec > 0:
            avg = elapsed_sec / done
            eta = int(round(avg * (total - done)))
            m, s = divmod(max(0, eta), 60)
            eta_text = f" | ETA {m:02d}:{s:02d}"
        self.home_progress_label.setText(
            f"Processed {done}/{total} reports | Current: {filename}{eta_text}"
        )

    def _on_process_error(self, message: str) -> None:
        self.home_log.append(f"Processing error: {message}")
        self.batch_state_label.setText("Processing failed.")

    def _on_process_finished(self) -> None:
        self.home_progress.setRange(0, 1)
        self.home_progress.setValue(1)
        self.home_progress.setVisible(False)
        self.home_progress_label.setText("")
        self.start_btn.setEnabled(True)
        self.process_worker = None

    def _resolve_output_tracker_paths(self) -> tuple[Path, Path]:
        input_dir = self.input_dir_edit.text().strip()
        if not input_dir:
            raise ValueError("Enter report folder first.")
        output_dir = self.output_dir_edit.text().strip() or ""
        if output_dir:
            output_path = Path(output_dir).expanduser().resolve()
        else:
            output_path = Path(input_dir).expanduser().resolve() / "redaction_output"
        # Tracker lives in report folder root by default (more discoverable for users).
        tracker_path = Path(input_dir).expanduser().resolve() / "redaction_tracker.csv"
        return output_path, tracker_path

    def _batch_signature_now(self) -> tuple[str, str, bool]:
        input_dir = str(Path(self.input_dir_edit.text().strip()).expanduser().resolve())
        output_path, _ = self._resolve_output_tracker_paths()
        overwrite = bool(self.overwrite_cache_chk.isChecked())
        return (input_dir, str(output_path), overwrite)

    def _reset_batch_state(self) -> None:
        self.batch_files_all = []
        self.batch_cursor = 0
        self.batch_sections = []
        self.batch_signature = None
        while self.batch_toolbox.count() > 0:
            self.batch_toolbox.removeItem(0)
        self.batch_state_label.setText("No batch started.")

    def _prepare_batch_queue_if_needed(self) -> None:
        sig = self._batch_signature_now()
        batch_size = int(self.batch_size_spin.value())
        self.batch_size = batch_size

        # New run definition: rebuild queue from scratch.
        if self.batch_signature != sig:
            self._reset_batch_state()
            input_path = Path(sig[0])
            files = list_input_files(input_path, recursive=False)
            if not files:
                raise ValueError(f"No supported files found under: {input_path}")
            self.batch_files_all = [str(p) for p in files]
            self.batch_signature = sig
            _, tracker_path = self._resolve_output_tracker_paths()
            if tracker_path.exists():
                tracker_path.unlink()
            self.home_log.append(f"Prepared {len(self.batch_files_all)} file(s) for batch processing.")

        total_batches = max(1, (len(self.batch_files_all) + self.batch_size - 1) // self.batch_size)
        current_batch = min(total_batches, (self.batch_cursor // self.batch_size) + 1)
        self.batch_state_label.setText(f"Batch {current_batch}/{total_batches}")

    def _start_next_batch(self) -> None:
        self._prepare_batch_queue_if_needed()
        if self.batch_cursor >= len(self.batch_files_all):
            self.home_log.append("All batches have already been processed.")
            self.batch_state_label.setText("All batches complete.")
            return

        input_dir = self.input_dir_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip() or None
        start = self.batch_cursor
        end = min(len(self.batch_files_all), start + self.batch_size)
        batch_files = self.batch_files_all[start:end]
        total_batches = max(1, (len(self.batch_files_all) + self.batch_size - 1) // self.batch_size)
        batch_index = (start // self.batch_size) + 1

        self.home_progress_label.setText(
            f"Processing batch {batch_index}/{total_batches} ({len(batch_files)} files). "
            "This can take a few minutes..."
        )
        self.home_progress.setVisible(True)
        self.home_progress.setRange(0, 0)
        self.start_btn.setEnabled(False)
        self.review_batch_btn.setEnabled(False)
        self.batch_state_label.setText(f"Running batch {batch_index}/{total_batches}")
        self.home_log.append(
            f"Started batch {batch_index}/{total_batches} with {len(batch_files)} file(s)."
        )

        self.batch_worker = BatchProcessWorker(
            input_dir=input_dir,
            output_dir=output_dir,
            overwrite=bool(self.overwrite_cache_chk.isChecked()),
            detector_backend="nlp",
            ocr_backend="glmocr",
            pdf_text_mode="hybrid_pdf_text",
            device="cpu",
            batch_files=batch_files,
            batch_index=batch_index,
            total_batches=total_batches,
            eu_mode=self._is_eu_mode(),
            full_date_overlay_mode=self._is_full_date_overlay_mode(),
        )
        self.batch_worker.batch_done.connect(self._on_batch_done)
        self.batch_worker.batch_error.connect(self._on_batch_error)
        self.batch_worker.finished.connect(self._on_batch_worker_finished)
        self.batch_worker.start()

    def _on_batch_done(self, df: object, batch_index: int, total_batches: int, batch_files: object) -> None:
        try:
            _ = df  # tracker is refreshed from disk below
            batch_list = [str(x) for x in list(batch_files)]
            self.batch_cursor += len(batch_list)
            output_path, tracker_path = self._resolve_output_tracker_paths()
            self.tracker_path = str(tracker_path)
            self.tracker_edit.setText(self.tracker_path)
            self.open_tracker()
            self._add_batch_section(batch_index, total_batches, batch_list)
            self.home_log.append(
                f"Batch {batch_index}/{total_batches} complete: {len(batch_list)} file(s)."
            )
            if self.batch_cursor >= len(self.batch_files_all):
                self.batch_state_label.setText("All batches complete.")
            else:
                next_batch = (self.batch_cursor // self.batch_size) + 1
                self.batch_state_label.setText(f"Ready for batch {next_batch}/{total_batches}")
        except Exception as exc:
            self.home_log.append(f"Batch completion error: {exc}")

    def _on_batch_error(self, message: str) -> None:
        self.home_log.append(f"Batch error: {message}")

    def _on_batch_worker_finished(self) -> None:
        total_files = len(self.batch_files_all)
        done_files = min(self.batch_cursor, total_files)
        if total_files > 0:
            self.home_progress.setVisible(True)
            self.home_progress.setRange(0, total_files)
            self.home_progress.setValue(done_files)
            if done_files >= total_files:
                self.home_progress_label.setText(
                    f"Processing complete: {done_files}/{total_files} files."
                )
            else:
                remaining = total_files - done_files
                next_batch = (self.batch_cursor // self.batch_size) + 1
                total_batches = max(1, (total_files + self.batch_size - 1) // self.batch_size)
                self.home_progress_label.setText(
                    f"Batch done. Processed {done_files}/{total_files} files "
                    f"({remaining} remaining). Click Run Redaction for batch {next_batch}/{total_batches}."
                )
        else:
            self.home_progress.setRange(0, 1)
            self.home_progress.setValue(1)
            self.home_progress.setVisible(False)
            self.home_progress_label.setText("")
        self.start_btn.setEnabled(True)
        self.review_batch_btn.setEnabled(True)
        self.batch_worker = None

    def _add_batch_section(self, batch_index: int, total_batches: int, batch_files: List[str]) -> None:
        if not self.tracker_path or not Path(self.tracker_path).exists():
            return
        df = load_tracker(self.tracker_path)
        batch_file_set = {str(Path(p).expanduser().resolve()) for p in batch_files}
        if "source_file" in df.columns:
            mask = df["source_file"].astype(str).apply(lambda x: str(Path(x).expanduser().resolve()) in batch_file_set)
            batch_df = df[mask].copy()
        else:
            batch_df = df.iloc[0:0].copy()

        cols = ["doc_id", "source_filename", "phi_found", "review_status", "approved_to_send"]
        cols = [c for c in cols if c in batch_df.columns]
        table = QTableWidget()
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels(cols)
        table.setRowCount(len(batch_df))
        for r, (_, row) in enumerate(batch_df.iterrows()):
            for c, name in enumerate(cols):
                table.setItem(r, c, QTableWidgetItem(str(row.get(name, ""))))

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(table)
        header = f"Batch {batch_index}/{total_batches} ({len(batch_df)} docs)"
        self.batch_toolbox.addItem(container, header)
        self.batch_toolbox.setCurrentIndex(self.batch_toolbox.count() - 1)

    def _refresh_review_tab(self) -> None:
        if not self.tracker_path:
            self.review_summary.setText("Load tracker to see review summary.")
            return
        df = load_tracker(self.tracker_path)
        if "first_name" in df.columns:
            df = df.sort_values(by=["first_name", "doc_id"], kind="stable", na_position="last")
        total = len(df)
        approved = int((df.get("approved_to_send", 0).fillna(0) == 1).sum()) if "approved_to_send" in df.columns else 0
        phi = int(df.get("phi_found", 0).fillna(0).sum()) if "phi_found" in df.columns else 0
        reviewed = int(df["review_status"].fillna("").astype(str).str.lower().eq("reviewed").sum()) if "review_status" in df.columns else 0
        self.review_summary.setText(f"Total: {total} | PHI flagged: {phi} | Reviewed: {reviewed} | Approved: {approved}")
        cols = [
            "doc_id",
            "input_kind",
            "site_id",
            "file_id",
            "force_id",
            "modality_instance",
            "modality_type",
            "raw_study_date",
            "study_date",
            "first_name",
            "last_name",
            "mrn",
            "raw_dob",
            "dob",
            "age_at_event",
            "gender",
            "dup",
            "review_status",
            "approved_to_send",
            "reviewer",
        ]
        cols = [c for c in cols if c in df.columns]
        self._updating_review_table = True
        self.review_table.setColumnCount(len(cols))
        self.review_table.setHorizontalHeaderLabels(cols)
        self.review_table.setRowCount(total)
        for r, (_, row) in enumerate(df.iterrows()):
            for c, name in enumerate(cols):
                self.review_table.setItem(r, c, QTableWidgetItem(str(row.get(name, ""))))
        self._updating_review_table = False
        self._refresh_send_summary()

    def on_review_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_review_table:
            return
        if not self.tracker_path:
            return
        row_idx = item.row()
        col_idx = item.column()
        header_item = self.review_table.horizontalHeaderItem(col_idx)
        if header_item is None:
            return
        col_name = header_item.text()
        editable_cols = {
            "site_id",
            "file_id",
            "force_id",
            "modality_instance",
            "modality_type",
            "study_date",
            "first_name",
            "last_name",
            "mrn",
            "dob",
            "age_at_event",
            "gender",
            "review_status",
            "approved_to_send",
            "reviewer",
        }
        if col_name not in editable_cols:
            return
        doc_col = None
        for c in range(self.review_table.columnCount()):
            hi = self.review_table.horizontalHeaderItem(c)
            if hi and hi.text() == "doc_id":
                doc_col = c
                break
        if doc_col is None:
            return
        doc_item = self.review_table.item(row_idx, doc_col)
        if doc_item is None:
            return
        doc_id = (doc_item.text() or "").strip()
        if not doc_id:
            return

        df = load_tracker(self.tracker_path)
        mask = df["doc_id"].astype(str) == str(doc_id)
        if not mask.any():
            return
        value = (item.text() or "").strip()
        if col_name in {"approved_to_send"}:
            try:
                value_num = int(float(value or "0"))
            except Exception:
                value_num = 0
            value_num = 1 if value_num == 1 else 0
            df.loc[mask, col_name] = value_num
            if "sent_to_aws" in df.columns:
                # Any explicit approval toggle re-queues for send.
                df.loc[mask, "sent_to_aws"] = 0
            if "sent_to_aws_at_utc" in df.columns:
                df.loc[mask, "sent_to_aws_at_utc"] = ""
            self._updating_review_table = True
            item.setText(str(value_num))
            self._updating_review_table = False
        elif col_name in {"modality_instance"}:
            try:
                inst = int(float(value or "1"))
            except Exception:
                inst = 1
            if inst < 1:
                inst = 1
            df.loc[mask, col_name] = inst
            fid = self._build_file_id(
                str(df.loc[mask, "force_id"].iloc[0]),
                str(df.loc[mask, "study_date"].iloc[0]),
                inst,
            )
            df.loc[mask, "file_id"] = fid
            self._updating_review_table = True
            item.setText(str(inst))
            self._updating_review_table = False
        elif col_name in {"study_date", "dob"} and value:
            ts = pd.to_datetime(value, errors="coerce")
            if pd.isna(ts):
                return
            ts = ts.replace(day=1)
            value = ts.strftime("%Y-%m-%d")
            df.loc[mask, col_name] = value
            raw_col = "raw_study_date" if col_name == "study_date" else "raw_dob"
            if raw_col in df.columns:
                df.loc[mask, raw_col] = value
            if col_name == "study_date":
                inst = int(pd.to_numeric(df.loc[mask, "modality_instance"], errors="coerce").fillna(1).iloc[0])
                fid = self._build_file_id(
                    str(df.loc[mask, "force_id"].iloc[0]),
                    value,
                    inst,
                )
                df.loc[mask, "file_id"] = fid
            self._updating_review_table = True
            item.setText(value)
            self._updating_review_table = False
            # Keep age_at_event synchronized when date fields change in MKF table.
            if "age_at_event" in df.columns:
                dob_raw = str(df.loc[mask, "dob"].iloc[0] if "dob" in df.columns else "")
                study_raw = str(df.loc[mask, "study_date"].iloc[0] if "study_date" in df.columns else "")
                tdob = pd.to_datetime(dob_raw, errors="coerce")
                tstudy = pd.to_datetime(study_raw, errors="coerce")
                age_value = ""
                if not pd.isna(tdob) and not pd.isna(tstudy):
                    age_value = str(max(0, int((tstudy - tdob).days)))
                df.loc[mask, "age_at_event"] = age_value
        else:
            df.loc[mask, col_name] = value

        resend_trigger_cols = {
            "site_id",
            "file_id",
            "force_id",
            "modality_instance",
            "modality_type",
            "study_date",
            "first_name",
            "last_name",
            "mrn",
            "dob",
            "age_at_event",
            "gender",
        }
        if col_name in resend_trigger_cols:
            if "sent_to_aws" in df.columns:
                df.loc[mask, "sent_to_aws"] = 0
            if "sent_to_aws_at_utc" in df.columns:
                df.loc[mask, "sent_to_aws_at_utc"] = ""

        # Keep legacy mirror in sync.
        if col_name == "force_id" and "patient_id" in df.columns:
            df.loc[mask, "patient_id"] = value
            inst = int(pd.to_numeric(df.loc[mask, "modality_instance"], errors="coerce").fillna(1).iloc[0])
            fid = self._build_file_id(
                value,
                str(df.loc[mask, "study_date"].iloc[0]),
                inst,
            )
            df.loc[mask, "file_id"] = fid

        save_tracker(df, self.tracker_path)
        self._refresh_send_summary()

    def collect_approved(self) -> None:
        if not self.tracker_path:
            self.send_log.append("Load tracker first.")
            return
        files = collect_approved_files(self.tracker_path)
        self.send_log.append(f"Collected {len(files)} approved file(s).")

    def send_approved(self) -> None:
        if not self.tracker_path:
            self.send_log.append("Load tracker first.")
            return
        site_id = self._get_selected_site_id().strip()
        if not site_id:
            self.send_log.append("Site ID is required.")
            return
        if not self.send_confirm_chk.isChecked():
            self.send_log.append("Please check the final review confirmation box before sending.")
            return
        presign_api_url, complete_api_url, session_start_api_url, review_event_api_url, _ = self._get_api_config_from_env()
        api_key = f"Bearer {self.cognito_token.id_token}" if self.cognito_token else None
        reviewer = self.send_reviewer_edit.text().strip()
        if not reviewer:
            self.send_log.append("Reviewer Name is required.")
            return
        if not presign_api_url or not complete_api_url:
            self.send_log.append("Missing API base URL configuration.")
            return
        if not api_key:
            self.send_log.append("Sign in with Cognito before sending approved reports.")
            return
        try:
            unsent_count = 0
            try:
                df = load_tracker(self.tracker_path)
                if "approved_to_send" in df.columns:
                    mask = pd.to_numeric(df["approved_to_send"], errors="coerce").fillna(0).eq(1)
                    if "sent_to_aws" in df.columns:
                        mask = mask & pd.to_numeric(df["sent_to_aws"], errors="coerce").fillna(0).ne(1)
                    unsent_count = int(mask.sum())
            except Exception:
                unsent_count = 0

            uploadable_count = self._count_uploadable_approved_cases()
            if unsent_count <= 0 or uploadable_count <= 0:
                self.send_log.append(
                    "No valid approved-unsent cases to upload (missing redacted PDF/OCR pair or already sent). "
                    "Skipping extraction session launch."
                )
                return

            total_steps = max(1, unsent_count * 5)  # presign/pdf/complete/review/manifest-ish
            step_counter = {"n": 0}

            self.send_progress.setVisible(True)
            self.send_progress.setRange(0, total_steps)
            self.send_progress.setValue(0)
            self.send_log.append("Starting upload of approved-unsent reports...")

            def _on_progress(msg: str) -> None:
                step_counter["n"] = min(total_steps, step_counter["n"] + 1)
                self.send_progress.setValue(step_counter["n"])
                self.send_log.append(msg)
                QApplication.processEvents()

            extraction_session_id = None
            self.send_log.append("Using SQS and Lambda extraction fallback.")

            uploaded = upload_approved_files_via_api(
                tracker_csv=self.tracker_path,
                site_id=site_id,
                reviewer=reviewer,
                presign_api_url=presign_api_url,
                complete_api_url=complete_api_url,
                review_event_api_url=review_event_api_url or None,
                api_key=api_key,
                extraction_session_id=extraction_session_id,
                progress_callback=_on_progress,
            )
            if not uploaded:
                self.send_log.append(
                    "No files uploaded. Approved reports may already be marked sent, or missing required files."
                )
            else:
                self.send_log.append(
                    f"Uploaded {len(uploaded)} object(s) via API pre-signed flow."
                )
        except Exception as exc:
            details = self._format_api_exception(exc)
            msg_lower = details.lower()
            if "site_id mismatch" in msg_lower or "site_acronym mismatch" in msg_lower:
                self.send_log.append(f"Upload error: site identity mismatch. {details}")
            elif "not mapped to an active site" in msg_lower or "no site mapping" in msg_lower:
                self.send_log.append(f"Upload error: API key/site mapping issue. {details}")
            elif "usage plan" in msg_lower or "forbidden" in msg_lower or "unauthorized" in msg_lower:
                self.send_log.append(
                    "Upload error: Cognito/API authorization failed. Confirm the user has the correct site group and rebuild the executable from the current source. "
                    + details
                )
            else:
                self.send_log.append(f"Upload error: {details}")
        finally:
            self.send_progress.setValue(self.send_progress.maximum())
            self.send_progress.setVisible(False)

    def test_api_connection(self) -> None:
        presign_api_url, _, _, _, _ = self._get_api_config_from_env()
        api_key = f"Bearer {self.cognito_token.id_token}" if self.cognito_token else None
        site_id = self._get_selected_site_id().strip() or "TEST"
        if not presign_api_url:
            self.send_log.append("Connection test failed: missing API base URL.")
            return
        if not api_key:
            self.send_log.append("Connection test failed: sign in with Cognito first.")
            return
        payload = {
            "site_id": site_id,
            "report_id": "TEST-AAAFFF-1_20260501_1",
            "filename": "healthcheck.pdf",
            "content_type": "application/pdf",
            "expires_seconds": 60,
        }
        self.send_log.append("Connection test uses /uploads/presign only (no extraction session/task launch).")
        try:
            resp = request_presigned_upload_via_api(
                presign_api_url,
                payload,
                api_key=api_key,
            )
            object_key = str(resp.get("object_key", "") or "")
            self.send_log.append(f"Connection test OK. Presign succeeded. object_key={object_key}")
        except Exception as exc:
            details = self._format_api_exception(exc)
            msg_lower = details.lower()
            if "site_id mismatch" in msg_lower or "site_acronym mismatch" in msg_lower:
                self.send_log.append(f"Connection test failed: site_id/site_acronym does not match API key mapping. {details}")
            elif "not mapped to an active site" in msg_lower or "no site mapping" in msg_lower:
                self.send_log.append(f"Connection test failed: API key exists but is not mapped to an active site. {details}")
            elif "usage plan" in msg_lower or "forbidden" in msg_lower or "unauthorized" in msg_lower:
                self.send_log.append(f"Connection test failed: Cognito/API authorization failed. {details}")
            else:
                self.send_log.append(f"Connection test failed: {details}")

    def _active_doc_combo(self) -> QComboBox:
        if self.pending_doc_combo.currentIndex() >= 0 and self.pending_doc_combo.count() > 0:
            return self.pending_doc_combo
        return self.reviewed_doc_combo

    def _select_doc_by_id(self, doc_id: str) -> bool:
        if not doc_id:
            return False
        idx = self.pending_doc_combo.findData(doc_id)
        if idx >= 0:
            self.pending_doc_combo.setCurrentIndex(idx)
            return True
        idx = self.reviewed_doc_combo.findData(doc_id)
        if idx >= 0:
            self.reviewed_doc_combo.setCurrentIndex(idx)
            return True
        return False

    def open_tracker(self) -> None:
        path = self.tracker_path
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open Tracker CSV", "", "CSV Files (*.csv)")
            if not path:
                return
            self.tracker_path = path
        if not Path(path).exists():
            self.status.setText(f"Tracker not found: {path}")
            self.home_log.append(f"Tracker not found: {path}")
            return
        df = load_tracker(path)
        if "approved_to_send" in df.columns:
            df["_approved_num"] = pd.to_numeric(df["approved_to_send"], errors="coerce").fillna(0).astype(int)
        else:
            df["_approved_num"] = 0
        if "review_status" in df.columns:
            df["_reviewed"] = df["review_status"].fillna("").astype(str).str.lower().eq("reviewed")
        else:
            df["_reviewed"] = False
        # priority: pending first, reviewed-not-approved next, approved last
        df["_status_priority"] = np.where(
            df["_approved_num"].eq(1),
            2,
            np.where(df["_reviewed"], 1, 0),
        )
        sort_cols = [c for c in ["_status_priority", "last_name", "first_name", "study_date", "doc_id"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(by=sort_cols, kind="stable", na_position="last")
        self.home_log.append(f"Opened tracker: {path} (rows={len(df)})")
        self.pending_doc_combo.blockSignals(True)
        self.reviewed_doc_combo.blockSignals(True)
        self.pending_doc_combo.clear()
        self.reviewed_doc_combo.clear()
        self._doc_combo_doc_ids = []
        for _, row in df.iterrows():
            doc_id = str(row.get("doc_id", "") or "").strip()
            if not doc_id:
                continue
            approved_num = int(row.get("_approved_num", 0) or 0)
            reviewed = bool(row.get("_reviewed", False))
            if approved_num == 1:
                status_label = "APPROVED"
            elif reviewed:
                status_label = "REVIEWED"
            else:
                status_label = "PENDING"
            display = f"{doc_id}  [{status_label}]"
            is_pending = status_label == "PENDING"
            combo = self.pending_doc_combo if is_pending else self.reviewed_doc_combo
            combo.addItem(display, doc_id)
            idx = combo.count() - 1
            # Visual cue in dropdown list.
            if status_label == "APPROVED":
                combo.setItemData(idx, QColor(120, 220, 120), Qt.ForegroundRole)
            elif status_label == "REVIEWED":
                combo.setItemData(idx, QColor(255, 220, 120), Qt.ForegroundRole)
            else:
                combo.setItemData(idx, QColor(240, 240, 240), Qt.ForegroundRole)
            self._doc_combo_doc_ids.append(doc_id)
        self.pending_doc_combo.blockSignals(False)
        self.reviewed_doc_combo.blockSignals(False)
        if self.pending_doc_combo.count() > 0 or self.reviewed_doc_combo.count() > 0:
            preferred_doc = self._first_renderable_doc_id(df)
            if preferred_doc:
                if not self._select_doc_by_id(preferred_doc):
                    if self.pending_doc_combo.count() > 0:
                        self.pending_doc_combo.setCurrentIndex(0)
                    else:
                        self.reviewed_doc_combo.setCurrentIndex(0)
            else:
                if self.pending_doc_combo.count() > 0:
                    self.pending_doc_combo.setCurrentIndex(0)
                else:
                    self.reviewed_doc_combo.setCurrentIndex(0)
            self.on_doc_selected()
        else:
            self.status.setText("Tracker loaded, but contains no document rows.")
            self.home_log.append("Tracker loaded, but contains no doc_id rows.")
        self.status.setText(f"Loaded tracker: {path}")
        self._refresh_review_tab()
        self._sync_identity_to_send()

    def _first_renderable_doc_id(self, df: pd.DataFrame) -> str:
        if df.empty or "doc_id" not in df.columns:
            return ""
        for _, row in df.iterrows():
            doc_id = str(row.get("doc_id", "") or "").strip()
            review_dir_raw = str(row.get("review_pages_dir", "") or "").strip()
            if not doc_id or not review_dir_raw:
                continue
            review_dir = Path(review_dir_raw)
            if review_dir.exists() and review_dir.is_dir():
                pngs = sorted(review_dir.glob("*.png"))
                if pngs:
                    return doc_id
        return ""

    def on_pending_doc_selected(self, index: int = -1) -> None:
        if index < 0:
            return
        self.reviewed_doc_combo.blockSignals(True)
        self.reviewed_doc_combo.setCurrentIndex(-1)
        self.reviewed_doc_combo.blockSignals(False)
        self.on_doc_selected(index=index, source="pending")

    def on_reviewed_doc_selected(self, index: int = -1) -> None:
        if index < 0:
            return
        self.pending_doc_combo.blockSignals(True)
        self.pending_doc_combo.setCurrentIndex(-1)
        self.pending_doc_combo.blockSignals(False)
        self.on_doc_selected(index=index, source="reviewed")

    def on_doc_selected(self, index: int = -1, source: str = "") -> None:
        if self.pending_doc_combo.count() == 0 and self.reviewed_doc_combo.count() == 0:
            self.status.setText("No documents loaded in tracker.")
            return
        combo = self.pending_doc_combo if source == "pending" else self.reviewed_doc_combo if source == "reviewed" else self._active_doc_combo()
        current_doc = combo.currentData()
        self.doc_id = str(current_doc or "").strip()
        if not self.doc_id:
            # Backward-compatible fallback if item data is missing.
            txt = str(combo.currentText() or "").strip()
            self.doc_id = txt.split("  [", 1)[0].strip()
        if not self.doc_id:
            self.status.setText("Selected document is empty.")
            return
        df = load_tracker(self.tracker_path)
        row_df = df[df["doc_id"].astype(str) == self.doc_id]
        if row_df.empty:
            self.status.setText(f"Document not found in tracker: {self.doc_id}")
            return
        row = row_df.iloc[-1]
        force_id = self._clean_tracker_value(row.get("force_id", "")) or self._clean_tracker_value(row.get("patient_id", ""))
        if not force_id:
            force_id = "XXX-LLLFFF-1"
        self.force_id_edit.setText(force_id)
        modality = self._clean_tracker_value(row.get("modality_type", ""))
        modality_idx = self.modality_combo.findText(modality) if modality else -1
        if modality_idx < 0:
            modality_idx = self.modality_combo.findText("Other")
        self.modality_combo.setCurrentIndex(max(0, modality_idx))
        study = self._clean_tracker_value(row.get("study_date", ""))
        inferred_study = self._infer_study_date_for_row(row)
        # If legacy tracker rows copied DOB into study_date, auto-correct on load for StressTest.
        if str(modality).strip().lower() == "stresstest":
            row_dob = self._normalize_to_first_of_month(self._clean_tracker_value(row.get("dob", "")))
            if inferred_study and (not study or (row_dob and self._normalize_to_first_of_month(study) == row_dob)):
                study = inferred_study
        elif not study:
            study = inferred_study
        self.study_date_edit.setText(study)
        inst_raw = row.get("modality_instance", 1)
        try:
            inst_val = int(float(inst_raw))
        except Exception:
            inst_val = 1
        if inst_val < 1:
            inst_val = 1
        self.instance_spin.setValue(inst_val)
        self._refresh_file_id_preview()
        inferred = self._infer_patient_demographics_for_row(row)
        first_name = self._clean_tracker_value(row.get("first_name", "")) or inferred.get("first_name", "")
        last_name = self._clean_tracker_value(row.get("last_name", "")) or inferred.get("last_name", "")
        self.first_name_edit.setText(first_name)
        self.last_name_edit.setText(last_name)
        self._auto_generate_force_id()
        mrn = self._clean_tracker_value(row.get("mrn", ""))
        if not mrn:
            mrn = self._infer_mrn_for_row(row)
        self.mrn_edit.setText(mrn)
        dob = self._clean_tracker_value(row.get("dob", "")) or inferred.get("dob", "")
        dob = self._normalize_to_first_of_month(dob)
        self.dob_edit.setText(dob)
        if hasattr(self, "age_at_event_edit"):
            row_age = self._clean_tracker_value(row.get("age_at_event", ""))
            # Recover obviously corrupted day values (e.g., repeated years->days conversions).
            sane_age = row_age
            try:
                parsed = float(row_age) if row_age else float("nan")
                if not pd.isna(parsed) and parsed > (130.0 * 365.25):
                    sane_age = ""
            except Exception:
                pass
            row_based_age = self._compute_age_at_event_from_row(row)
            self.age_at_event_edit.setText(row_based_age or sane_age or self._compute_age_at_study())
        gender = self._clean_tracker_value(row.get("gender", "")) or inferred.get("gender", "")
        g = gender.strip().lower()
        if g in {"m", "male"}:
            self.gender_edit.setCurrentText("M")
        elif g in {"f", "female"}:
            self.gender_edit.setCurrentText("F")
        else:
            self.gender_edit.setCurrentText("")
        review_dir_raw = self._clean_tracker_value(row.get("review_pages_dir", ""))
        if review_dir_raw:
            review_dir = Path(review_dir_raw)
        else:
            manifest_raw = self._clean_tracker_value(row.get("manifest_json", ""))
            if manifest_raw:
                review_dir = Path(manifest_raw).parent / "review_pages"
            else:
                redacted_raw = self._clean_tracker_value(row.get("redacted_file", ""))
                if redacted_raw:
                    review_dir = Path(redacted_raw).parent / "review_pages"
                else:
                    review_dir = Path(self.tracker_path).resolve().parent / self.doc_id / "review_pages"
        self.page_paths = sorted(review_dir.glob("*.png")) if review_dir.exists() and review_dir.is_dir() else []
        if not self.page_paths:
            self.scene.clear()
            self.base_boxes = []
            self.base_box_items = []
            self.manual_boxes = []
            self.refresh_lists()
            self.status.setText(
                f"{self.doc_id}: no review page images found at {review_dir}. "
                "This file may have failed processing."
            )
            self.home_log.append(
                f"{self.doc_id}: no review pages found at {review_dir}. "
                "Likely processing error for this file."
            )
            return
        self.page_spin.setMaximum(max(1, len(self.page_paths)))
        self._update_page_indicator()
        self.page_spin.setValue(1)
        self.load_page(1)

    def _on_tab_changed(self, index: int) -> None:
        # If user enters Manual Edits and nothing is loaded, try to refresh automatically.
        if index != 1:
            return
        if not self.tracker_path:
            maybe_tracker = self._resolve_tracker_path_from_inputs()
            if maybe_tracker:
                self.tracker_path = maybe_tracker
        if (self.pending_doc_combo.count() + self.reviewed_doc_combo.count()) == 0 and self.tracker_path:
            self.open_tracker()

    def on_page_changed(self) -> None:
        if self.page_paths:
            self.load_page(int(self.page_spin.value()))

    def next_page(self) -> None:
        if not self.page_paths:
            return
        nxt = min(len(self.page_paths), int(self.page_spin.value()) + 1)
        self.page_spin.setValue(nxt)

    def prev_page(self) -> None:
        if not self.page_paths:
            return
        prev = max(1, int(self.page_spin.value()) - 1)
        self.page_spin.setValue(prev)

    def _snapshot_view_state(self) -> dict:
        return {
            "transform": self.view.transform(),
            "h": self.view.horizontalScrollBar().value(),
            "v": self.view.verticalScrollBar().value(),
            "zoom": self.current_zoom,
        }

    def _restore_view_state(self, state: dict | None) -> None:
        if not state:
            return
        try:
            self.view.setTransform(state["transform"])
            self.view.horizontalScrollBar().setValue(int(state.get("h", 0)))
            self.view.verticalScrollBar().setValue(int(state.get("v", 0)))
            self.current_zoom = float(state.get("zoom", 1.0))
            self._set_zoom_label()
        except Exception:
            self.reset_zoom()

    def load_page(self, page_number: int, preserve_view: bool = False) -> None:
        if not self.page_paths:
            return
        if page_number < 1 or page_number > len(self.page_paths):
            self.status.setText(f"Invalid page index: {page_number}")
            return
        scope = (str(self.doc_id), int(page_number))
        if self._deleted_candidates_scope != scope:
            self._deleted_base_candidates = []
            self._deleted_candidates_scope = scope
        prior_state = self._snapshot_view_state() if preserve_view else None
        self.page_number = page_number
        self._update_page_indicator()
        img_path = self.page_paths[page_number - 1]
        if not img_path.exists():
            self.scene.clear()
            self._pixmap_item = None
            self._overlay_items = []
            self.base_box_items = []
            self.page_text_overlays = []
            self.base_boxes = []
            self.manual_boxes = []
            self.refresh_lists()
            self.status.setText(
                f"{self.doc_id} page {page_number} image is missing: {img_path.name}"
            )
            self.home_log.append(f"Missing review image for {self.doc_id}: {img_path}")
            return
        pix = QPixmap(str(img_path))
        if pix.isNull():
            self.scene.clear()
            self._pixmap_item = None
            self._overlay_items = []
            self.base_box_items = []
            self.page_text_overlays = []
            self.base_boxes = []
            self.manual_boxes = []
            self.refresh_lists()
            self.status.setText(
                f"{self.doc_id} page {page_number} could not be rendered (invalid image: {img_path.name})."
            )
            self.home_log.append(
                f"Warning: failed to render review image {img_path} (null QPixmap)."
            )
            return
        self.scene.clear()
        self._pixmap_item = self.scene.addPixmap(pix)
        self.scene.setSceneRect(QRectF(pix.rect()))
        self._overlay_items = []
        self.base_box_items = get_page_redaction_items(self.tracker_path, self.doc_id, page_number)
        self.page_text_overlays = get_page_text_overlays(self.tracker_path, self.doc_id, page_number)
        self.base_boxes = [tuple(item["bbox_xyxy"]) for item in self.base_box_items]
        self.manual_boxes = []
        self.selected_base_row = -1
        self.selected_manual_row = -1
        self.resize_mode_kind = None
        self.resize_mode_idx = -1
        self.view.clear_resize_mode()
        if not preserve_view:
            self.undo_stack = []
        self.refresh_lists()
        self.render_overlays()
        if preserve_view:
            self._restore_view_state(prior_state)
        else:
            self.reset_zoom()
        if hasattr(self, "inscription_edit"):
            self.inscription_edit.clear()
        self.status.setText(f"{self.doc_id} page {page_number}/{max(1, len(self.page_paths))} loaded.")

    def _update_page_indicator(self) -> None:
        total = max(1, len(self.page_paths))
        current = int(self.page_spin.value()) if hasattr(self, "page_spin") else int(self.page_number)
        current = max(1, min(total, current))
        if hasattr(self, "page_indicator_label"):
            self.page_indicator_label.setText(f"{current}/{total}")
        try:
            self.page_spin.setMaximum(total)
        except Exception:
            pass

    def _set_zoom_label(self) -> None:
        self.zoom_label.setText(f"Zoom: {int(round(self.current_zoom * 100))}%")

    def reset_zoom(self) -> None:
        self.current_zoom = DEFAULT_PAGE_ZOOM
        self.view.resetTransform()
        self.view.scale(DEFAULT_PAGE_ZOOM, DEFAULT_PAGE_ZOOM)
        self._set_zoom_label()

    def zoom_in(self) -> None:
        self.current_zoom *= 1.2
        self.view.scale(1.2, 1.2)
        self._set_zoom_label()

    def zoom_out(self) -> None:
        self.current_zoom /= 1.2
        self.view.scale(1 / 1.2, 1 / 1.2)
        self._set_zoom_label()

    def fit_to_view(self) -> None:
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        # Approximate zoom label after fit.
        self.current_zoom = 1.0
        self._set_zoom_label()

    def on_box_drawn(self, box: Box) -> None:
        self._push_undo_state(persisted=False)
        if self._deleted_base_candidates:
            bx1, by1, bx2, by2 = box
            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0
            best_i = -1
            best_d = None
            for i, cand in enumerate(self._deleted_base_candidates):
                cb = cand.get("bbox_xyxy", [0, 0, 0, 0])
                if not isinstance(cb, (list, tuple)) or len(cb) != 4:
                    continue
                ccx = (float(cb[0]) + float(cb[2])) / 2.0
                ccy = (float(cb[1]) + float(cb[3])) / 2.0
                d = abs(ccx - cx) + abs(ccy - cy)
                if best_d is None or d < best_d:
                    best_d = d
                    best_i = i
            if best_i >= 0 and best_d is not None and best_d <= 220:
                inherited = dict(self._deleted_base_candidates.pop(best_i))
                inherited["bbox_xyxy"] = [int(bx1), int(by1), int(bx2), int(by2)]
                self.base_boxes.append((int(bx1), int(by1), int(bx2), int(by2)))
                self.base_box_items.append(inherited)
                self.selected_base_row = len(self.base_boxes) - 1
                self.refresh_lists()
                self.base_list.setCurrentRow(self.selected_base_row)
                self.render_overlays()
                return
        self.manual_boxes.append(box)
        self.refresh_lists()
        self.render_overlays()

    def _capture_edit_state(self, *, persisted: bool) -> dict:
        return {
            "base_boxes": deepcopy(self.base_boxes),
            "base_box_items": deepcopy(self.base_box_items),
            "page_text_overlays": deepcopy(self.page_text_overlays),
            "manual_boxes": deepcopy(self.manual_boxes),
            "selected_base_row": int(self.selected_base_row),
            "selected_manual_row": int(self.selected_manual_row),
            "persisted": bool(persisted),
        }

    def _push_undo_state(self, *, persisted: bool) -> None:
        self.undo_stack.append(self._capture_edit_state(persisted=persisted))
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)

    def _restore_edit_state(self, state: dict) -> None:
        self.base_boxes = deepcopy(state.get("base_boxes", []))
        self.base_box_items = deepcopy(state.get("base_box_items", []))
        self.page_text_overlays = deepcopy(state.get("page_text_overlays", []))
        self.manual_boxes = deepcopy(state.get("manual_boxes", []))
        self.selected_base_row = int(state.get("selected_base_row", -1))
        self.selected_manual_row = int(state.get("selected_manual_row", -1))

    def _persist_current_page_state(self) -> None:
        final_boxes = [*self.base_boxes, *self.manual_boxes]
        override_page_redactions(
            tracker_csv=self.tracker_path,
            doc_id=self.doc_id,
            page_number=self.page_number,
            boxes_xyxy=final_boxes,
            redaction_items=self._all_redaction_items(),
            text_overlays=list(self.page_text_overlays or []),
        )

    def _all_redaction_items(self) -> List[dict]:
        items: List[dict] = []
        for item in self.base_box_items:
            bbox = item.get("bbox_xyxy", [0, 0, 0, 0])
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            items.append(
                {
                    "bbox_xyxy": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    "tag": str(item.get("tag", "") or ""),
                    "text": str(item.get("text", "") or ""),
                }
            )
        for box in self.manual_boxes:
            x1, y1, x2, y2 = box
            items.append(
                {
                    "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                    "tag": "manual_override",
                    "text": "",
                }
            )
        return items

    def on_box_selected(self, kind: str, idx: int) -> None:
        if kind == "base":
            self.selected_manual_row = -1
            self.base_list.setCurrentRow(idx)
            self.selected_base_row = idx
        elif kind == "manual":
            self.selected_base_row = -1
            self.selected_manual_row = idx
        self._sync_inscription_editor_for_selection()
        self.render_overlays()

    def on_resize_mode_changed(self, kind: str, idx: int) -> None:
        if kind and idx >= 0:
            self.resize_mode_kind = str(kind)
            self.resize_mode_idx = int(idx)
            self.status.setText(
                "Resize mode active (blue). Drag edge/corner to resize. Double-click same box to exit."
            )
        else:
            self.resize_mode_kind = None
            self.resize_mode_idx = -1
            self.status.setText("Resize mode off. Double-click any box to enable resize mode.")
        self.render_overlays()

    def on_box_moved(self, kind: str, idx: int, box: Box) -> None:
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return
        self._push_undo_state(persisted=False)
        if kind == "base" and 0 <= idx < len(self.base_boxes):
            old_box = self.base_boxes[idx]
            self.base_boxes[idx] = (x1, y1, x2, y2)
            self._reanchor_overlay_for_moved_bbox(
                [int(old_box[0]), int(old_box[1]), int(old_box[2]), int(old_box[3])],
                [int(x1), int(y1), int(x2), int(y2)],
            )
            if 0 <= idx < len(self.base_box_items):
                self.base_box_items[idx]["bbox_xyxy"] = [int(x1), int(y1), int(x2), int(y2)]
            self.selected_base_row = idx
            self.refresh_lists()
            self.base_list.setCurrentRow(idx)
            self.render_overlays()
            self.status.setText(f"Moved AI/NLP box {idx + 1}. Press Enter to apply page.")
            return
        if kind == "manual" and 0 <= idx < len(self.manual_boxes):
            self.manual_boxes[idx] = (x1, y1, x2, y2)
            self.selected_manual_row = idx
            self.render_overlays()
            self.status.setText(f"Moved manual box {idx + 1}. Press Enter to apply page.")

    def delete_selected_base_box(self) -> None:
        row = self.base_list.currentRow()
        if row < 0 or row >= len(self.base_boxes):
            return
        self._push_undo_state(persisted=True)
        removed = self.base_boxes.pop(row)
        self._remove_overlay_for_bbox([int(removed[0]), int(removed[1]), int(removed[2]), int(removed[3])])
        if 0 <= row < len(self.base_box_items):
            removed_item = dict(self.base_box_items.pop(row))
            removed_item["bbox_xyxy"] = [int(removed[0]), int(removed[1]), int(removed[2]), int(removed[3])]
            self._deleted_base_candidates.append(removed_item)
        self.selected_base_row = -1

        # Persist deletion immediately (no extra "Run Box Redaction" needed).
        final_boxes = [*self.base_boxes, *self.manual_boxes]
        self._persist_current_page_state()
        self.load_page(self.page_number, preserve_view=True)
        self.status.setText(
            f"Deleted selected AI/NLP box and applied {len(final_boxes)} remaining box(es) on page {self.page_number}."
        )

    def on_base_row_changed(self, row: int) -> None:
        self.selected_base_row = int(row)
        self._sync_inscription_editor_for_selection()
        self.render_overlays()

    def _boxes_overlap(self, a: Sequence[int], b: Sequence[int]) -> bool:
        ax1, ay1, ax2, ay2 = [int(v) for v in a]
        bx1, by1, bx2, by2 = [int(v) for v in b]
        return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

    def _find_overlay_index_for_bbox(self, bbox_xyxy: Sequence[int]) -> int:
        best_idx = -1
        best_score = None
        for i, ov in enumerate(list(self.page_text_overlays or [])):
            b = ov.get("bbox_xyxy", [0, 0, 0, 0])
            if not isinstance(b, (list, tuple)) or len(b) != 4:
                continue
            try:
                ob = [int(float(v)) for v in b]
            except Exception:
                continue
            if not self._boxes_overlap(ob, bbox_xyxy):
                continue
            # Prefer nearest center-match overlay.
            cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0
            cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0
            ocx = (ob[0] + ob[2]) / 2.0
            ocy = (ob[1] + ob[3]) / 2.0
            score = abs(cx - ocx) + abs(cy - ocy)
            if best_score is None or score < best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _remove_overlay_for_bbox(self, bbox_xyxy: Sequence[int]) -> None:
        idx = self._find_overlay_index_for_bbox(bbox_xyxy)
        if 0 <= idx < len(self.page_text_overlays):
            self.page_text_overlays.pop(idx)

    def _reanchor_overlay_for_moved_bbox(
        self,
        old_bbox_xyxy: Sequence[int],
        new_bbox_xyxy: Sequence[int],
    ) -> None:
        idx = self._find_overlay_index_for_bbox(old_bbox_xyxy)
        if not (0 <= idx < len(self.page_text_overlays)):
            return
        ov = dict(self.page_text_overlays[idx] or {})
        try:
            old_x1, old_y1, old_x2, old_y2 = [int(float(v)) for v in old_bbox_xyxy]
            new_x1, new_y1, new_x2, new_y2 = [int(float(v)) for v in new_bbox_xyxy]
        except Exception:
            return
        old_cx = int(round((old_x1 + old_x2) / 2.0))
        old_cy = int(round((old_y1 + old_y2) / 2.0))
        new_cx = int(round((new_x1 + new_x2) / 2.0))
        new_cy = int(round((new_y1 + new_y2) / 2.0))
        dx = int(new_cx - old_cx)
        dy = int(new_cy - old_cy)
        try:
            cur_x = int(float(ov.get("x", old_cx) or old_cx))
        except Exception:
            cur_x = old_cx
        try:
            cur_y = int(float(ov.get("y", old_cy) or old_cy))
        except Exception:
            cur_y = old_cy
        ov["x"] = int(max(0, cur_x + dx))
        ov["y"] = int(max(0, cur_y + dy))
        ov["bbox_xyxy"] = [new_x1, new_y1, new_x2, new_y2]
        self.page_text_overlays[idx] = ov

    def _extract_days_from_inscription_text(self, text: str) -> str:
        raw = str(text or "").strip().lower()
        if not raw:
            return ""
        m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(?:d|day|days)?\s*$", raw)
        if not m:
            return ""
        try:
            return str(max(0, int(round(float(m.group(1))))))
        except Exception:
            return ""

    def _sync_inscription_editor_for_selection(self) -> None:
        if not hasattr(self, "inscription_edit"):
            return
        if not (0 <= self.selected_base_row < len(self.base_boxes)):
            self.inscription_edit.clear()
            return
        box = self.base_boxes[int(self.selected_base_row)]
        idx = self._find_overlay_index_for_bbox([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
        if 0 <= idx < len(self.page_text_overlays):
            self.inscription_edit.setText(str(self.page_text_overlays[idx].get("text", "") or ""))
        else:
            self.inscription_edit.clear()

    def set_selected_box_inscription(self) -> None:
        if not (self.tracker_path and self.doc_id and self.page_number >= 1):
            self.status.setText("Load tracker and select a document first.")
            return
        if not (0 <= self.selected_base_row < len(self.base_boxes)):
            self.status.setText("Select an AI/NLP box first.")
            return
        text = self.inscription_edit.text().strip() if hasattr(self, "inscription_edit") else ""
        if not text:
            self.status.setText("Enter inscription text first, or use Delete Inscription.")
            return
        self._push_undo_state(persisted=True)
        box = self.base_boxes[int(self.selected_base_row)]
        bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        cx = int(round((bbox[0] + bbox[2]) / 2.0))
        cy = int(round((bbox[1] + bbox[3]) / 2.0))
        idx = self._find_overlay_index_for_bbox(bbox)
        payload = {
            "text": text,
            "x": int(max(0, cx)),
            "y": int(max(0, cy)),
            "size": 18,
            "color": "#00D7FF",
            "stroke_fill": "black",
            "stroke_width": 2,
            "bbox_xyxy": bbox,
        }
        if 0 <= idx < len(self.page_text_overlays):
            self.page_text_overlays[idx] = payload
        else:
            self.page_text_overlays.append(payload)
        # Keep Required Metadata synchronized when inscription is an age-in-days value.
        days_value = self._extract_days_from_inscription_text(text)
        if days_value and hasattr(self, "age_at_event_edit"):
            self.age_at_event_edit.setText(days_value)
            self._sync_age_to_tracker_current_doc()
        try:
            self._persist_current_page_state()
            self.load_page(self.page_number, preserve_view=True)
            self.status.setText("Updated inscription for selected box.")
        except Exception as exc:
            self.status.setText(f"Inscription update failed: {exc}")

    def clear_selected_box_inscription(self) -> None:
        if not (self.tracker_path and self.doc_id and self.page_number >= 1):
            self.status.setText("Load tracker and select a document first.")
            return
        if not (0 <= self.selected_base_row < len(self.base_boxes)):
            self.status.setText("Select an AI/NLP box first.")
            return
        self._push_undo_state(persisted=True)
        box = self.base_boxes[int(self.selected_base_row)]
        bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        self._remove_overlay_for_bbox(bbox)
        if hasattr(self, "inscription_edit"):
            self.inscription_edit.clear()
        try:
            self._persist_current_page_state()
            self.load_page(self.page_number, preserve_view=True)
            self.status.setText("Deleted inscription for selected box.")
        except Exception as exc:
            self.status.setText(f"Delete inscription failed: {exc}")

    def _nudge_selected_box(self, dx: int, dy: int) -> bool:
        if self.resize_mode_kind is not None and self.resize_mode_idx >= 0:
            self.status.setText("Resize mode active: movement disabled. Double-click blue box to exit resize mode.")
            return False
        # Prefer selected AI/NLP box; otherwise selected manual box.
        if 0 <= self.selected_base_row < len(self.base_boxes):
            idx = int(self.selected_base_row)
            x1, y1, x2, y2 = self.base_boxes[idx]
            nx1, ny1, nx2, ny2 = x1 + dx, y1 + dy, x2 + dx, y2 + dy
            self.base_boxes[idx] = (nx1, ny1, nx2, ny2)
            self._reanchor_overlay_for_moved_bbox(
                [int(x1), int(y1), int(x2), int(y2)],
                [int(nx1), int(ny1), int(nx2), int(ny2)],
            )
            if 0 <= idx < len(self.base_box_items):
                self.base_box_items[idx]["bbox_xyxy"] = [int(nx1), int(ny1), int(nx2), int(ny2)]
            self.refresh_lists()
            self.base_list.setCurrentRow(idx)
            self.render_overlays()
            self.status.setText(f"Moved AI/NLP box {idx + 1} by ({dx}, {dy}).")
            return True
        if 0 <= self.selected_manual_row < len(self.manual_boxes):
            idx = int(self.selected_manual_row)
            x1, y1, x2, y2 = self.manual_boxes[idx]
            nx1, ny1, nx2, ny2 = x1 + dx, y1 + dy, x2 + dx, y2 + dy
            self.manual_boxes[idx] = (nx1, ny1, nx2, ny2)
            self.render_overlays()
            self.status.setText(f"Moved manual box {idx + 1} by ({dx}, {dy}).")
            return True
        return False

    def undo_last(self) -> None:
        if not self.undo_stack:
            self.status.setText("Nothing to undo.")
            return
        state = self.undo_stack.pop()
        persisted = bool(state.get("persisted", False))
        self._restore_edit_state(state)
        if persisted and self.tracker_path and self.doc_id and self.page_number >= 1:
            try:
                self._persist_current_page_state()
                self.load_page(self.page_number, preserve_view=True)
                self.status.setText("Undid last persisted edit.")
                return
            except Exception as exc:
                self.status.setText(f"Undo failed: {exc}")
        self.refresh_lists()
        self.render_overlays()
        self.status.setText("Undid last queued edit.")

    def _on_ctrl_z(self) -> None:
        if self.tabs.currentIndex() == 1:
            self.undo_last()

    def _nudge_from_shortcut(self, dx: int, dy: int) -> None:
        if self.tabs.currentIndex() != 1:
            return
        fw = QApplication.focusWidget()
        if isinstance(fw, (QLineEdit, QTextEdit, QComboBox, QSpinBox)):
            return
        has_target = (
            (0 <= self.selected_base_row < len(self.base_boxes))
            or (0 <= self.selected_manual_row < len(self.manual_boxes))
        )
        if not has_target:
            # In page review mode (no selected box), left/right arrows should navigate pages.
            if dy == 0:
                if dx < 0:
                    self.prev_page()
                elif dx > 0:
                    self.next_page()
            return
        self._push_undo_state(persisted=False)
        self._nudge_selected_box(dx, dy)

    def clear_manual(self) -> None:
        self._push_undo_state(persisted=False)
        self.manual_boxes = []
        self.refresh_lists()
        self.render_overlays()

    def apply_page(self) -> None:
        self._push_undo_state(persisted=True)
        final_boxes = [*self.base_boxes, *self.manual_boxes]
        try:
            self._persist_current_page_state()
            # Reload fresh PNG + canonical boxes after apply.
            self.load_page(self.page_number, preserve_view=True)
            self.status.setText(
                f"Applied {len(final_boxes)} box(es) on page {self.page_number}."
            )
        except Exception as exc:
            self.status.setText(f"Apply failed: {exc}")

    def keyPressEvent(self, event):  # type: ignore[override]
        if self.modality_combo.hasFocus() and event.key() in (Qt.Key_Down, Qt.Key_Up):
            idx = self.modality_combo.currentIndex()
            if event.key() == Qt.Key_Down:
                idx = min(self.modality_combo.count() - 1, idx + 1)
            else:
                idx = max(0, idx - 1)
            self.modality_combo.setCurrentIndex(idx)
            event.accept()
            return
        if self.tabs.currentIndex() == 1 and event.key() in (
            Qt.Key_Left,
            Qt.Key_Right,
            Qt.Key_Up,
            Qt.Key_Down,
        ):
            fw = QApplication.focusWidget()
            if isinstance(fw, (QLineEdit, QTextEdit, QComboBox, QSpinBox)):
                super().keyPressEvent(event)
                return
            has_target = (
                (0 <= self.selected_base_row < len(self.base_boxes))
                or (0 <= self.selected_manual_row < len(self.manual_boxes))
            )
            if not has_target and event.key() in (Qt.Key_Left, Qt.Key_Right):
                if event.key() == Qt.Key_Left:
                    self.prev_page()
                else:
                    self.next_page()
                event.accept()
                return
            step = 5 if bool(event.modifiers() & Qt.ShiftModifier) else 1
            dx = 0
            dy = 0
            if event.key() == Qt.Key_Left:
                dx = -step
            elif event.key() == Qt.Key_Right:
                dx = step
            elif event.key() == Qt.Key_Up:
                dy = -step
            elif event.key() == Qt.Key_Down:
                dy = step
            if dx != 0 or dy != 0:
                if has_target:
                    self._push_undo_state(persisted=False)
                if self._nudge_selected_box(dx, dy):
                    event.accept()
                    return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            # Trigger quick apply from keyboard while on Manual Edits tab.
            if self.tabs.currentIndex() == 1 and (
                self.manual_boxes
                or self.base_boxes
            ):
                self.apply_page()
                event.accept()
                return
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.tabs.currentIndex() == 1:
                if self.base_list.currentRow() >= 0:
                    self.delete_selected_base_box()
                    event.accept()
                    return
                if 0 <= self.selected_manual_row < len(self.manual_boxes):
                    self._push_undo_state(persisted=False)
                    self.manual_boxes.pop(self.selected_manual_row)
                    self.selected_manual_row = -1
                    self.refresh_lists()
                    self.render_overlays()
                    event.accept()
                    return
        super().keyPressEvent(event)

    def _infer_study_date_for_row(self, row) -> str:
        manifest_path = Path(str(row.get("manifest_json", "") or ""))
        if not manifest_path.exists():
            return ""
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            def _parse_study_like_date(value: str) -> str:
                raw = str(value or "").strip()
                if not raw:
                    return ""
                ts = pd.to_datetime(raw, errors="coerce")
                if not pd.isna(ts):
                    ts = ts.replace(day=1)
                    return ts.strftime("%Y-%m-%d")
                # Month-name + year fallback (for OCR text like "-Jan-2004").
                m = re.search(r"\b(?P<mon>[A-Za-z]{3,9})[-/\s,]+(?P<yy>\d{4})\b", raw, flags=re.IGNORECASE)
                if m:
                    mon = str(m.group("mon") or "")
                    yy = str(m.group("yy") or "")
                    ts2 = pd.to_datetime(f"{mon} 01 {yy}", errors="coerce")
                    if not pd.isna(ts2):
                        return ts2.strftime("%Y-%m-%d")
                return ""

            # 1) Prefer explicit study/test-date labels (avoid DOB leakage into study date).
            label_pat = re.compile(
                r"\b(?:date\s*of\s*test|test\s*date|study\s*date|date\s*of\s*study|exam\s*date|service\s*date|performed\s*on)\b",
                flags=re.IGNORECASE,
            )
            date_pat = re.compile(
                r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}-[A-Za-z]{3,9}-\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})\b"
            )
            for page in manifest.get("pages", []):
                page_text = str(page.get("normalized_text_for_export", "") or page.get("text", "") or "")
                if not page_text:
                    continue
                for line in page_text.splitlines():
                    line_pat = label_pat.search(line)
                    if not line_pat:
                        continue
                    # Ignore birth-related lines even if they contain "date".
                    if re.search(r"\b(?:dob|date\s*of\s*birth|birth)\b", line, flags=re.IGNORECASE):
                        continue
                    tail = line[line_pat.end() :]
                    dm = date_pat.search(tail)
                    if not dm:
                        dm = date_pat.search(line)
                    if dm:
                        parsed = _parse_study_like_date(dm.group(0))
                        if parsed:
                            return parsed
                    # Fallback for month-year OCR (for example "-Jan-2004" when day is clipped/redacted).
                    parsed_line = _parse_study_like_date(tail or line)
                    if parsed_line:
                        return parsed_line

            # 2) Fall back to detected generic dates (exclude dob).
            for page in manifest.get("pages", []):
                for span in page.get("pii_spans", []):
                    tag = str(span.get("tag", "") or "").lower()
                    if tag != "date":
                        continue
                    value = str(span.get("text", "") or "").strip()
                    parsed = _parse_study_like_date(value)
                    if parsed:
                        return parsed
            # 3) Last resort: DOB if nothing else parseable is found.
            for page in manifest.get("pages", []):
                for span in page.get("pii_spans", []):
                    tag = str(span.get("tag", "") or "").lower()
                    if tag != "dob":
                        continue
                    value = str(span.get("text", "") or "").strip()
                    parsed = _parse_study_like_date(value)
                    if parsed:
                        return parsed
        except Exception:
            return ""
        return ""

    def _clean_tracker_value(self, value: object) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        out = str(value).strip()
        if out.lower() in {"", "nan", "none", "<na>", "nat"}:
            return ""
        return out

    def _normalize_to_first_of_month(self, value: str) -> str:
        raw = self._clean_tracker_value(value)
        if not raw:
            return ""
        raw_norm = re.sub(r"[■█▓▒]+", " ", raw)
        raw_norm = re.sub(r"[,\u200b\u200c\u200d]+", " ", raw_norm)
        raw_norm = re.sub(r"\s+", " ", raw_norm).strip()
        ts = pd.to_datetime(raw_norm, errors="coerce")
        if pd.isna(ts):
            # Month-year fallback (for OCR like "Dec. 1999" or noisy "Dec ■ 1999").
            m = re.search(r"\b([A-Za-z]{3,9})\.?\s+(\d{4})\b", raw_norm, flags=re.IGNORECASE)
            if m:
                ts2 = pd.to_datetime(f"{m.group(1)} 01 {m.group(2)}", errors="coerce")
                if not pd.isna(ts2):
                    return ts2.strftime("%Y-%m-%d")
            return raw_norm
        ts = ts.replace(day=1)
        return ts.strftime("%Y-%m-%d")

    def _infer_patient_demographics_for_row(self, row) -> dict:
        manifest_path = Path(self._clean_tracker_value(row.get("manifest_json", "")))
        result = {"first_name": "", "last_name": "", "dob": "", "gender": ""}
        if not manifest_path.exists():
            return result
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return result

        candidate_names: list[tuple[str, int]] = []
        all_text_parts: list[str] = []
        dob_candidates: list[str] = []
        combined = ""
        date_hints = {"raw_dob": "", "dob": "", "raw_study_date": "", "study_date": ""}
        try:
            date_hints = infer_raw_dates_from_manifest(manifest_path)
        except Exception:
            date_hints = {"raw_dob": "", "dob": "", "raw_study_date": "", "study_date": ""}

        org_terms = {
            "cardiology",
            "department",
            "hospital",
            "clinic",
            "health",
            "healthcare",
            "medical",
            "medicine",
            "center",
            "centre",
            "university",
            "institute",
            "program",
            "service",
            "team",
            "group",
            "lab",
            "laboratory",
            "radiology",
            "echo",
            "cath",
            "ct",
            "cmr",
        }
        staff_terms = {
            "physician",
            "provider",
            "doctor",
            "dr",
            "md",
            "do",
            "pa",
            "np",
            "rn",
        }

        def _normalize_name_candidate(value: str) -> str:
            v = re.sub(r"\s+", " ", value or "").strip(" ,;:-")
            # Keep letters/space/comma/apostrophe/hyphen/period for initials.
            v = re.sub(r"[^A-Za-z ,.'-]", " ", v)
            v = re.sub(r"\s+", " ", v).strip(" ,;:-")
            return v

        def _is_bad_name(value: str) -> bool:
            low = value.lower()
            if not value:
                return True
            if len(value) < 3 or len(value) > 60:
                return True
            if "demographics" in low:
                return True
            if re.fullmatch(r"(?:patient\s*name|name)", low):
                return True
            if any(t in low for t in org_terms):
                return True
            if any(re.search(rf"\b{re.escape(t)}\b", low) for t in staff_terms):
                return True
            tokens = [t for t in re.split(r"[,\s]+", value) if t]
            alpha_tokens = [t for t in tokens if re.search(r"[A-Za-z]", t)]
            if len(alpha_tokens) < 2:
                return True
            if len(alpha_tokens) > 4:
                return True
            return False

        def _score_name(value: str, source_score: int) -> int:
            v = _normalize_name_candidate(value)
            if _is_bad_name(v):
                return -999
            s = source_score
            if "," in v:
                s += 2
            tokens = [t for t in re.split(r"[,\s]+", v) if t]
            if 2 <= len(tokens) <= 3:
                s += 2
            # Favor title-like casing for person names.
            title_like = sum(1 for t in tokens if t[:1].isupper())
            s += min(2, title_like)
            return s

        def _split_name(value: str) -> tuple[str, str]:
            v = _normalize_name_candidate(value)
            if "," in v:
                last, first = [p.strip() for p in v.split(",", 1)]
                first_token = first.split()[0] if first else ""
                last_token = last.split()[0] if last else ""
                return first_token, last_token
            parts = [p for p in v.split() if p]
            if len(parts) >= 2:
                return parts[0], parts[-1]
            return "", ""

        name_label_patterns = [
            r"\bpatient\s*name\s*[:\-]\s*([A-Za-z][A-Za-z ,.'-]{2,60})",
            r"\bname\s*[:\-]\s*([A-Za-z][A-Za-z ,.'-]{2,60})",
        ]
        for page in manifest.get("pages", []):
            page_text = str(
                page.get("source_text_for_dates", "")
                or page.get("source_text_for_export", "")
                or page.get("normalized_text_for_export", "")
                or page.get("text", "")
                or ""
            )
            if page_text:
                all_text_parts.append(page_text)
                # Highest priority: explicit "Patient Name:" / "Name:" labels.
                for patt in name_label_patterns:
                    for m in re.finditer(patt, page_text, flags=re.IGNORECASE):
                        raw = _normalize_name_candidate(m.group(1))
                        # Trim likely trailing fields after name segment.
                        raw = re.split(
                            r"\b(?:mrn|dob|date|gender|sex|study|exam|account|id)\b",
                            raw,
                            flags=re.IGNORECASE,
                        )[0].strip(" ,;:-")
                        score = _score_name(raw, 20)
                        if score > -999:
                            candidate_names.append((raw, score))
            for span in page.get("pii_spans", []):
                tag = str(span.get("tag", "") or "").lower()
                value = self._clean_tracker_value(span.get("text", ""))
                if not value:
                    continue
                if tag == "patient_name":
                    score = _score_name(value, 10)
                    if score > -999:
                        candidate_names.append((_normalize_name_candidate(value), score))
                elif tag == "dob":
                    dob_candidates.append(value)
            # Also leverage box-level patient_name values (table fallback path).
            for box in page.get("redaction_boxes", []):
                tag = str(box.get("tag", "") or "").lower()
                if tag != "patient_name":
                    continue
                value = self._clean_tracker_value(box.get("text", ""))
                if not value:
                    continue
                score = _score_name(value, 18)
                if score > -999:
                    candidate_names.append((_normalize_name_candidate(value), score))

        if candidate_names:
            # Prefer highest score (label-based candidates usually win).
            best_name, _ = sorted(candidate_names, key=lambda x: x[1], reverse=True)[0]
            first_name, last_name = _split_name(best_name)
            result["first_name"] = first_name
            result["last_name"] = last_name
        else:
            # Footer fallback:
            # Some reports emit patient identity as "<Name> - <MRN>" near "Study on ..."
            # and may miss patient_name tagging in the main demographics table.
            footer_text = "\n".join(all_text_parts)
            footer_lines = [str(ln or "").strip() for ln in footer_text.splitlines() if str(ln or "").strip()]
            footer_pat_comma = re.compile(
                r"(?i)^\s*([A-Za-z][A-Za-z'`\- ]{0,40},\s*[A-Za-z][A-Za-z'`\- ]{0,40})\s*-\s*\d{4,}\s*$"
            )
            footer_pat_space = re.compile(
                r"(?i)^\s*([A-Za-z][A-Za-z'`\- ]{1,40}\s+[A-Za-z][A-Za-z'`\- ]{1,40})\s*-\s*\d{4,}\s*$"
            )

            def _try_footer_name(line_value: str, source_score: int) -> None:
                nonlocal candidate_names
                if not line_value:
                    return
                m1 = footer_pat_comma.search(line_value)
                if m1:
                    raw = _normalize_name_candidate(str(m1.group(1) or ""))
                    score = _score_name(raw, source_score)
                    if score > -999:
                        candidate_names.append((raw, score))
                    return
                m2 = footer_pat_space.search(line_value)
                if m2:
                    raw = _normalize_name_candidate(str(m2.group(1) or ""))
                    score = _score_name(raw, source_score - 1)
                    if score > -999:
                        candidate_names.append((raw, score))

            for idx, line in enumerate(footer_lines):
                _try_footer_name(line, 16)
                # If this looks like the "Study on ..." footer marker, probe nearby lines
                # where the name/MRN line often sits.
                if re.search(r"\bstudy\s+on\b", line, flags=re.IGNORECASE):
                    if idx > 0:
                        _try_footer_name(footer_lines[idx - 1], 18)
                    if idx > 1:
                        _try_footer_name(footer_lines[idx - 2], 17)
                    if idx + 1 < len(footer_lines):
                        _try_footer_name(footer_lines[idx + 1], 14)

            if candidate_names:
                best_name, _ = sorted(candidate_names, key=lambda x: x[1], reverse=True)[0]
                first_name, last_name = _split_name(best_name)
                result["first_name"] = first_name
                result["last_name"] = last_name

        if dob_candidates:
            result["dob"] = self._normalize_to_first_of_month(dob_candidates[0])
        else:
            # Fallback: parse DOB/Date of Birth pattern from OCR text.
            combined = "\n".join(all_text_parts)
            dob_label_pat = r"(?:DOB|Date\s*of\s*B(?:i)?rth|Date\s*of\s*Brth|Born)"
            m = re.search(
                rf"\b{dob_label_pat}\b\s*[:\-]?\s*("
                r"[0-9]{1,4}[/-][0-9]{1,2}[/-][0-9]{1,4}|"
                r"[A-Za-z]{3,9}\.?\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{4}|"
                r"[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}|"
                r"[A-Za-z]{3,9}\.?\s*[-/]\s*\d{4}|"
                r"[-xX#]{1,4}\s*[-/]\s*[A-Za-z]{3,9}\.?\s*[-/]\s*\d{4}|"
                r"[A-Za-z]{3,9}\.?\s+\d{4})",
                combined,
                flags=re.IGNORECASE,
            )
            if m:
                result["dob"] = self._normalize_to_first_of_month(m.group(1))
            else:
                # Older report style: "Born ..." without explicit DOB label.
                born_match = re.search(
                    r"\bborn\b\s*[:\-]?\s*"
                    r"([0-9]{1,4}[/-][0-9]{1,2}[/-][0-9]{1,4}|"
                    r"[-xX#]{1,4}\s*[-/]\s*[A-Za-z]{3,9}\.?\s*[-/]\s*\d{4}|"
                    r"[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}|"
                    r"[A-Za-z]{3,9}\.?\s+\d{4}|"
                    r"[A-Za-z]{3,9}\.?\s*[^\w\s]{0,4}\s*\d{4}|"
                    r"\d{1,2}-[A-Za-z]{3}-\d{4})",
                    combined,
                    flags=re.IGNORECASE,
                )
                if born_match:
                    result["dob"] = self._normalize_to_first_of_month(born_match.group(1))

        if not combined:
            combined = "\n".join(all_text_parts)

        def _extract_date_like_tokens(text: str) -> list[str]:
            pats = re.compile(
                r"("
                r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|"
                r"\d{4}[./-]\d{1,2}[./-]\d{1,2}|"
                r"\d{1,2}-[A-Za-z]{3,9}-\d{2,4}|"
                r"[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}|"
                r"[A-Za-z]{3,9}\.?\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{4}|"
                r"[A-Za-z]{3,9}\.?\s+\d{4}|"
                r"[A-Za-z]{3,9}\.?\s*[-/]\s*\d{4}"
                r")",
                flags=re.IGNORECASE,
            )
            out: list[str] = []
            for m in pats.finditer(str(text or "")):
                out.append(str(m.group(1) or "").strip())
            return out

        def _is_valid_dob_vs_study(candidate_norm: str) -> bool:
            if not candidate_norm:
                return False
            dob_ts = pd.to_datetime(candidate_norm, errors="coerce")
            if pd.isna(dob_ts):
                return False
            study_hint = (
                self._clean_tracker_value(row.get("study_date", ""))
                or self._clean_tracker_value(date_hints.get("raw_study_date", ""))
                or self._clean_tracker_value(date_hints.get("study_date", ""))
            )
            study_norm = self._normalize_to_first_of_month(study_hint)
            if not study_norm:
                return True
            study_ts = pd.to_datetime(study_norm, errors="coerce")
            if pd.isna(study_ts):
                return True
            return bool(dob_ts < study_ts)

        # Final DOB rescue for stress-test style rows where label is present but OCR text is
        # partially redacted/clipped (e.g., "-May-2006" or "x-May-2006").
        if not result.get("dob", ""):
            line_pat = re.compile(
                r"(?im)^\s*(?:date\s*of\s*b(?:i)?rth|date\s*of\s*brth|dob|born)\s*[:\-]?\s*(?P<val>[^\n]{1,40})$"
            )
            month_year_pat = re.compile(
                r"(?i)(?:[-xX#]{1,4}\s*[-/]\s*)?(?P<mon>[A-Za-z]{3,9})\.?\s*[-/\s]\s*(?P<yr>\d{4})"
            )
            for lm in line_pat.finditer(combined):
                val = str(lm.group("val") or "").strip()
                mm = month_year_pat.search(val)
                if not mm:
                    continue
                mon = str(mm.group("mon") or "").strip()
                yr = str(mm.group("yr") or "").strip()
                candidate = f"{mon} {yr}"
                norm = self._normalize_to_first_of_month(candidate)
                if norm:
                    result["dob"] = norm
                    break

        # Cross-column/cross-line fallback for table OCR where "Date of Birth"
        # and its month-year value may be split into adjacent columns or next line.
        if not result.get("dob", ""):
            broad_pat = re.compile(
                r"(?is)\b(?:date\s*of\s*b(?:i)?rth|date\s*of\s*brth|dob|born)\b"
                r".{0,140}?"
                r"(?:[-xX#]{1,4}\s*[-/]\s*)?"
                r"(?P<mon>[A-Za-z]{3,9})\.?\s*[-/\s]\s*(?P<yr>\d{4})"
            )
            bm = broad_pat.search(combined)
            if bm:
                mon = str(bm.group("mon") or "").strip()
                yr = str(bm.group("yr") or "").strip()
                norm = self._normalize_to_first_of_month(f"{mon} {yr}")
                if norm:
                    result["dob"] = norm

        # Utility-level fallback: uses all manifest page text sources + span hints
        # and is more robust for split OCR table rows ("Date of Birth" + value in another cell).
        if not result.get("dob", ""):
            hint = (
                self._clean_tracker_value(date_hints.get("raw_dob", ""))
                or self._clean_tracker_value(date_hints.get("dob", ""))
            )
            norm = self._normalize_to_first_of_month(hint)
            if norm and _is_valid_dob_vs_study(norm):
                result["dob"] = norm

        # Split-row / reversed-order rescue:
        # table OCR can emit "<date>" then "Date of Birth" (or vice versa) on adjacent lines.
        if not result.get("dob", ""):
            lines = [str(ln or "").strip() for ln in combined.splitlines()]
            dob_label_rx = re.compile(r"\b(?:dob|date\s*of\s*b(?:i)?rth|date\s*of\s*brth|born)\b", re.IGNORECASE)
            bad_neighbor_rx = re.compile(r"\b(?:signed\s*on|study\s*time|performed\s*on)\b", re.IGNORECASE)
            for i, ln in enumerate(lines):
                prev_ln = lines[i - 1] if i > 0 else ""
                next_ln = lines[i + 1] if i + 1 < len(lines) else ""
                cands: list[str] = []
                if dob_label_rx.search(ln):
                    cands.extend(_extract_date_like_tokens(ln))
                    if not bad_neighbor_rx.search(prev_ln):
                        cands.extend(_extract_date_like_tokens(prev_ln))
                    if not bad_neighbor_rx.search(next_ln):
                        cands.extend(_extract_date_like_tokens(next_ln))
                # reverse order: date line followed/preceded by DOB label
                if _extract_date_like_tokens(ln) and (dob_label_rx.search(prev_ln) or dob_label_rx.search(next_ln)):
                    if not bad_neighbor_rx.search(ln):
                        cands.extend(_extract_date_like_tokens(ln))
                for cand in cands:
                    norm = self._normalize_to_first_of_month(cand)
                    if norm and _is_valid_dob_vs_study(norm):
                        result["dob"] = norm
                        break
                if result.get("dob", ""):
                    break

        combined_text = "\n".join(all_text_parts)
        g = re.search(
            r"\b(?:sex|gender)\s*[:\-]?\s*(male|female|m|f|non[-\s]?binary|other)\b",
            combined_text,
            flags=re.IGNORECASE,
        )
        if g:
            raw_g = g.group(1).strip().lower()
            if raw_g in {"m", "male"}:
                result["gender"] = "Male"
            elif raw_g in {"f", "female"}:
                result["gender"] = "Female"
            elif "non" in raw_g:
                result["gender"] = "Non-binary"
            else:
                result["gender"] = "Other"
        elif not result.get("gender", ""):
            lines = [str(ln or "").strip() for ln in combined_text.splitlines()]
            label_rx = re.compile(r"\b(?:sex|gender)\b", re.IGNORECASE)
            value_rx = re.compile(r"^(male|female|m|f|non[-\s]?binary|other)$", re.IGNORECASE)
            inline_value_rx = re.compile(r"\b(male|female|m|f|non[-\s]?binary|other)\b", re.IGNORECASE)
            for i, ln in enumerate(lines):
                prev_ln = lines[i - 1] if i > 0 else ""
                next_ln = lines[i + 1] if i + 1 < len(lines) else ""
                if label_rx.search(ln):
                    mv = inline_value_rx.search(ln) or value_rx.search(next_ln) or value_rx.search(prev_ln)
                    if mv:
                        raw_g = str(mv.group(1) or "").strip().lower()
                        if raw_g in {"m", "male"}:
                            result["gender"] = "Male"
                        elif raw_g in {"f", "female"}:
                            result["gender"] = "Female"
                        elif "non" in raw_g:
                            result["gender"] = "Non-binary"
                        else:
                            result["gender"] = "Other"
                        break
                if value_rx.search(ln) and (label_rx.search(prev_ln) or label_rx.search(next_ln)):
                    raw_g = str(value_rx.search(ln).group(1) or "").strip().lower()
                    if raw_g in {"m", "male"}:
                        result["gender"] = "Male"
                    elif raw_g in {"f", "female"}:
                        result["gender"] = "Female"
                    elif "non" in raw_g:
                        result["gender"] = "Non-binary"
                    else:
                        result["gender"] = "Other"
                    break

        # Demographics-header fallback for interleaved table order where label/value
        # may be split across columns (e.g., "Female ... Gender" on separate lines).
        if not result.get("dob", "") or not result.get("gender", ""):
            header_lines: list[str] = []
            for ln in combined_text.splitlines():
                s = str(ln or "").strip()
                if not s:
                    continue
                if re.search(r"\bsummary\b", s, flags=re.IGNORECASE):
                    break
                header_lines.append(s)
                if len(header_lines) >= 80:
                    break
            header_text = "\n".join(header_lines)

            def _study_ts_hint() -> pd.Timestamp | None:
                study_hint = (
                    self._clean_tracker_value(row.get("study_date", ""))
                    or self._clean_tracker_value(date_hints.get("raw_study_date", ""))
                    or self._clean_tracker_value(date_hints.get("study_date", ""))
                )
                study_norm = self._normalize_to_first_of_month(study_hint)
                if not study_norm:
                    return None
                ts = pd.to_datetime(study_norm, errors="coerce")
                return None if pd.isna(ts) else ts

            def _looks_like_gender_line(s: str) -> str:
                m = re.fullmatch(r"(?i)\s*(male|female|m|f|other|non[-\s]?binary)\s*", s)
                if not m:
                    return ""
                raw_g = str(m.group(1) or "").strip().lower()
                if raw_g in {"m", "male"}:
                    return "Male"
                if raw_g in {"f", "female"}:
                    return "Female"
                if "non" in raw_g:
                    return "Non-binary"
                return "Other"

            if not result.get("gender", ""):
                # First standalone gender token in demographics header wins.
                for ln in header_lines:
                    guessed = _looks_like_gender_line(ln)
                    if guessed:
                        result["gender"] = guessed
                        break
                # Weak fallback: token appearance near demographics terms.
                if not result.get("gender", ""):
                    mg = re.search(r"\b(male|female|non[-\s]?binary|other)\b", header_text, flags=re.IGNORECASE)
                    if mg:
                        raw_g = str(mg.group(1) or "").strip().lower()
                        if raw_g == "male":
                            result["gender"] = "Male"
                        elif raw_g == "female":
                            result["gender"] = "Female"
                        elif "non" in raw_g:
                            result["gender"] = "Non-binary"
                        else:
                            result["gender"] = "Other"

            if not result.get("dob", ""):
                study_ts = _study_ts_hint()
                # Collect date-like candidates from demographics header and choose
                # the oldest plausible value before study date.
                date_like = re.compile(
                    r"("
                    r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|"
                    r"\d{4}[./-]\d{1,2}[./-]\d{1,2}|"
                    r"\d{1,2}-[A-Za-z]{3,9}-\d{2,4}|"
                    r"[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}|"
                    r"[A-Za-z]{3,9}\.?\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{4}|"
                    r"[A-Za-z]{3,9}\.?\s+\d{4}|"
                    r"[A-Za-z]{3,9}\.?\s*[-/]\s*\d{4}"
                    r")",
                    flags=re.IGNORECASE,
                )
                dob_candidates_ranked: list[tuple[pd.Timestamp, str]] = []
                for m in date_like.finditer(header_text):
                    raw_val = str(m.group(1) or "").strip()
                    norm = self._normalize_to_first_of_month(raw_val)
                    if not norm:
                        continue
                    ts = pd.to_datetime(norm, errors="coerce")
                    if pd.isna(ts):
                        continue
                    if study_ts is not None and not (ts < study_ts):
                        continue
                    # Avoid obviously procedural/event dates from trailing history section.
                    if ts.year < 1900 or ts.year > 2100:
                        continue
                    dob_candidates_ranked.append((ts, norm))
                if dob_candidates_ranked:
                    # DOB is typically the oldest date in demographics block.
                    dob_candidates_ranked.sort(key=lambda t: t[0])
                    result["dob"] = dob_candidates_ranked[0][1]

        return result

    def _infer_mrn_for_row(self, row) -> str:
        manifest_path = Path(self._clean_tracker_value(row.get("manifest_json", "")))
        if not manifest_path.exists():
            return ""
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for page in manifest.get("pages", []):
                for span in page.get("pii_spans", []):
                    tag = str(span.get("tag", "") or "").lower()
                    if tag != "mrn":
                        continue
                    value = str(span.get("text", "") or "").strip()
                    if value:
                        return value
                # Fallback: box-level MRN (label/value geometry fallback path).
                for box in page.get("redaction_boxes", []):
                    tag = str(box.get("tag", "") or "").lower()
                    if tag != "mrn":
                        continue
                    value = str(box.get("text", "") or "").strip()
                    if value:
                        return value
        except Exception:
            return ""
        return ""

    def _build_file_id(self, force_id: str, study_date: str, instance: int) -> str:
        fid = (force_id or "").strip().upper()
        if not fid:
            fid = "XXX-LLLFFF-1"
        ts = pd.to_datetime(study_date, errors="coerce")
        ymd = "19000101" if pd.isna(ts) else ts.strftime("%Y%m%d")
        inst = int(instance)
        if inst < 1:
            inst = 1
        return f"{fid}_{ymd}_{inst}"

    def _token3(self, value: str) -> str:
        letters = re.sub(r"[^A-Za-z]", "", (value or "").upper())
        return (letters[:3] + "XXX")[:3]

    def _is_eu_mode(self) -> bool:
        return bool(getattr(self, "home_eu_mode_chk", None) and self.home_eu_mode_chk.isChecked())

    def _is_full_date_overlay_mode(self) -> bool:
        # EU mode takes precedence and uses age-at-event overlays.
        if self._is_eu_mode():
            return False
        return bool(
            getattr(self, "home_full_date_overlay_chk", None)
            and self.home_full_date_overlay_chk.isChecked()
        )

    def _is_numeric_force_id_mode(self) -> bool:
        if self._is_eu_mode():
            return True
        return bool(
            getattr(self, "home_numeric_force_id_chk", None)
            and self.home_numeric_force_id_chk.isChecked()
        )

    def _on_home_eu_mode_changed(self, _state: int) -> None:
        if self._is_eu_mode() and hasattr(self, "home_numeric_force_id_chk"):
            self.home_numeric_force_id_chk.setChecked(True)
        if self._is_eu_mode() and hasattr(self, "home_full_date_overlay_chk"):
            self.home_full_date_overlay_chk.setChecked(False)
        self._auto_generate_force_id()
        self._update_eu_metadata_ui()

    def _on_home_full_date_overlay_changed(self, _state: int) -> None:
        # Full-date overlay is a non-EU mode; prevent dual-mode confusion.
        if self._is_eu_mode() and hasattr(self, "home_full_date_overlay_chk"):
            self.home_full_date_overlay_chk.setChecked(False)

    def _resolve_mkf_tracker_path(self) -> str:
        explicit = self.tracker_edit.text().strip() if hasattr(self, "tracker_edit") else ""
        if explicit:
            try:
                return str(Path(explicit).expanduser().resolve())
            except Exception:
                return ""
        if self.tracker_path:
            return str(self.tracker_path)
        return self._resolve_tracker_path_from_inputs()

    def _demo_norm(self, value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    def _find_existing_numeric_force_id(self, site: str) -> str:
        tracker_path = self._resolve_mkf_tracker_path()
        if not tracker_path or not Path(tracker_path).exists():
            return ""
        try:
            df = load_tracker(tracker_path)
        except Exception:
            return ""

        need_cols = {"first_name", "last_name", "dob", "gender", "force_id"}
        if not need_cols.issubset(set(df.columns)):
            return ""
        fn = self._demo_norm(self.first_name_edit.text().strip() if hasattr(self, "first_name_edit") else "")
        ln = self._demo_norm(self.last_name_edit.text().strip() if hasattr(self, "last_name_edit") else "")
        dob = self._normalize_to_first_of_month(self.dob_edit.text().strip() if hasattr(self, "dob_edit") else "")
        gen = self._demo_norm(self.gender_edit.currentText().strip() if hasattr(self, "gender_edit") else "")
        if not (fn and ln and dob and gen):
            return ""

        for _, r in df.iterrows():
            rfn = self._demo_norm(r.get("first_name", ""))
            rln = self._demo_norm(r.get("last_name", ""))
            rdob = self._normalize_to_first_of_month(str(r.get("dob", "") or ""))
            rgen = self._demo_norm(r.get("gender", ""))
            if not (rfn == fn and rln == ln and rdob == dob and rgen == gen):
                continue
            fid = str(r.get("force_id", "") or "").strip().upper()
            if re.fullmatch(rf"{re.escape(site)}-\d{{6}}-\d+", fid):
                return fid
        return ""

    def _next_numeric_force_id(self, site: str) -> str:
        tracker_path = self._resolve_mkf_tracker_path()
        max_i = 0
        if tracker_path and Path(tracker_path).exists():
            try:
                df = load_tracker(tracker_path)
                if "force_id" in df.columns:
                    pat = re.compile(rf"^{re.escape(site)}-(\d{{6}})-(\d+)$")
                    for raw in df["force_id"].fillna("").astype(str):
                        m = pat.match(raw.strip().upper())
                        if not m:
                            continue
                        max_i = max(max_i, int(m.group(1)))
            except Exception:
                pass
        next_i = max_i + 1
        return f"{site}-{next_i:06d}-1"

    def _auto_generate_force_id(self) -> None:
        if not hasattr(self, "force_id_edit"):
            return
        site = self._token3(self._get_selected_site_id())
        first = self._token3(self.first_name_edit.text().strip() if hasattr(self, "first_name_edit") else "")
        last = self._token3(self.last_name_edit.text().strip() if hasattr(self, "last_name_edit") else "")
        existing = self.force_id_edit.text().strip().upper()
        if self._is_numeric_force_id_mode():
            reused = self._find_existing_numeric_force_id(site)
            if reused:
                generated = reused
            else:
                generated = self._next_numeric_force_id(site)
        else:
            m = re.match(r"^[A-Z]{3}-[A-Z0-9]{6}-(\d+)$", existing)
            suffix = m.group(1) if m else "1"
            generated = f"{site}-{last}{first}-{suffix}"
        self.force_id_edit.setText(generated)

    def _build_default_file_id(self, force_id: str, study_date: str) -> str:
        inst = int(self.instance_spin.value()) if hasattr(self, "instance_spin") else 1
        return self._build_file_id(force_id, study_date, inst)

    def _refresh_file_id_preview(self) -> None:
        force_id = self.force_id_edit.text().strip()
        study_date = self.study_date_edit.text().strip()
        file_id = self._build_default_file_id(force_id, study_date)
        self.file_id_edit.setText(file_id)

    def _save_metadata_for_current(self) -> None:
        if not self.tracker_path or not self.doc_id:
            raise ValueError("Load tracker and select a document first.")
        file_id = self._build_default_file_id(
            self.force_id_edit.text().strip(),
            self.study_date_edit.text().strip(),
        )
        self.file_id_edit.setText(file_id)
        set_review_metadata(
            tracker_csv=self.tracker_path,
            doc_id=self.doc_id,
            site_id=self._get_selected_site_id(),
            force_id=self.force_id_edit.text().strip(),
            file_id=file_id,
            modality_instance=int(self.instance_spin.value()),
            first_name=self.first_name_edit.text().strip(),
            last_name=self.last_name_edit.text().strip(),
            mrn=self.mrn_edit.text().strip(),
            dob=self._normalize_to_first_of_month(self.dob_edit.text().strip()),
            age_at_event=self.age_at_event_edit.text().strip() if hasattr(self, "age_at_event_edit") else "",
            gender=self.gender_edit.currentText().strip(),
            modality_type=self.modality_combo.currentText(),
            study_date=self.study_date_edit.text().strip(),
            eu_mode=self._is_eu_mode(),
        )
        # Ensure age_at_event is populated for all sites.
        self._sync_age_to_tracker_current_doc()

    def _missing_required_metadata_fields(self) -> List[str]:
        missing: List[str] = []
        checks: List[tuple[str, str]] = [
            ("FORCE ID", self.force_id_edit.text().strip() if hasattr(self, "force_id_edit") else ""),
            ("File ID", self.file_id_edit.text().strip() if hasattr(self, "file_id_edit") else ""),
            ("Modality", self.modality_combo.currentText().strip() if hasattr(self, "modality_combo") else ""),
            ("First Name", self.first_name_edit.text().strip() if hasattr(self, "first_name_edit") else ""),
            ("Last Name", self.last_name_edit.text().strip() if hasattr(self, "last_name_edit") else ""),
            ("MRN", self.mrn_edit.text().strip() if hasattr(self, "mrn_edit") else ""),
            ("DOB", self.dob_edit.text().strip() if hasattr(self, "dob_edit") else ""),
            ("Gender", self.gender_edit.currentText().strip() if hasattr(self, "gender_edit") else ""),
        ]
        for label, value in checks:
            if not value:
                missing.append(label)
        if self._is_eu_mode():
            age_val = self.age_at_event_edit.text().strip() if hasattr(self, "age_at_event_edit") else ""
            if not age_val:
                missing.append("Age at Event")
        else:
            study_val = self.study_date_edit.text().strip() if hasattr(self, "study_date_edit") else ""
            if not study_val:
                missing.append("Study Date")
        return missing

    def _autofill_dob_if_missing(self) -> bool:
        if not hasattr(self, "dob_edit"):
            return False
        if self._clean_tracker_value(self.dob_edit.text()):
            return False
        if not self.tracker_path or not self.doc_id:
            return False
        try:
            df = load_tracker(self.tracker_path)
        except Exception:
            return False
        row_df = df[df["doc_id"].astype(str) == str(self.doc_id)]
        if row_df.empty:
            return False
        row = row_df.iloc[-1]
        candidates: List[str] = []
        # High-confidence only: existing tracker DOB fields.
        candidates.append(self._clean_tracker_value(row.get("raw_dob", "")))
        candidates.append(self._clean_tracker_value(row.get("dob", "")))
        manifest_path = Path(self._clean_tracker_value(row.get("manifest_json", "")))
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for page in list(manifest.get("pages", []) or []):
                    for span in list(page.get("pii_spans", []) or []):
                        if str(span.get("tag", "") or "").strip().lower() == "dob":
                            candidates.append(self._clean_tracker_value(span.get("text", "")))
                    for box in list(page.get("redaction_boxes", []) or []):
                        if str(box.get("tag", "") or "").strip().lower() == "dob":
                            candidates.append(self._clean_tracker_value(box.get("text", "")))
                # Fallback to manifest-level inference only with strict temporal validation.
                hints = infer_raw_dates_from_manifest(manifest_path)
                hint_dob = self._clean_tracker_value(hints.get("raw_dob", "")) or self._clean_tracker_value(
                    hints.get("dob", "")
                )
                hint_study = (
                    self._clean_tracker_value(hints.get("raw_study_date", ""))
                    or self._clean_tracker_value(hints.get("study_date", ""))
                    or self._clean_tracker_value(self.study_date_edit.text() if hasattr(self, "study_date_edit") else "")
                )
                norm_hint_dob = self._normalize_to_first_of_month(hint_dob)
                norm_hint_study = self._normalize_to_first_of_month(hint_study)
                dob_ts = pd.to_datetime(norm_hint_dob, errors="coerce") if norm_hint_dob else pd.NaT
                study_ts = pd.to_datetime(norm_hint_study, errors="coerce") if norm_hint_study else pd.NaT
                if norm_hint_dob and not pd.isna(dob_ts):
                    if pd.isna(study_ts) or dob_ts < study_ts:
                        candidates.append(hint_dob)
            except Exception:
                pass
        # In-memory page edits (highest priority in current review session).
        for item in list(getattr(self, "base_box_items", []) or []):
            tag = str(item.get("tag", "") or "").strip().lower()
            if tag == "dob":
                candidates.append(self._clean_tracker_value(item.get("text", "")))
        for cand in candidates:
            norm = self._normalize_to_first_of_month(cand)
            if norm:
                self.dob_edit.setText(norm)
                self.status.setText("Auto-filled DOB from report metadata.")
                QApplication.processEvents()
                return True
        return False

    def _update_approval_progress(self, dialog: QProgressDialog | None, message: str) -> None:
        self.status.setText(message)
        if dialog is not None:
            dialog.setLabelText(message)
            dialog.setValue(0)
        QApplication.processEvents()

    def _auto_redact_patient_name_from_metadata(self) -> int:
        if not self.tracker_path or not self.doc_id:
            return 0
        first = self._clean_tracker_value(self.first_name_edit.text() if hasattr(self, "first_name_edit") else "")
        last = self._clean_tracker_value(self.last_name_edit.text() if hasattr(self, "last_name_edit") else "")
        if not first or not last:
            return 0
        phrases: List[str] = []
        phrases.append(f"{first} {last}")
        phrases.append(f"{last}, {first}")
        # Handle occasional middle initial/suffix after first name in table value.
        phrases.append(f"{last}, {first} ")
        cleaned: List[str] = []
        seen: set[str] = set()
        for p in phrases:
            norm = re.sub(r"\s+", " ", str(p or "").strip())
            if len(norm) < 4:
                continue
            if norm.lower() in seen:
                continue
            seen.add(norm.lower())
            cleaned.append(norm)
        if not cleaned:
            return 0
        try:
            _df, total_boxes = apply_manual_phrase_redaction(
                tracker_csv=self.tracker_path,
                doc_id=self.doc_id,
                phrases=cleaned,
            )
            return int(total_boxes)
        except Exception:
            return 0

    def _advance_to_next_or_send(self) -> None:
        # Prioritize unresolved queue. If none left, jump to Review tab.
        if self.pending_doc_combo.count() > 0:
            self.pending_doc_combo.setCurrentIndex(0)
            self.on_pending_doc_selected(self.pending_doc_combo.currentIndex())
            self.tabs.setCurrentIndex(1)
            return
        if self.reviewed_doc_combo.count() > 0:
            self.reviewed_doc_combo.setCurrentIndex(0)
            self.on_reviewed_doc_selected(self.reviewed_doc_combo.currentIndex())
        self.tabs.setCurrentIndex(2)

    def _apply_review_decision(self, approved: bool) -> None:
        if not self.tracker_path or not self.doc_id:
            QMessageBox.information(self, "Review", "Load tracker and select a document first.")
            return
        reviewer = self.user_name_edit.text().strip()
        if not reviewer:
            QMessageBox.warning(self, "Required", "Reviewer Name is required.")
            return
        if approved:
            self._autofill_dob_if_missing()
        if approved:
            missing = self._missing_required_metadata_fields()
            if missing:
                QMessageBox.warning(
                    self,
                    "Required metadata missing",
                    "Cannot mark OK to Send until all required metadata fields are completed.\n\n"
                    f"Missing: {', '.join(missing)}",
                )
                return
        busy_dialog = None
        try:
            if approved:
                busy_dialog = QProgressDialog("Preparing approval...", "", 0, 0, self)
                busy_dialog.setCancelButton(None)
                busy_dialog.setWindowModality(Qt.WindowModal)
                busy_dialog.setMinimumDuration(0)
                busy_dialog.setAutoClose(False)
                busy_dialog.setAutoReset(False)
                busy_dialog.show()
                QApplication.processEvents()
            self._update_approval_progress(busy_dialog, "Saving review decision...")
            self._save_metadata_for_current()
            if approved:
                self._update_approval_progress(
                    busy_dialog,
                    "Applying patient-name safety redaction from metadata...",
                )
                auto_name_boxes = self._auto_redact_patient_name_from_metadata()
                if auto_name_boxes > 0:
                    self.status.setText(
                        f"Applied {auto_name_boxes} additional patient-name redaction box(es) from metadata."
                    )
                    QApplication.processEvents()
                # Ensure the final redacted PDF reflects latest manual edits before marking approved.
                self._update_approval_progress(
                    busy_dialog,
                    "Applying final redactions to document...",
                )
                compile_redacted_document(self.tracker_path, self.doc_id)
            note = "OK to send" if approved else "Not OK to send"
            self._update_approval_progress(busy_dialog, "Persisting review decision...")
            set_review_decision(
                tracker_csv=self.tracker_path,
                doc_id=self.doc_id,
                approved_to_send=approved,
                reviewer=reviewer,
                review_notes=note,
                review_status="reviewed",
                approved_session_id=self.current_session_id,
            )
            # Verify persistence before refreshing UI.
            verify_df = load_tracker(self.tracker_path)
            vmask = verify_df["doc_id"].astype(str).eq(str(self.doc_id))
            if not bool(vmask.any()):
                raise RuntimeError(f"Could not verify saved decision for doc_id={self.doc_id}")
            vrow = verify_df[vmask].iloc[-1]
            approved_num = int(pd.to_numeric(vrow.get("approved_to_send", 0), errors="coerce") or 0)
            reviewed_txt = str(vrow.get("review_status", "") or "").strip().lower()
            if approved and approved_num != 1:
                raise RuntimeError("Review decision save failed: approved_to_send did not update.")
            if reviewed_txt != "reviewed":
                raise RuntimeError("Review decision save failed: review_status did not update to reviewed.")
            staged_path = ""
            if approved:
                self._update_approval_progress(
                    busy_dialog,
                    "Generating redacted OCR JSON and staging transfer files...",
                )
                staged_path = self._stage_current_doc_for_transfer()
            # Rebuild dropdowns immediately so current case moves from
            # "Needs Review" to "Reviewed / Approved".
            self._update_approval_progress(busy_dialog, "Refreshing tracker views...")
            current_doc = str(self.doc_id or "")
            self.open_tracker()
            self._refresh_review_tab()
            self._refresh_send_summary()
            if approved and staged_path:
                self.status.setText(f"Saved review decision: {note}. Staged at: {staged_path}")
            else:
                self.status.setText(f"Saved review decision: {note}.")
            # If current doc still exists in reviewed list, keep linkage.
            if current_doc:
                self._select_doc_by_id(current_doc)
            self._advance_to_next_or_send()
        except Exception as exc:
            self.status.setText(f"Save decision failed: {exc}")
            QMessageBox.warning(self, "Review", str(exc))
        finally:
            if busy_dialog is not None:
                try:
                    busy_dialog.close()
                except Exception:
                    pass

    def mark_ok_to_send(self) -> None:
        self._apply_review_decision(True)

    def mark_not_ok_to_send(self) -> None:
        self._apply_review_decision(False)

    def save_manual_metadata(self) -> None:
        if not self.tracker_path or not self.doc_id:
            QMessageBox.information(self, "Metadata", "Load tracker and select a document first.")
            return
        try:
            self._save_metadata_for_current()
            self.status.setText("Metadata saved.")
            self._refresh_review_tab()
            self.open_tracker()
            # Restore current selected doc after refresh.
            self._select_doc_by_id(self.doc_id)
        except Exception as exc:
            QMessageBox.warning(self, "Metadata", str(exc))

    def _stage_current_doc_for_transfer(self) -> str:
        if not self.tracker_path or not self.doc_id:
            raise ValueError("Load tracker and select a document first.")
        df = load_tracker(self.tracker_path)
        row_df = df[df["doc_id"].astype(str) == str(self.doc_id)]
        if row_df.empty:
            raise ValueError(f"doc_id not found in tracker: {self.doc_id}")
        row = row_df.iloc[-1]
        src = Path(str(row.get("redacted_file", "") or "")).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Redacted file not found for staging: {src}")
        out_dir = Path(self.tracker_path).resolve().parent / "approved_for_transfer" / "pdf"
        out_dir.mkdir(parents=True, exist_ok=True)

        file_id = self._clean_tracker_value(row.get("file_id", ""))
        if file_id:
            dest = out_dir / f"{file_id}{src.suffix.lower()}"
        else:
            dest = out_dir / src.name
        shutil.copy2(src, dest)
        # Generate/store redacted OCR payload immediately at approval time.
        generate_local_redacted_ocr_for_doc(
            tracker_csv=self.tracker_path,
            doc_id=str(self.doc_id),
        )
        return str(dest)

    def refresh_lists(self) -> None:
        self.base_list.clear()
        for i, item in enumerate(self.base_box_items, start=1):
            box = tuple(item.get("bbox_xyxy", [0, 0, 0, 0]))
            tag = str(item.get("tag", "") or "")
            text = str(item.get("text", "") or "").strip()
            text_short = (text[:80] + "...") if len(text) > 80 else text
            line = f"{i}: {box} | {tag} | {text_short if text_short else '[no text]'}"
            widget_item = QListWidgetItem(line)
            if text:
                widget_item.setToolTip(text)
            self.base_list.addItem(widget_item)

    def render_overlays(self) -> None:
        if self._pixmap_item is None:
            return
        try:
            if self._pixmap_item.pixmap().isNull():
                return
        except Exception:
            return
        for item in self._overlay_items:
            try:
                self.scene.removeItem(item)
            except RuntimeError:
                pass
        self._overlay_items = []

        # AI/NLP boxes in red.
        for idx, (x1, y1, x2, y2) in enumerate(self.base_boxes):
            item = QGraphicsRectItem(QRectF(x1, y1, x2 - x1, y2 - y1))
            item.setData(0, ("base", idx))
            if 0 <= idx < len(self.base_box_items):
                src = self.base_box_items[idx]
                tag = str(src.get("tag", "") or "").strip()
                txt = str(src.get("text", "") or "").strip()
                tooltip = ""
                if tag and txt:
                    tooltip = f"{tag}: {txt}"
                elif tag:
                    tooltip = tag
                elif txt:
                    tooltip = txt
                if tooltip:
                    item.setToolTip(tooltip)
            if self.resize_mode_kind == "base" and self.resize_mode_idx == idx:
                # Resize mode (selected for resize): light-blue cue.
                pen = QPen(QColor(0, 102, 255), 3)
                item.setBrush(QColor(120, 190, 255, 70))
            elif idx == self.selected_base_row:
                pen = QPen(QColor(255, 196, 0), 3)
                item.setBrush(QColor(255, 196, 0, 35))
            else:
                pen = QPen(QColor(255, 0, 0), 2)
                item.setBrush(QColor(255, 0, 0, 12))
            item.setPen(pen)
            self.scene.addItem(item)
            self._overlay_items.append(item)

        # Manual queued boxes in blue.
        for idx, (x1, y1, x2, y2) in enumerate(self.manual_boxes):
            item = QGraphicsRectItem(QRectF(x1, y1, x2 - x1, y2 - y1))
            item.setData(0, ("manual", idx))
            item.setToolTip("Manual redaction box")
            if self.resize_mode_kind == "manual" and self.resize_mode_idx == idx:
                # Resize mode (selected for resize): light-blue cue.
                pen = QPen(QColor(0, 102, 255), 3)
                item.setBrush(QColor(120, 190, 255, 70))
            elif idx == self.selected_manual_row:
                pen = QPen(QColor(255, 196, 0), 3)
                item.setBrush(QColor(255, 196, 0, 35))
            else:
                pen = QPen(QColor(0, 102, 255), 2)
                item.setBrush(QColor(0, 102, 255, 12))
            item.setPen(pen)
            self.scene.addItem(item)
            self._overlay_items.append(item)


def main() -> None:
    if sys.platform.startswith("win"):
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("FORCE.SafeHarborAI")
        except Exception:
            pass
    app = QApplication(sys.argv)
    icon_path = _resolve_app_icon_path()
    if icon_path is not None:
        icon = QIcon(str(icon_path))
        if not icon.isNull():
            app.setWindowIcon(icon)
    app.setStyleSheet(APP_QSS)
    window = DesktopRedactor()
    if icon_path is not None:
        icon = QIcon(str(icon_path))
        if not icon.isNull():
            window.setWindowIcon(icon)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


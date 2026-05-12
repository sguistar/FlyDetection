from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QPointF, QProcess, QProcessEnvironment, QRectF, QSize, Qt, QTimer, QUrl
from PyQt6.QtGui import QColor, QDesktopServices, QFont, QPainter, QPainterPath, QPen, QTextCursor
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
RUNS_ROOT = REPO_ROOT / "outputs" / "qt_gui" / "runs"
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov"}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from web.parsers import parse_log_progress, parse_tracks_csv, read_metrics_csv, read_table_csv, tail_text  # noqa: E402
from web.video import ensure_browser_playable_mp4  # noqa: E402

try:  # pragma: no cover - optional at runtime
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


THEMES: dict[str, dict[str, str]] = {
    "microscope-dark": {
        "label": "显微",
        "bg": "#08110f",
        "panel": "#101a18",
        "panel2": "#16231f",
        "text": "#e6fff5",
        "muted": "#88a99d",
        "line": "#26453c",
        "accent": "#4ff0b6",
        "accent2": "#7db4ff",
        "warn": "#ffcc66",
        "bad": "#ff7676",
        "video": "#040807",
    },
    "lab-light": {
        "label": "明亮",
        "bg": "#edf4f1",
        "panel": "#ffffff",
        "panel2": "#e5efeb",
        "text": "#17231f",
        "muted": "#5c7069",
        "line": "#bfd4cb",
        "accent": "#087f5b",
        "accent2": "#2f6ccf",
        "warn": "#a36b00",
        "bad": "#c92a2a",
        "video": "#dfe9e5",
    },
    "contrast": {
        "label": "高对比",
        "bg": "#000000",
        "panel": "#0e0e0e",
        "panel2": "#191919",
        "text": "#ffffff",
        "muted": "#d8d8d8",
        "line": "#f2f2f2",
        "accent": "#00ff99",
        "accent2": "#40c4ff",
        "warn": "#ffe600",
        "bad": "#ff3b3b",
        "video": "#000000",
    },
}


def qss(theme: dict[str, str]) -> str:
    return f"""
    QWidget {{
        background: {theme["bg"]};
        color: {theme["text"]};
        font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
        font-size: 13px;
    }}
    QMainWindow, QScrollArea, QSplitter {{
        background: {theme["bg"]};
    }}
    QLabel#eyebrow {{
        color: {theme["accent"]};
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
    }}
    QLabel#title {{
        font-size: 23px;
        font-weight: 800;
    }}
    QLabel#sectionTitle {{
        color: {theme["text"]};
        font-size: 15px;
        font-weight: 800;
        padding-top: 8px;
    }}
    QFrame#topbar, QFrame#panel, QFrame#videoPane, QFrame#trajectoryPane, QFrame#logDock {{
        background: {theme["panel"]};
        border: 1px solid {theme["line"]};
        border-radius: 8px;
    }}
    QFrame#rail, QFrame#inspector {{
        background: {theme["panel"]};
        border-right: 1px solid {theme["line"]};
        border-left: 1px solid {theme["line"]};
    }}
    QComboBox, QSpinBox, QTextEdit {{
        background: {theme["panel2"]};
        color: {theme["text"]};
        border: 1px solid {theme["line"]};
        border-radius: 6px;
        min-height: 30px;
        padding: 3px 8px;
        selection-background-color: {theme["accent"]};
    }}
    QComboBox::drop-down {{
        border: 0;
        width: 24px;
    }}
    QComboBox QAbstractItemView {{
        background: {theme["panel"]};
        color: {theme["text"]};
        border: 1px solid {theme["line"]};
        selection-background-color: {theme["accent"]};
    }}
    QPushButton {{
        background: {theme["panel2"]};
        color: {theme["text"]};
        border: 1px solid {theme["line"]};
        border-radius: 7px;
        min-height: 32px;
        padding: 5px 12px;
        font-weight: 700;
    }}
    QPushButton:hover {{
        border-color: {theme["accent"]};
        background: {mix(theme["panel2"], theme["accent"], 0.18)};
    }}
    QPushButton:pressed {{
        background: {mix(theme["panel2"], theme["accent"], 0.30)};
        padding-top: 7px;
        padding-bottom: 3px;
    }}
    QPushButton:disabled {{
        color: {theme["muted"]};
        border-color: {mix(theme["line"], theme["bg"], 0.50)};
    }}
    QPushButton#primary {{
        background: {theme["accent"]};
        color: {theme["bg"]};
        border-color: {theme["accent"]};
    }}
    QPushButton#themeButton[active="true"] {{
        background: {theme["accent"]};
        color: {theme["bg"]};
        border-color: {theme["accent"]};
    }}
    QCheckBox {{
        spacing: 8px;
        color: {theme["text"]};
        font-weight: 600;
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 5px;
        border: 1px solid {theme["line"]};
        background: {theme["panel2"]};
    }}
    QCheckBox::indicator:checked {{
        background: {theme["accent"]};
        border-color: {theme["accent"]};
    }}
    QSlider::groove:horizontal {{
        height: 6px;
        border-radius: 3px;
        background: {theme["panel2"]};
    }}
    QSlider::sub-page:horizontal {{
        border-radius: 3px;
        background: {theme["accent"]};
    }}
    QSlider::handle:horizontal {{
        width: 16px;
        height: 16px;
        margin: -5px 0;
        border-radius: 8px;
        background: {theme["text"]};
    }}
    QLabel#muted, QLabel#metricName, QLabel#artifactMeta {{
        color: {theme["muted"]};
    }}
    QLabel#metricValue {{
        font-size: 22px;
        font-weight: 850;
        color: {theme["accent"]};
    }}
    QFrame#metricCell, QFrame#eventRow, QFrame#artifactRow {{
        background: {theme["panel2"]};
        border-left: 3px solid {theme["accent"]};
        border-radius: 6px;
    }}
    QProgressBar {{
        background: {theme["panel2"]};
        border: 1px solid {theme["line"]};
        border-radius: 5px;
        height: 10px;
        text-align: center;
    }}
    QProgressBar::chunk {{
        border-radius: 4px;
        background: {theme["accent"]};
    }}
    """


def mix(left: str, right: str, amount: float) -> str:
    a = QColor(left)
    b = QColor(right)
    r = int(a.red() * (1 - amount) + b.red() * amount)
    g = int(a.green() * (1 - amount) + b.green() * amount)
    bl = int(a.blue() * (1 - amount) + b.blue() * amount)
    return QColor(r, g, bl).name()


class TrackCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(QSize(420, 280))
        self.tracks: list[dict[str, Any]] = []
        self.bounds: dict[str, Any] = {}
        self.theme = THEMES["microscope-dark"]
        self.phase = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(45)

    def set_theme(self, theme: dict[str, str]) -> None:
        self.theme = theme
        self.update()

    def load_tracks(self, payload: dict[str, Any]) -> None:
        self.tracks = payload.get("tracks") or []
        self.bounds = payload.get("bounds") or {}
        self.update()

    def _tick(self) -> None:
        self.phase = (self.phase + 0.025) % 6.283
        self.update()

    def paintEvent(self, _event: Any) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(14, 14, -14, -14)
        painter.fillRect(self.rect(), QColor(self.theme["video"]))
        self._draw_grid(painter, rect)
        if not self.tracks or self.bounds.get("minX") is None:
            painter.setPen(QColor(self.theme["muted"]))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "等待轨迹数据")
            return
        colors = [self.theme["accent"], self.theme["accent2"], self.theme["warn"], "#ff7ad9", "#9cff6a", "#ff9b75"]
        for index, track in enumerate(self.tracks):
            points = track.get("points") or []
            if len(points) < 2:
                continue
            path = QPainterPath()
            first = self._project(points[0], rect)
            path.moveTo(first)
            for point in points[1:]:
                path.lineTo(self._project(point, rect))
            color = QColor(colors[index % len(colors)])
            color.setAlpha(215)
            painter.setPen(QPen(color, 2.2))
            painter.drawPath(path)
            head = self._project(points[-1], rect)
            pulse = 4.0 + 2.0 * abs(math.sin(self.phase + index))
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(head, pulse, pulse)

    def _draw_grid(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(QColor(self.theme["line"]), 1))
        for i in range(12):
            x = rect.left() + rect.width() * i / 11
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        for i in range(8):
            y = rect.top() + rect.height() * i / 7
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        painter.setPen(QPen(QColor(self.theme["accent"]), 2))
        painter.drawRoundedRect(rect, 8, 8)

    def _project(self, point: dict[str, Any], rect: QRectF) -> QPointF:
        min_x = float(self.bounds.get("minX") or 0)
        max_x = float(self.bounds.get("maxX") or min_x + 1)
        min_y = float(self.bounds.get("minY") or 0)
        max_y = float(self.bounds.get("maxY") or min_y + 1)
        min_f = float(self.bounds.get("minFrame") or 0)
        max_f = float(self.bounds.get("maxFrame") or min_f + 1)
        nx = (float(point.get("x") or 0) - min_x) / max(max_x - min_x, 1)
        ny = (float(point.get("y") or 0) - min_y) / max(max_y - min_y, 1)
        nf = (float(point.get("f") or 0) - min_f) / max(max_f - min_f, 1)
        x = rect.left() + nx * rect.width() * 0.86 + nf * rect.width() * 0.10
        y = rect.top() + ny * rect.height() * 0.78 + nf * rect.height() * 0.16
        return QPointF(x, y)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        self.theme_name = "microscope-dark"
        self.job_id: str | None = None
        self.job_dir: Path | None = None
        self.output_root: Path | None = None
        self.process: QProcess | None = None
        self.external_video: Path | None = None
        self.total_frames: int | None = None
        self.result_loaded_for: str | None = None
        self.artifacts: dict[str, Path] = {}

        self.setWindowTitle("Fly MOT PyQt6 GUI")
        self.resize(1440, 930)
        self.setMinimumSize(1120, 720)
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_job)

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(12)
        outer.addWidget(self._build_topbar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_stage())
        splitter.addWidget(self._build_inspector())
        splitter.setSizes([310, 820, 310])
        outer.addWidget(splitter, 1)
        self.apply_theme("microscope-dark")
        self.load_defaults()

    def _build_topbar(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("topbar")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(18, 12, 18, 12)
        title_box = QVBoxLayout()
        eyebrow = QLabel("Fly MOT Native Console")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("果蝇多目标跟踪 PyQt6 控制台")
        title.setObjectName("title")
        title_box.addWidget(eyebrow)
        title_box.addWidget(title)
        layout.addLayout(title_box, 1)

        self.theme_buttons: dict[str, QPushButton] = {}
        for name, theme in THEMES.items():
            button = QPushButton(theme["label"])
            button.setObjectName("themeButton")
            button.clicked.connect(lambda _checked=False, n=name: self.apply_theme(n))
            self.theme_buttons[name] = button
            layout.addWidget(button)
        self.status_label = QLabel("待命")
        self.status_label.setMinimumWidth(86)
        layout.addWidget(self.status_label)
        return frame

    def _build_controls(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        rail = QFrame()
        rail.setObjectName("rail")
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.video_combo = QComboBox()
        self.model_combo = QComboBox()
        self.gt_combo = QComboBox()
        self.pick_video_btn = QPushButton("选择外部视频")
        self.pick_video_btn.clicked.connect(self.pick_external_video)
        layout.addWidget(section_label("输入"))
        layout.addWidget(field("视频文件", self.video_combo))
        layout.addWidget(self.pick_video_btn)
        layout.addWidget(field("模型权重", self.model_combo))
        layout.addWidget(field("GT 标注", self.gt_combo))

        self.conf_slider, self.conf_value = slider_pair(1, 500, 50, lambda v: f"{v / 1000:.3f}")
        self.iou_slider, self.iou_value = slider_pair(10, 95, 80, lambda v: f"{v / 100:.2f}")
        self.imgsz_spin = spinbox(320, 4096, 2560, 32)
        self.max_det_spin = spinbox(1, 256, 32)
        layout.addWidget(section_label("检测"))
        layout.addWidget(field("置信度", self.conf_slider, self.conf_value))
        layout.addWidget(field("IoU", self.iou_slider, self.iou_value))
        layout.addWidget(field("图像尺寸", self.imgsz_spin))
        layout.addWidget(field("最大检测", self.max_det_spin))

        self.num_flies_spin = spinbox(1, 64, 6)
        self.identity_slots_spin = spinbox(1, 64, 6)
        self.max_frames_spin = spinbox(0, 1_000_000, 0)
        self.use_cuda = QCheckBox("CUDA")
        self.use_cuda.setChecked(True)
        self.half_precision = QCheckBox("半精度")
        self.half_precision.setChecked(True)
        self.save_video = QCheckBox("渲染视频")
        self.save_video.setChecked(True)
        self.evaluation_enabled = QCheckBox("评估")
        self.evaluation_enabled.setChecked(True)
        layout.addWidget(section_label("跟踪"))
        layout.addWidget(field("果蝇数量", self.num_flies_spin))
        layout.addWidget(field("ID 槽位", self.identity_slots_spin))
        layout.addWidget(field("最大帧数（0 为完整）", self.max_frames_spin))
        for checkbox in [self.use_cuda, self.half_precision, self.save_video, self.evaluation_enabled]:
            layout.addWidget(checkbox)

        self.trail_slider, self.trail_value = slider_pair(0, 120, 24, str)
        self.draw_labels = QCheckBox("绘制标签")
        self.draw_labels.setChecked(True)
        layout.addWidget(section_label("显示"))
        layout.addWidget(field("轨迹长度", self.trail_slider, self.trail_value))
        layout.addWidget(self.draw_labels)

        self.start_btn = QPushButton("启动跟踪")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.start_job)
        self.cancel_btn = QPushButton("停止")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_job)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.cancel_btn)
        layout.addStretch(1)
        scroll.setWidget(rail)
        scroll.setMinimumWidth(290)
        return scroll

    def _build_stage(self) -> QWidget:
        stage = QWidget()
        layout = QVBoxLayout(stage)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header = QFrame()
        header.setObjectName("panel")
        h = QHBoxLayout(header)
        self.stage_title = QLabel("等待任务")
        self.stage_title.setObjectName("title")
        self.frame_label = QLabel("0 / 0")
        h.addWidget(self.stage_title, 1)
        h.addWidget(self.frame_label)
        layout.addWidget(header)

        split = QSplitter(Qt.Orientation.Horizontal)
        video_frame = QFrame()
        video_frame.setObjectName("videoPane")
        video_layout = QVBoxLayout(video_frame)
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(310)
        self.player.setVideoOutput(self.video_widget)
        video_layout.addWidget(self.video_widget)
        split.addWidget(video_frame)

        traj_frame = QFrame()
        traj_frame.setObjectName("trajectoryPane")
        traj_layout = QVBoxLayout(traj_frame)
        self.track_canvas = TrackCanvas()
        traj_layout.addWidget(self.track_canvas, 1)
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("采样"))
        self.stride_combo = QComboBox()
        for value in [1, 2, 5, 10]:
            self.stride_combo.addItem(f"{value}x", value)
        self.stride_combo.setCurrentIndex(2)
        self.stride_combo.currentIndexChanged.connect(self.load_tracks)
        reload_btn = QPushButton("刷新轨迹")
        reload_btn.clicked.connect(self.load_tracks)
        toolbar.addWidget(self.stride_combo)
        toolbar.addWidget(reload_btn)
        toolbar.addStretch(1)
        traj_layout.addLayout(toolbar)
        split.addWidget(traj_frame)
        split.setSizes([540, 430])
        layout.addWidget(split, 5)

        log_frame = QFrame()
        log_frame.setObjectName("logDock")
        log_layout = QVBoxLayout(log_frame)
        row = QHBoxLayout()
        row.addWidget(QLabel("运行日志"), 1)
        copy_btn = QPushButton("复制任务 ID")
        copy_btn.clicked.connect(self.copy_job_id)
        row.addWidget(copy_btn)
        log_layout.addLayout(row)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(150)
        log_layout.addWidget(self.log_edit)
        layout.addWidget(log_frame, 2)
        return stage

    def _build_inspector(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QFrame()
        panel.setObjectName("inspector")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        self.metrics_grid = QGridLayout()
        self.events_box = QVBoxLayout()
        self.artifacts_box = QVBoxLayout()
        layout.addWidget(section_label("指标"))
        layout.addLayout(self.metrics_grid)
        layout.addWidget(section_label("事件"))
        layout.addLayout(self.events_box)
        layout.addWidget(section_label("文件"))
        layout.addLayout(self.artifacts_box)
        layout.addStretch(1)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(285)
        self.render_empty_results()
        return scroll

    def apply_theme(self, name: str) -> None:
        self.theme_name = name
        theme = THEMES[name]
        QApplication.instance().setStyleSheet(qss(theme))
        for key, button in self.theme_buttons.items():
            button.setProperty("active", key == name)
            button.style().unpolish(button)
            button.style().polish(button)
        if hasattr(self, "track_canvas"):
            self.track_canvas.set_theme(theme)

    def load_defaults(self) -> None:
        self.video_combo.clear()
        self.model_combo.clear()
        self.gt_combo.clear()
        add_files(self.video_combo, discover_files([REPO_ROOT], VIDEO_EXTS), "未发现项目视频")
        add_files(self.model_combo, [p for p in discover_files([REPO_ROOT, REPO_ROOT / "outputs" / "models"], {".pt"}) if not p.name.lower().startswith("appearance_encoder")], "未发现模型")
        self.gt_combo.addItem("不使用 GT", "")
        for file_path in discover_files([REPO_ROOT / "coords"], {".csv"}):
            self.gt_combo.addItem(file_path.name, rel_or_abs(file_path))

    def pick_external_video(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频", str(REPO_ROOT), "Videos (*.mp4 *.avi *.mkv *.mov)")
        if not file_name:
            return
        self.external_video = Path(file_name)
        self.pick_video_btn.setText(f"外部：{self.external_video.name}")

    def start_job(self) -> None:
        try:
            video_path = self._selected_video()
            model_path = self._selected_model()
            gt_path = self._selected_gt()
        except Exception as exc:
            QMessageBox.warning(self, "配置不完整", str(exc))
            return

        self.job_id = f"{int(time.time())}-{random.randrange(16**8):08x}"
        self.job_dir = RUNS_ROOT / self.job_id
        self.output_root = self.job_dir / "outputs"
        self.job_dir.mkdir(parents=True, exist_ok=True)
        max_frames = self.max_frames_spin.value() or None
        self.total_frames = video_frame_count(video_path)
        expected_frames = min(self.total_frames, max_frames) if self.total_frames and max_frames else self.total_frames
        spec = {
            "job_id": self.job_id,
            "job_dir": str(self.job_dir),
            "output_root": str(self.output_root),
            "video_path": str(video_path),
            "model_path": str(model_path),
            "gt_csv_path": "" if gt_path is None else str(gt_path),
            "overrides": self._overrides(max_frames),
        }
        (self.job_dir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.job_dir / "job.json").write_text(
            json.dumps({"job_id": self.job_id, "status": "running", "total_frames": expected_frames}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.total_frames = expected_frames
        self.result_loaded_for = None
        self.render_empty_results()
        self.set_status("运行中")
        self.stage_title.setText(f"任务 {self.job_id} · tracking")
        self.log_edit.clear()
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        self.process = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONPATH", os.pathsep.join([str(REPO_ROOT), str(SRC_ROOT), env.value("PYTHONPATH", "")]))
        self.process.setProcessEnvironment(env)
        self.process.setWorkingDirectory(str(REPO_ROOT))
        self.process.readyReadStandardOutput.connect(self.read_process_output)
        self.process.readyReadStandardError.connect(self.read_process_output)
        self.process.finished.connect(self.job_finished)
        self.process.errorOccurred.connect(self.job_error)
        self.process.start(str(python_executable()), ["-m", "src.web.runner", str(self.job_dir / "spec.json")])
        self.poll_timer.start(1200)

    def _selected_video(self) -> Path:
        if self.external_video:
            return assert_file(self.external_video, VIDEO_EXTS)
        value = self.video_combo.currentData()
        if not value:
            raise ValueError("请选择视频文件。")
        return assert_file(resolve_project(value), VIDEO_EXTS)

    def _selected_model(self) -> Path:
        value = self.model_combo.currentData()
        if not value:
            raise ValueError("请选择模型权重。")
        return assert_file(resolve_project(value), {".pt"})

    def _selected_gt(self) -> Path | None:
        value = self.gt_combo.currentData()
        if not value:
            return None
        return assert_file(resolve_project(value), {".csv"})

    def _overrides(self, max_frames: int | None) -> dict[str, Any]:
        overrides = {
            "detection.conf_thres": self.conf_slider.value() / 1000,
            "detection.iou_thres": self.iou_slider.value() / 100,
            "detection.imgsz": self.imgsz_spin.value(),
            "detection.max_det": self.max_det_spin.value(),
            "track.num_flies": self.num_flies_spin.value(),
            "track.identity_slots": self.identity_slots_spin.value(),
            "runtime.use_cuda": self.use_cuda.isChecked(),
            "runtime.half_precision": self.half_precision.isChecked(),
            "runtime.save_video": self.save_video.isChecked(),
            "evaluation.enabled": self.evaluation_enabled.isChecked(),
            "render.trail_len": self.trail_slider.value(),
            "render.draw_labels": self.draw_labels.isChecked(),
        }
        if max_frames:
            overrides["runtime.max_frames"] = max_frames
        return overrides

    def cancel_job(self) -> None:
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()
            QTimer.singleShot(4000, self.process.kill)
        self.set_status("已停止")
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.poll_timer.stop()

    def read_process_output(self) -> None:
        if not self.process:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        text += bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        if text:
            self.log_edit.moveCursor(QTextCursor.MoveOperation.End)
            self.log_edit.insertPlainText(text)

    def poll_job(self) -> None:
        if not self.output_root:
            return
        log_path = self.output_root / "logs" / "run.log"
        log_text = tail_text(log_path, lines=180)
        if log_text:
            self.log_edit.setPlainText(log_text)
            self.log_edit.verticalScrollBar().setValue(self.log_edit.verticalScrollBar().maximum())
        progress = parse_log_progress(log_path)
        processed = progress.get("processed_frame")
        self.frame_label.setText(f"{processed or 0} / {self.total_frames or '?'}")
        if self.job_id:
            phase = "exporting" if self.total_frames and processed and processed + 1 >= self.total_frames else "tracking"
            self.stage_title.setText(f"任务 {self.job_id} · {phase}")

    def job_finished(self, code: int, _status: QProcess.ExitStatus) -> None:
        self.poll_timer.stop()
        self.poll_job()
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if code == 0:
            self.set_status("完成")
            self.load_results()
        else:
            self.set_status("失败")
            self.stage_title.setText(f"任务 {self.job_id or '-'} · failed")

    def job_error(self, error: QProcess.ProcessError) -> None:
        self.set_status("启动失败")
        self.log_edit.append(f"进程错误：{error.name}")
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.poll_timer.stop()

    def load_results(self) -> None:
        if not self.output_root or self.result_loaded_for == self.job_id:
            return
        self.result_loaded_for = self.job_id
        metrics = read_metrics_csv(self.output_root / "csv" / "metrics.csv")
        events = read_table_csv(self.output_root / "csv" / "events.csv", limit=300)
        self.render_metrics(metrics)
        self.render_events(events)
        self.render_artifacts()
        result_video = self.output_root / "videos" / "result.mp4"
        if result_video.exists():
            try:
                playable = ensure_browser_playable_mp4(result_video)
            except Exception:
                playable = result_video
            self.player.setSource(QUrl.fromLocalFile(str(playable)))
            self.player.play()
            self.player.pause()
        self.load_tracks()

    def load_tracks(self) -> None:
        if not self.output_root:
            return
        stride = int(self.stride_combo.currentData() or 1)
        payload = parse_tracks_csv(self.output_root / "csv" / "tracks.csv", stride=stride)
        self.track_canvas.load_tracks(payload)

    def render_empty_results(self) -> None:
        clear_layout(self.metrics_grid)
        clear_layout(self.events_box)
        clear_layout(self.artifacts_box)
        self.metrics_grid.addWidget(simple_label("等待指标"), 0, 0, 1, 2)
        self.events_box.addWidget(simple_label("等待事件"))
        self.artifacts_box.addWidget(simple_label("等待输出文件"))
        self.track_canvas.load_tracks({"tracks": [], "bounds": {}})
        self.player.stop()
        self.player.setSource(QUrl())

    def render_metrics(self, metrics: dict[str, Any]) -> None:
        clear_layout(self.metrics_grid)
        keys = [
            ("idf1", "IDF1"),
            ("point_hota", "Point HOTA"),
            ("mota_like", "MOTA-like"),
            ("det_a", "DetA"),
            ("assoc_a", "AssocA"),
            ("num_tracks", "Tracks"),
            ("matched_points", "Matched"),
            ("idsw", "IDSW"),
        ]
        for index, (key, label) in enumerate(keys):
            self.metrics_grid.addWidget(metric_cell(label, metrics.get(key)), index // 2, index % 2)

    def render_events(self, events: list[dict[str, Any]]) -> None:
        clear_layout(self.events_box)
        if not events:
            self.events_box.addWidget(simple_label("没有事件"))
            return
        for event in events[:14]:
            frame = event.get("frame_idx", event.get("frame", "-"))
            pair = " / ".join(str(v) for v in [event.get("display_track_a"), event.get("display_track_b")] if v is not None)
            self.events_box.addWidget(row_cell(str(event.get("type") or "event"), f"frame {frame}{' · ID ' + pair if pair else ''}"))

    def render_artifacts(self) -> None:
        clear_layout(self.artifacts_box)
        if not self.output_root:
            return
        self.artifacts = {
            "结果视频": self.output_root / "videos" / "result.mp4",
            "轨迹 CSV": self.output_root / "csv" / "tracks.csv",
            "事件 CSV": self.output_root / "csv" / "events.csv",
            "指标 CSV": self.output_root / "csv" / "metrics.csv",
            "检测 CSV": self.output_root / "csv" / "detections.csv",
            "日志": self.output_root / "logs" / "run.log",
        }
        for label, file_path in self.artifacts.items():
            self.artifacts_box.addWidget(artifact_cell(label, file_path, self.open_path))

    def open_path(self, file_path: Path) -> None:
        if file_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(file_path)))

    def copy_job_id(self) -> None:
        if self.job_id:
            QApplication.clipboard().setText(self.job_id)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)


def section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionTitle")
    return label


def simple_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("muted")
    return label


def field(label: str, widget: QWidget, value_label: QLabel | None = None) -> QWidget:
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    row = QHBoxLayout()
    name = QLabel(label)
    name.setObjectName("muted")
    row.addWidget(name, 1)
    if value_label is not None:
        row.addWidget(value_label)
    layout.addLayout(row)
    layout.addWidget(widget)
    return box


def slider_pair(minimum: int, maximum: int, value: int, formatter: Any) -> tuple[QSlider, QLabel]:
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(minimum, maximum)
    slider.setValue(value)
    label = QLabel(formatter(value))
    label.setMinimumWidth(52)
    slider.valueChanged.connect(lambda v: label.setText(formatter(v)))
    return slider, label


def spinbox(minimum: int, maximum: int, value: int, step: int = 1) -> QSpinBox:
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setValue(value)
    return box


def metric_cell(name: str, value: Any) -> QWidget:
    frame = QFrame()
    frame.setObjectName("metricCell")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(10, 8, 10, 8)
    value_label = QLabel(format_value(value))
    value_label.setObjectName("metricValue")
    name_label = QLabel(name)
    name_label.setObjectName("metricName")
    layout.addWidget(value_label)
    layout.addWidget(name_label)
    return frame


def row_cell(title: str, detail: str) -> QWidget:
    frame = QFrame()
    frame.setObjectName("eventRow")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(10, 7, 10, 7)
    strong = QLabel(title)
    strong.setFont(QFont(strong.font().family(), strong.font().pointSize(), QFont.Weight.Bold))
    small = QLabel(detail)
    small.setObjectName("muted")
    layout.addWidget(strong)
    layout.addWidget(small)
    return frame


def artifact_cell(label: str, file_path: Path, opener: Any) -> QWidget:
    frame = QFrame()
    frame.setObjectName("artifactRow")
    layout = QHBoxLayout(frame)
    layout.setContentsMargins(10, 7, 10, 7)
    layout.addWidget(QLabel(label), 1)
    meta = QLabel(format_bytes(file_path.stat().st_size) if file_path.exists() else "pending")
    meta.setObjectName("artifactMeta")
    layout.addWidget(meta)
    button = QPushButton("打开")
    button.setEnabled(file_path.exists())
    button.clicked.connect(lambda: opener(file_path))
    layout.addWidget(button)
    return frame


def clear_layout(layout: Any) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child = item.layout()
        if widget is not None:
            widget.deleteLater()
        elif child is not None:
            clear_layout(child)


def discover_files(roots: list[Path], extensions: set[str]) -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for file_path in root.iterdir():
            resolved = file_path.resolve()
            if not file_path.is_file() or file_path.suffix.lower() not in extensions or resolved in seen:
                continue
            if is_inside(resolved, REPO_ROOT / "outputs" / "qt_gui"):
                continue
            seen.add(resolved)
            files.append(resolved)
    return sorted(files, key=lambda p: p.name.lower())


def add_files(combo: QComboBox, files: list[Path], empty_label: str) -> None:
    if not files:
        combo.addItem(empty_label, "")
        return
    for file_path in files:
        combo.addItem(file_path.name, rel_or_abs(file_path))


def resolve_project(value: str) -> Path:
    path = Path(value)
    candidate = path if path.is_absolute() else (REPO_ROOT / path)
    resolved = candidate.resolve()
    if not is_inside(resolved, REPO_ROOT):
        raise ValueError("项目内文件路径必须留在仓库目录中。")
    return resolved


def assert_file(file_path: Path, extensions: set[str]) -> Path:
    resolved = file_path.resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"文件不存在：{file_path}")
    if resolved.suffix.lower() not in extensions:
        raise ValueError(f"文件类型不支持：{file_path}")
    return resolved


def is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def rel_or_abs(file_path: Path) -> str:
    resolved = file_path.resolve()
    return str(resolved.relative_to(REPO_ROOT)).replace("\\", "/") if is_inside(resolved, REPO_ROOT) else str(resolved)


def python_executable() -> Path | str:
    local = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    return local if local.exists() else sys.executable


def video_frame_count(video_path: Path) -> int | None:
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(str(video_path))
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return frames if frames > 0 else None
    finally:
        cap.release()


def format_value(value: Any) -> str:
    if value in (None, ""):
        return "-"
    if isinstance(value, (int, float)):
        if abs(value) >= 100:
            return f"{value:,.0f}"
        if abs(value) >= 10:
            return f"{value:.1f}"
        return f"{value:.3f}"
    return str(value)


def format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    index = 0
    while value >= 1024 and index < len(units) - 1:
        value /= 1024
        index += 1
    return f"{value:.0f} {units[index]}" if index == 0 else f"{value:.1f} {units[index]}"


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Fly MOT PyQt6 GUI")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

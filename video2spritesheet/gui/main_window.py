"""Main application window wiring UI to placeholder pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot, Qt, QEvent
from PySide6.QtGui import QAction, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGroupBox,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QCheckBox,
    QPushButton,
    QProgressBar,
    QLineEdit,
    QSpinBox,
    QSizePolicy,
    QSlider,
    QStatusBar,
    QStyle,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
)
from PIL import Image, ImageFilter

from ..core import FrameInfo, GenerationSettings, ProcessingOutcome, VideoMetadata
from ..core import frame_extractor, manifest_writer, spritesheet_builder, video_loader
from ..core.errors import InvalidVideoError, ProcessingError, ValidationError
from ..gui.file_picker import open_video_file_dialog
from ..gui.settings_panel import SettingsPanel
from ..gui.preview_dialog import PreviewDialog
from ..utils import file_tools
from ..utils.validators import ALLOWED_VIDEO_EXTENSIONS

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS = BASE_DIR / "assets"
DEFAULT_OUTPUT = BASE_DIR / "artifacts"


class WorkerSignals(QObject):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(object, object, object, object)  # outcome, PIL.Image, infos, thumbnails
    error = Signal(str)
    cancelled = Signal()


class GenerationWorker(QRunnable):
    """Background task to extract frames and build spritesheet."""

    def __init__(
        self,
        settings: GenerationSettings,
        metadata: Optional[VideoMetadata],
        selected_indices: Optional[list[int]] = None,
    ):
        super().__init__()
        self.settings = settings
        self.metadata = metadata
        self.selected_indices = selected_indices
        self.signals = WorkerSignals()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @Slot()
    def run(self) -> None:  # pragma: no cover - runs in thread pool
        try:
            meta = self.metadata or video_loader.load_metadata(self.settings.video_path)
            self.signals.status.emit("Extracting frames...")
            self.signals.progress.emit(10)
            times = frame_extractor._compute_sample_times(  # type: ignore[attr-defined]
                meta,
                self.settings.frame_count,
                self.settings.frame_interval,
                self.settings.max_frames,
                self.settings.start_time,
                self.settings.end_time,
            )
            frame_iter = frame_extractor.iter_frames(self.settings, meta, times)
            self.signals.progress.emit(40)

            if self._cancelled:
                self.signals.cancelled.emit()
                return

            # Apply selection filtering by wrapping generator if needed
            if self.selected_indices:
                selected_set = set(self.selected_indices)

                def filtered_iter():
                    for frame, info in frame_iter:
                        if info.index in selected_set:
                            yield frame, info

                frame_iter = filtered_iter()

            if self._cancelled:
                self.signals.cancelled.emit()
                return

            self.signals.status.emit("Building spritesheet...")
            spritesheet_path, sheet_image, packed_infos = spritesheet_builder.build_spritesheet_streaming(
                frame_iter, len(times), self.settings
            )
            self.signals.progress.emit(75)

            thumbnails = []
            for info in packed_infos[:12]:
                crop = sheet_image.crop((info.x, info.y, info.x + info.width, info.y + info.height))
                crop.thumbnail((96, 96))
                thumbnails.append(crop)

            manifest_path = None
            if self.settings.generate_manifest:
                self.signals.status.emit("Writing manifest...")
                columns, rows = spritesheet_builder._resolve_grid(len(times), self.settings.columns, self.settings.rows)
                manifest_path = manifest_writer.write_manifest(packed_infos, self.settings, columns, rows)
            self.signals.progress.emit(90)

            outcome = ProcessingOutcome(spritesheet_path=spritesheet_path, manifest_path=manifest_path)
            self.signals.progress.emit(100)
            self.signals.finished.emit(outcome, sheet_image, packed_infos, thumbnails)
        except Exception as exc:
            logger.exception("Generation failed")
            self.signals.error.emit(str(exc))


def _build_thumbnails(frames: list) -> list:
    """Create small thumbnails from PIL Images."""

    thumbs: list = []
    for frame in frames:
        thumb = frame.copy()
        thumb.thumbnail((96, 96))
        thumbs.append(thumb)
    return thumbs


class MainWindow(QMainWindow):
    """Primary application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video to Spritesheet")
        self.setMinimumSize(1100, 720)
        self.setWindowIcon(QIcon(self.style().standardIcon(QStyle.SP_FileDialogInfoView)))
        self._apply_styles()

        self.thread_pool = QThreadPool.globalInstance()
        self.progress_timer = QTimer(self)
        self.progress_timer.setInterval(150)
        self.progress_timer.timeout.connect(self._tick_progress)

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

        self.video_label = QLabel("No video selected", self)
        self.metadata_label = QLabel("Resolution: -, Duration: -, FPS: -", self)
        self.select_button = QPushButton("Select Video", self)
        self.generate_button = QPushButton("Generate Spritesheet", self)
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setEnabled(False)
        self.preview_button = QPushButton("Preview", self)
        self.play_button = QPushButton("Play Preview", self)
        self.stop_button = QPushButton("Stop Preview", self)
        self.preview_window_button = QPushButton("Open Preview Window", self)
        self.live_preview_checkbox = QCheckBox("Live preview", self)
        self.live_preview_checkbox.setChecked(False)
        self.mask_preview_checkbox = QCheckBox("Mask view", self)
        self.outline_preview_checkbox = QCheckBox("Outline view", self)
        self.working_dir_input = QLineEdit(self)
        self.working_dir_browse = QPushButton("Browse workspace", self)
        self.output_dir_input = QLineEdit(self)
        self.output_dir_browse = QPushButton("Browse outputs", self)
        self.refresh_lists_button = QPushButton("Refresh lists", self)
        self.outline_thickness = QSpinBox(self)
        self.outline_thickness.setRange(1, 10)
        self.outline_thickness.setValue(1)
        self.histogram_button = QPushButton("Show Alpha Histogram", self)
        self.progress_bar = QProgressBar(self)
        self.settings_panel = SettingsPanel(self)
        self.video_list = QListWidget(self)
        self.video_list.setSelectionMode(QListWidget.MultiSelection)
        self.output_list = QListWidget(self)
        self.output_list.setSelectionMode(QListWidget.SingleSelection)
        self.frame_list = QListWidget(self)
        self.frame_list.setSelectionMode(QListWidget.MultiSelection)
        self.preview_label = QLabel("Preview will appear here", self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(200)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.zoom_slider = self._build_zoom_slider()

        self.selected_video: Optional[Path] = None
        self.metadata: Optional[VideoMetadata] = None
        self.frame_infos: list[FrameInfo] = []
        self.thumbnails: list = []
        self._current_worker = None
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._step_animation)
        self._animation_index = 0
        self._preview_window: Optional[PreviewDialog] = None
        self.preview_label.setFrameShape(QFrame.Panel)
        self.preview_label.setFrameShadow(QFrame.Sunken)
        self._auto_preview_timer = QTimer(self)
        self._auto_preview_timer.setSingleShot(True)
        self._auto_preview_timer.timeout.connect(self._on_preview)
        self.preview_label.setToolTip("Preview (double-click frame to set key color from thumbnail, click preview to sample)")
        self.frame_list.setToolTip("Select/hover frames; double-click a frame to sample its key color")
        self.preview_button.setToolTip("Generate preview with current settings")
        self.live_preview_checkbox.setToolTip("Automatically refresh preview after edits")
        self.preview_label.setMouseTracking(True)
        self._last_image = None
        self._batch_queue: list[Path] = []

        # Sensible defaults for workspace/output
        default_workspace = DEFAULT_ASSETS if DEFAULT_ASSETS.exists() else BASE_DIR
        default_output = DEFAULT_OUTPUT if DEFAULT_OUTPUT.exists() else BASE_DIR / "artifacts"
        self.working_dir_input.setText(str(default_workspace))
        self.output_dir_input.setText(str(default_output))

        self._build_menu()
        self._build_layout()
        self._wire_signals()
        self._refresh_lists()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_action = QAction("Open", self)
        open_action.triggered.connect(self._on_select_video)
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        settings_menu = self.menuBar().addMenu("Settings")
        settings_menu.addAction(QAction("Preferences (placeholder)", self))

        help_menu = self.menuBar().addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _build_layout(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Workspace:", self))
        dir_row.addWidget(self.working_dir_input, 1)
        dir_row.addWidget(self.working_dir_browse)
        layout.addLayout(dir_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output folder:", self))
        out_row.addWidget(self.output_dir_input, 1)
        out_row.addWidget(self.output_dir_browse)
        layout.addLayout(out_row)

        layout.addWidget(self.metadata_label)
        content_row = QHBoxLayout()
        settings_box = QGroupBox("Output Settings", self)
        settings_layout = QVBoxLayout(settings_box)
        settings_layout.addWidget(self.settings_panel)
        content_row.addWidget(settings_box, 1)

        list_column = QVBoxLayout()
        video_box = QGroupBox("Workspace Videos", self)
        video_layout = QVBoxLayout(video_box)
        video_layout.addWidget(self.select_button)
        video_layout.addWidget(self.video_list, 1)
        video_layout.addWidget(self.refresh_lists_button)
        list_column.addWidget(video_box)

        output_box = QGroupBox("Output Spritesheets", self)
        output_layout = QVBoxLayout(output_box)
        output_layout.addWidget(self.output_list, 1)
        list_column.addWidget(output_box)

        frame_box = QGroupBox("Frames & Preview", self)
        frame_box_layout = QVBoxLayout(frame_box)
        frame_box_layout.addWidget(QLabel("Frames (toggle selection):", self))
        frame_box_layout.addWidget(self.frame_list, 1)
        preview_controls = QHBoxLayout()
        preview_controls.addWidget(self.preview_button)
        preview_controls.addWidget(self.play_button)
        preview_controls.addWidget(self.stop_button)
        preview_controls.addWidget(self.preview_window_button)
        preview_controls.addWidget(self.live_preview_checkbox)
        preview_controls.addWidget(self.mask_preview_checkbox)
        preview_controls.addWidget(self.outline_preview_checkbox)
        preview_controls.addWidget(QLabel("Outline px:", self))
        preview_controls.addWidget(self.outline_thickness)
        preview_controls.addWidget(self.histogram_button)
        preview_controls.addStretch(1)
        frame_box_layout.addLayout(preview_controls)
        frame_box_layout.addWidget(self.preview_label, 2)
        frame_box_layout.addWidget(self.zoom_slider)
        frame_box.setLayout(frame_box_layout)
        content_row.addLayout(list_column, 1)
        content_row.addWidget(frame_box, 1)

        layout.addLayout(content_row)
        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self.generate_button)
        buttons_row.addWidget(self.cancel_button)
        buttons_row.addWidget(self.progress_bar)
        layout.addLayout(buttons_row)

        self.setCentralWidget(central)

    def _wire_signals(self) -> None:
        self.select_button.clicked.connect(self._on_select_video)
        self.generate_button.clicked.connect(self._on_generate)
        self.cancel_button.clicked.connect(self._on_cancel)
        self.preview_button.clicked.connect(self._on_preview)
        self.play_button.clicked.connect(self._start_animation)
        self.stop_button.clicked.connect(self._stop_animation)
        self.preview_window_button.clicked.connect(self._open_preview_window)
        self.working_dir_browse.clicked.connect(self._on_browse_working_dir)
        self.output_dir_browse.clicked.connect(self._on_browse_output_dir)
        self.refresh_lists_button.clicked.connect(self._refresh_lists)
        self.frame_list.itemSelectionChanged.connect(self._on_frame_selected)
        self.frame_list.itemDoubleClicked.connect(self._on_frame_double_clicked)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        self.live_preview_checkbox.stateChanged.connect(self._on_live_toggle)
        self.settings_panel.changed.connect(self._schedule_auto_preview)
        self.mask_preview_checkbox.stateChanged.connect(self._refresh_preview_mask_toggle)
        self.outline_preview_checkbox.stateChanged.connect(self._refresh_preview_mask_toggle)
        self.outline_thickness.valueChanged.connect(self._refresh_preview_mask_toggle)
        self.preview_label.installEventFilter(self)
        self.video_list.itemSelectionChanged.connect(self._on_video_selected)
        self.output_list.itemSelectionChanged.connect(self._on_output_selected)
        self.histogram_button.clicked.connect(self._show_histogram)

    @Slot()
    def _on_select_video(self) -> None:
        chosen = open_video_file_dialog(self)
        if chosen is None:
            return
        self._load_single_video(chosen)

    def _update_metadata_label(self, metadata: VideoMetadata) -> None:
        self.metadata_label.setText(
            f"Resolution: {metadata.width}x{metadata.height}, Duration: {metadata.duration_seconds}s, FPS: {metadata.fps}"
        )

    def _populate_frames_placeholder(self, metadata: VideoMetadata) -> None:
        """Populate list with placeholder entries based on current frame selection inputs."""

        self.frame_list.clear()
        estimated = min(int(metadata.duration_seconds * metadata.fps // 2) or 1, 120)
        for idx in range(estimated):
            item = QListWidgetItem(f"Frame {idx:04d}")
            item.setCheckState(Qt.Checked)
            self.frame_list.addItem(item)
        self.frame_infos = []
        self.thumbnails = []

    def _selected_indices(self) -> list[int]:
        indices: list[int] = []
        for row in range(self.frame_list.count()):
            item = self.frame_list.item(row)
            if item.checkState() == Qt.Checked:
                if self.frame_infos and row < len(self.frame_infos):
                    indices.append(self.frame_infos[row].index)
                else:
                    indices.append(row)
        return indices

    def _populate_frames_from_infos(self, infos: list[FrameInfo], thumbnails) -> None:
        """Populate the frame list with actual extracted frame metadata."""

        if not infos:
            return
        self.frame_infos = list(infos)
        self.thumbnails = list(thumbnails) if thumbnails else []
        self.frame_list.clear()
        for idx, info in enumerate(infos):
            label = f"Frame {info.index:04d} @ {info.timestamp:.2f}s"
            item = QListWidgetItem(label)
            if thumbnails and idx < len(thumbnails):
                pix = self._pixmap_from_image(thumbnails[idx])
                if pix:
                    item.setIcon(pix)
            item.setCheckState(Qt.Checked)
            self.frame_list.addItem(item)

    def _populate_video_list(self) -> None:
        """Populate available videos from workspace directory."""

        self.video_list.clear()
        root_text = self.working_dir_input.text().strip()
        if not root_text:
            return
        root = Path(root_text)
        candidates = file_tools.list_files_with_extensions(root, ALLOWED_VIDEO_EXTENSIONS)
        for path in candidates:
            item = QListWidgetItem(path.name)
            item.setData(Qt.UserRole, str(path))
            self.video_list.addItem(item)

    def _populate_output_list(self) -> None:
        """Populate available spritesheets from output directory."""

        self.output_list.clear()
        root_text = self.output_dir_input.text().strip()
        if not root_text:
            return
        root = Path(root_text)
        output_exts = {".png", ".jpg", ".jpeg"}
        candidates = file_tools.list_files_with_extensions(root, output_exts)
        for path in candidates:
            item = QListWidgetItem(path.name)
            item.setData(Qt.UserRole, str(path))
            self.output_list.addItem(item)

    def _load_single_video(self, chosen: Path) -> None:
        try:
            self.selected_video = chosen
            self.metadata = video_loader.load_metadata(chosen)
            target_dir = Path(self.output_dir_input.text()) if self.output_dir_input.text() else chosen.parent
            filename = file_tools.format_output_filename(
                chosen, self.settings_panel.output_pattern_input.text().strip() or None
            )
            self.settings_panel.output_path.setText(str(target_dir / filename))
            self.settings_panel.fill_original_dimensions(self.metadata)
            self.video_label.setText(str(chosen))
            self._update_metadata_label(self.metadata)
            self.status_bar.showMessage(f"Video loaded: {chosen.name}")
            self._populate_frames_placeholder(self.metadata)
        except InvalidVideoError as exc:
            QMessageBox.warning(self, "Invalid video", str(exc))
            self.status_bar.showMessage("Invalid video selected")

    def _update_output_path(self) -> None:
        """Recompute output path when output folder changes."""

        if not self.selected_video:
            return
        target_dir = Path(self.output_dir_input.text()) if self.output_dir_input.text() else self.selected_video.parent
        filename = file_tools.format_output_filename(
            self.selected_video, self.settings_panel.output_pattern_input.text().strip() or None
        )
        self.settings_panel.output_path.setText(str(target_dir / filename))

    @Slot()
    def _on_video_selected(self) -> None:
        selected = self.video_list.selectedItems()
        if not selected:
            return
        chosen = Path(selected[0].data(Qt.UserRole))
        self._load_single_video(chosen)

    @Slot()
    def _on_output_selected(self) -> None:
        selected = self.output_list.selectedItems()
        if not selected:
            return
        path = Path(selected[0].data(Qt.UserRole))
        if not path.exists():
            return
        try:
            from PIL import Image
            image = Image.open(path).convert("RGBA")
            self._update_preview_label(image)
            self.status_bar.showMessage(f"Loaded spritesheet preview: {path.name}")
        except Exception as exc:
            QMessageBox.warning(self, "Preview error", f"Could not load {path}: {exc}")

    @Slot()
    def _on_preview(self) -> None:
        """Generate a preview spritesheet without blocking UI."""

        if self.cancel_button.isEnabled():
            return  # already running
        if not self.selected_video:
            QMessageBox.information(self, "No video", "Select a video before previewing.")
            return
        try:
            settings = self.settings_panel.gather_settings(
                video_path=self.selected_video, metadata=self.metadata
            )
        except ValidationError as exc:
            QMessageBox.warning(self, "Validation error", str(exc))
            return

        worker = GenerationWorker(settings=settings, metadata=self.metadata, selected_indices=self._selected_indices())
        worker.signals.status.connect(self.status_bar.showMessage)
        worker.signals.progress.connect(self.progress_bar.setValue)
        worker.signals.error.connect(self._on_worker_error)
        worker.signals.finished.connect(self._on_preview_finished)
        worker.signals.cancelled.connect(self._on_worker_cancelled)

        self._start_progress()
        self.generate_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._current_worker = worker
        self.thread_pool.start(worker)

    @Slot(object, object, object)
    def _on_preview_finished(self, outcome: ProcessingOutcome, image, infos, thumbnails) -> None:
        """Handle preview completion; display pixmap and keep progress state."""

        self._on_worker_finished(outcome, image, infos, thumbnails, preview_only=True)

    def _start_batch(self, template_settings: GenerationSettings) -> None:
        """Start batch processing using the current queue."""

        if not self._batch_queue:
            return
        next_video = self._batch_queue.pop(0)
        try:
            metadata = video_loader.load_metadata(next_video)
            # Build settings for this video based on template
            output_dir = Path(self.output_dir_input.text()) if self.output_dir_input.text() else next_video.parent
            file_tools.ensure_directory(output_dir)
            filename = file_tools.format_output_filename(next_video, template_settings.output_pattern)
            output_path = output_dir / filename
            manifest_path = output_path.with_suffix(".json") if template_settings.generate_manifest else None
            settings = GenerationSettings(
                video_path=next_video,
                output_path=output_path,
                output_width=template_settings.output_width,
                output_height=template_settings.output_height,
                frame_count=template_settings.frame_count,
                frame_interval=template_settings.frame_interval,
                start_time=template_settings.start_time,
                end_time=template_settings.end_time,
                columns=template_settings.columns,
                rows=template_settings.rows,
                generate_manifest=template_settings.generate_manifest,
                manifest_path=manifest_path,
                max_frames=template_settings.max_frames,
                background_color=template_settings.background_color,
                remove_black_background=template_settings.remove_black_background,
                chroma_key_color=template_settings.chroma_key_color,
                chroma_key_tolerance=template_settings.chroma_key_tolerance,
                auto_edge_cutout=template_settings.auto_edge_cutout,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Batch error", f"Failed to prepare {next_video}: {exc}")
            self._process_next_batch()
            return

        worker = GenerationWorker(
            settings=settings,
            metadata=metadata,
            selected_indices=[],
        )
        worker.signals.status.connect(self.status_bar.showMessage)
        worker.signals.progress.connect(self.progress_bar.setValue)
        worker.signals.error.connect(self._on_worker_error)
        worker.signals.finished.connect(self._on_worker_finished)
        worker.signals.cancelled.connect(self._on_worker_cancelled)

        self._start_progress()
        self.generate_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status_bar.showMessage(f"Processing {next_video.name} ({len(self._batch_queue)} remaining)")
        self._current_worker = worker
        self.thread_pool.start(worker)

    def _process_next_batch(self) -> None:
        """If batch queue remains, continue; else restore UI."""

        if self._batch_queue:
            # Use last used settings as template
            try:
                template = self.settings_panel.gather_settings(
                    video_path=self.selected_video or Path(""), metadata=self.metadata
                )
            except ValidationError:
                template = None
            if template:
                self._start_batch(template)
                return
        # Batch complete
        self.generate_button.setEnabled(True)
        self.preview_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._current_worker = None
    @Slot()
    def _on_generate(self) -> None:
        try:
            primary_settings = self.settings_panel.gather_settings(
                video_path=self.selected_video or Path(""), metadata=self.metadata
            )
        except ValidationError as exc:
            QMessageBox.warning(self, "Validation error", str(exc))
            return

        selected_videos = [self.video_list.item(i).data(Qt.UserRole) for i in range(self.video_list.count()) if self.video_list.item(i).isSelected()]
        if selected_videos:
            self._batch_queue = [Path(p) for p in selected_videos]
        else:
            self._batch_queue = [primary_settings.video_path]

        self._start_batch(primary_settings)

    def _start_progress(self) -> None:
        self.progress_bar.setValue(0)

    def _tick_progress(self) -> None:
        # Timer no longer used; kept for compatibility
        pass

    @Slot()
    def _on_browse_working_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Workspace Directory")
        if directory:
            self.working_dir_input.setText(directory)
            self._refresh_lists()

    @Slot()
    def _on_browse_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_input.setText(directory)
            self._refresh_lists()
            self._update_output_path()

    @Slot()
    def _refresh_lists(self) -> None:
        """Refresh video and output listings."""

        self._populate_video_list()
        self._populate_output_list()

    @Slot()
    def _on_cancel(self) -> None:
        """Request cancellation of the current worker."""

        worker = getattr(self, "_current_worker", None)
        if worker:
            worker.cancel()
        self.status_bar.showMessage("Cancelling...")

    @Slot(str)
    def _on_worker_error(self, message: str) -> None:
        self.progress_timer.stop()
        self.progress_bar.setValue(0)
        self.generate_button.setEnabled(True)
        self.preview_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._current_worker = None
        QMessageBox.critical(self, "Generation failed", message)
        self.status_bar.showMessage("Generation failed")

    @Slot()
    def _on_worker_cancelled(self) -> None:
        self.progress_timer.stop()
        self.progress_bar.setValue(0)
        self.generate_button.setEnabled(True)
        self.preview_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._current_worker = None
        self.status_bar.showMessage("Cancelled")
        self._stop_animation()

    def _update_preview_label(self, image) -> None:
        """Render a PIL image to the preview label."""

        if image is None:
            self.preview_label.setText("No preview available.")
            return
        self._last_image = image
        display_image = self._mask_preview(image) if self.mask_preview_checkbox.isChecked() else image
        if self.outline_preview_checkbox.isChecked():
            display_image = self._outline_preview(display_image)
        pixmap = self._pixmap_from_image(display_image)
        if pixmap:
            scaled = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_label.setPixmap(scaled)
            if self._preview_window:
                self._preview_window.update_pixmap(pixmap)
        else:
            self.preview_label.setText("Preview unavailable (ImageQt missing).")

    def _pixmap_from_image(self, image):
        """Convert a PIL image to QPixmap if ImageQt is available."""

        try:
            from PIL.ImageQt import ImageQt
        except ImportError:
            return None
        q_image = ImageQt(image)
        return QPixmap.fromImage(q_image)

    def _average_color(self, image) -> tuple[int, int, int, int]:
        """Return an average RGBA color from a PIL image thumbnail."""

        img = image.convert("RGBA").resize((8, 8))
        pixels = list(img.getdata())
        r = sum(p[0] for p in pixels) // len(pixels)
        g = sum(p[1] for p in pixels) // len(pixels)
        b = sum(p[2] for p in pixels) // len(pixels)
        a = sum(p[3] for p in pixels) // len(pixels)
        return (r, g, b, a)

    def _mask_preview(self, image):
        """Generate a red mask visualization from alpha."""

        img = image.convert("RGBA")
        alpha = img.split()[-1]
        red = Image.new("L", img.size, 255)
        transparent = Image.new("L", img.size, 0)
        return Image.merge("RGBA", (red, transparent, transparent, alpha))

    def _outline_preview(self, image):
        """Overlay an outline from the alpha channel onto the image."""

        img = image.convert("RGBA")
        alpha = img.split()[-1]
        thickness = max(1, self.outline_thickness.value())
        edges = alpha.filter(ImageFilter.FIND_EDGES)
        if thickness > 1:
            for _ in range(thickness - 1):
                edges = edges.filter(ImageFilter.MaxFilter(3))
        outline = edges.point(lambda v: 255 if v > 12 else 0)
        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        overlay.putalpha(outline)
        return Image.alpha_composite(img, overlay)

    def handle_preview_click(self, event) -> None:  # pragma: no cover - UI callback
        """Sample key color from the preview pixmap on click."""

        pixmap = self.preview_label.pixmap()
        if not pixmap:
            return
        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        if pos.x() < 0 or pos.y() < 0 or pos.x() >= pixmap.width() or pos.y() >= pixmap.height():
            return
        image = pixmap.toImage()
        color = image.pixelColor(pos)
        sampled = (color.red(), color.green(), color.blue(), color.alpha())
        self.settings_panel.set_chroma_color(sampled)
        self.status_bar.showMessage(f"Key color set from preview: rgba({sampled[0]},{sampled[1]},{sampled[2]},{sampled[3]})")

    def handle_preview_hover(self, event, pixmap=None) -> None:  # pragma: no cover - UI callback
        """Show RGBA under cursor in status bar."""

        target = pixmap or self.preview_label.pixmap()
        if not target:
            return
        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        if pos.x() < 0 or pos.y() < 0 or pos.x() >= target.width() or pos.y() >= target.height():
            return
        image = target.toImage()
        color = image.pixelColor(pos)
        self.status_bar.showMessage(
            f"Cursor RGBA: {color.red()},{color.green()},{color.blue()},{color.alpha()}"
        )

    def eventFilter(self, obj, event):  # pragma: no cover - UI hook
        if obj is self.preview_label and event.type() == QEvent.MouseMove:
            self.handle_preview_hover(event)
        return super().eventFilter(obj, event)

    @Slot()
    def _show_histogram(self) -> None:
        """Display a simple alpha histogram in status bar."""

        if self._last_image is None:
            self.status_bar.showMessage("No preview to analyze.")
            return
        alpha = self._last_image.convert("RGBA").split()[-1]
        hist = alpha.histogram()
        non_zero = sum(hist[1:])
        zero = hist[0]
        coverage = (non_zero / max(1, non_zero + zero)) * 100
        self.status_bar.showMessage(f"Alpha coverage: {coverage:.1f}% non-transparent pixels")

    def _build_zoom_slider(self):
        slider = QSlider(Qt.Horizontal, self)
        slider.setRange(25, 200)
        slider.setValue(100)
        slider.setTickInterval(25)
        slider.setTickPosition(QSlider.TicksBelow)
        return slider

    @Slot()
    def _on_frame_selected(self) -> None:
        """Update preview when a frame is selected."""

        selected = self.frame_list.selectedIndexes()
        if not selected or not self.thumbnails:
            return
        idx = selected[0].row()
        if idx < len(self.thumbnails):
            self._update_preview_label(self.thumbnails[idx])
            self._animation_index = idx

    @Slot()
    def _on_frame_double_clicked(self, item: QListWidgetItem) -> None:
        """Sample a key color from the clicked thumbnail."""

        row = self.frame_list.row(item)
        if row < len(self.thumbnails):
            thumb = self.thumbnails[row]
            color = self._average_color(thumb)
            self.settings_panel.set_chroma_color(color)
            self.status_bar.showMessage(f"Key color set from frame {row:04d}")

    @Slot(int)
    def _on_zoom_changed(self, value: int) -> None:
        """Adjust preview scaling based on slider."""

        if self.preview_label.pixmap():
            scaled = self.preview_label.pixmap().scaled(
                self.preview_label.size() * (value / 100.0), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled)
        if self._preview_window and self.preview_label.pixmap():
            self._preview_window.update_pixmap(self.preview_label.pixmap())

    def _refresh_preview_mask_toggle(self) -> None:
        """Re-render preview with/without mask overlay."""

        if self._last_image is not None:
            self._update_preview_label(self._last_image)

    @Slot()
    def _start_animation(self) -> None:
        """Play through selected thumbnails as an animation."""

        if not self.thumbnails:
            return
        self._animation_index = 0
        self._animation_timer.start(self._animation_interval_ms())

    @Slot()
    def _stop_animation(self) -> None:
        """Stop the animation timer."""

        self._animation_timer.stop()

    def _animation_interval_ms(self) -> int:
        """Compute interval based on zoom slider (reuse as speed control)."""

        # Map slider 25-200 to approx 6-48 fps
        value = self.zoom_slider.value()
        fps = max(6.0, min(48.0, value / 4.0))
        return int(1000 / fps)

    def _step_animation(self) -> None:
        """Advance animation frame."""

        indices = self._selected_indices()
        pool = indices if indices else list(range(len(self.thumbnails)))
        if not pool:
            return
        self._animation_index = (self._animation_index + 1) % len(pool)
        idx = pool[self._animation_index]
        if idx < len(self.thumbnails):
            self._update_preview_label(self.thumbnails[idx])

    def _open_preview_window(self) -> None:
        """Open a detached preview window for larger viewing."""

        if self._preview_window is None:
            self._preview_window = PreviewDialog(self)
        self._preview_window.show()
        if self.preview_label.pixmap():
            self._preview_window.update_pixmap(self.preview_label.pixmap())

    @Slot(int)
    def _on_live_toggle(self, state: int) -> None:
        if state:
            self._schedule_auto_preview()
        else:
            self._auto_preview_timer.stop()

    def _schedule_auto_preview(self) -> None:
        """Debounced auto-preview when live preview is enabled."""

        if not self.live_preview_checkbox.isChecked():
            return
        if not self.selected_video or self.cancel_button.isEnabled():
            return
        self._auto_preview_timer.start(400)

    @Slot(object, object, object)
    def _on_worker_finished(
        self, outcome: ProcessingOutcome, image, infos, thumbnails, preview_only: bool = False
    ) -> None:
        self.progress_bar.setValue(100)
        self.generate_button.setEnabled(True)
        self.preview_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._stop_animation()
        self._current_worker = None
        status = f"Spritesheet ready at {outcome.spritesheet_path}"
        if outcome.manifest_path:
            status += f" | Manifest: {outcome.manifest_path}"
        self.status_bar.showMessage(status)
        self._update_preview_label(image)
        self._populate_frames_from_infos(infos, thumbnails)
        if not preview_only:
            self._populate_output_list()
        if not preview_only:
            self._process_next_batch()
        if preview_only:
            # Preview run only; do not show modal
            return

    def _apply_styles(self) -> None:
        """Apply a simple modern palette."""

        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #0b1324;
                color: #e5e7eb;
                font-family: 'Segoe UI', 'Arial';
                font-size: 13pt;
            }
            QGroupBox {
                border: 1px solid #1f2937;
                border-radius: 6px;
                margin-top: 8px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: #cbd5e1;
            }
            QPushButton {
                background-color: #2563eb;
                color: #f8fafc;
                border-radius: 4px;
                padding: 8px 14px;
                font-size: 12pt;
            }
            QPushButton:disabled {
                background-color: #1f2937;
                color: #94a3b8;
            }
            QListWidget {
                background-color: #0f172a;
                border: 1px solid #1f2937;
                color: #e5e7eb;
                font-size: 12pt;
            }
            QLabel {
                color: #e5e7eb;
                font-size: 13pt;
            }
            QProgressBar {
                border: 1px solid #1f2937;
                border-radius: 4px;
                text-align: center;
                background: #0f172a;
                color: #e5e7eb;
                font-size: 11pt;
            }
            QProgressBar::chunk {
                background-color: #22c55e;
                width: 10px;
            }
            QLineEdit {
                background-color: #0b1220;
                border: 1px solid #1f2937;
                color: #e5e7eb;
                padding: 8px 10px;
                border-radius: 4px;
                font-size: 12pt;
            }
            QListWidget::item:selected {
                background-color: #2563eb;
                color: #e5e7eb;
            }
            QCheckBox { color: #e5e7eb; }
            QMenuBar {
                background-color: #0f172a;
                color: #e5e7eb;
                font-size: 11pt;
            }
            QMenuBar::item:selected {
                background: #1f2937;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #1f2937;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #2563eb;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            """
        )

    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            "About",
            "Video2SpriteSheet\nPySide6 UI with moviepy+pillow-based spritesheet generation and manifest stubs.",
        )

    def closeEvent(self, event) -> None:  # pragma: no cover - Qt lifecycle
        self.progress_timer.stop()
        super().closeEvent(event)

"""Settings panel UI for spritesheet generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSize, Signal
from PySide6.QtWidgets import (
    QColorDialog,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core import GenerationSettings, VideoMetadata
from ..core.errors import ValidationError
from ..utils import file_tools, validators

logger = logging.getLogger(__name__)


class SettingsPanel(QWidget):
    """Collects output and layout settings from the user."""

    changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.output_path = QLineEdit(self)
        self.output_browse = QPushButton("Browse", self)
        self.output_path.setReadOnly(True)
        self.output_browse.setVisible(False)

        self.width_input = QLineEdit(self)
        self.height_input = QLineEdit(self)
        self.frame_count_input = QLineEdit(self)
        self.frame_interval_input = QLineEdit(self)
        self.columns_input = QLineEdit(self)
        self.rows_input = QLineEdit(self)
        self.start_time_input = QLineEdit(self)
        self.end_time_input = QLineEdit(self)
        self.max_frames_input = QLineEdit(self)
        self.generate_manifest = QCheckBox("Generate manifest JSON", self)
        self.remove_black_checkbox = QCheckBox("Make black pixels transparent", self)
        self.background_color_btn = QPushButton("Background color...", self)
        self.background_color_label = QLabel("Current: transparent", self)
        self._background_color: Optional[tuple[int, int, int, int]] = None
        self.chroma_color_btn = QPushButton("Key color...", self)
        self.chroma_color_label = QLabel("Key: black")
        self.chroma_tolerance = QSpinBox(self)
        self.chroma_tolerance.setRange(0, 255)
        self.chroma_tolerance.setValue(12)
        self._chroma_color: Optional[tuple[int, int, int, int]] = (0, 0, 0, 255)
        self.auto_edge_checkbox = QCheckBox("Edge-aware cutout", self)
        self.auto_edge_checkbox.setChecked(True)
        self.padding_input = QSpinBox(self)
        self.padding_input.setRange(0, 64)
        self.padding_input.setValue(0)
        self.output_pattern_input = QLineEdit(self)
        self.output_pattern_input.setPlaceholderText("{stem}.png")

        self._build_layout()
        self.output_browse.clicked.connect(self._open_output_dialog)
        self.background_color_btn.clicked.connect(self._choose_color)
        self.chroma_color_btn.clicked.connect(self._choose_chroma_color)
        for line_edit in (
            self.width_input,
            self.height_input,
            self.frame_count_input,
            self.frame_interval_input,
            self.start_time_input,
            self.end_time_input,
            self.columns_input,
            self.rows_input,
            self.max_frames_input,
            self.output_pattern_input,
        ):
            line_edit.textChanged.connect(self.changed.emit)
        self.generate_manifest.stateChanged.connect(self.changed.emit)
        self.remove_black_checkbox.stateChanged.connect(self.changed.emit)
        self.chroma_tolerance.valueChanged.connect(self.changed.emit)
        self.auto_edge_checkbox.stateChanged.connect(self.changed.emit)
        self.padding_input.valueChanged.connect(self.changed.emit)

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)

        # Output path selector
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(self.output_browse)
        layout.addLayout(output_layout)

        form_layout = QFormLayout()
        self.width_input.setPlaceholderText("Keep original")
        self.height_input.setPlaceholderText("Keep original")
        form_layout.addRow("Output width", self.width_input)
        form_layout.addRow("Output height", self.height_input)

        self.frame_count_input.setPlaceholderText("Total frames (optional)")
        self.frame_interval_input.setPlaceholderText("Seconds between frames")
        form_layout.addRow("Frame count", self.frame_count_input)
        form_layout.addRow("Frame interval", self.frame_interval_input)

        self.start_time_input.setPlaceholderText("Start time (sec)")
        self.end_time_input.setPlaceholderText("End time (sec)")
        form_layout.addRow("Start at", self.start_time_input)
        form_layout.addRow("End at", self.end_time_input)

        self.columns_input.setPlaceholderText("Auto")
        self.rows_input.setPlaceholderText("Auto")
        form_layout.addRow("Grid columns", self.columns_input)
        form_layout.addRow("Grid rows", self.rows_input)

        self.max_frames_input.setPlaceholderText("180 (cap) or 0 for all")
        form_layout.addRow("Max frames", self.max_frames_input)

        form_layout.addRow("Frame padding (px)", self.padding_input)
        form_layout.addRow("Output pattern", self.output_pattern_input)

        bg_row = QHBoxLayout()
        bg_row.addWidget(self.background_color_btn)
        bg_row.addWidget(self.background_color_label)
        form_layout.addRow("Spritesheet background", bg_row)
        form_layout.addRow("", self.remove_black_checkbox)
        chroma_row = QHBoxLayout()
        chroma_row.addWidget(self.chroma_color_btn)
        chroma_row.addWidget(self.chroma_color_label)
        form_layout.addRow("Key color", chroma_row)
        form_layout.addRow("Key tolerance", self.chroma_tolerance)
        form_layout.addRow("", self.auto_edge_checkbox)

        layout.addLayout(form_layout)
        layout.addWidget(self.generate_manifest)
        layout.addStretch(1)

    def set_default_output(self, video_path: Path) -> None:
        """Populate a default output path next to the video."""

        default = file_tools.default_output_path(video_path)
        self.output_path.setText(str(default))

    def fill_original_dimensions(self, metadata: VideoMetadata) -> None:
        """Set width/height fields to the source dimensions."""

        self.width_input.setText(str(metadata.width))
        self.height_input.setText(str(metadata.height))

    def _open_output_dialog(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            current = Path(self.output_path.text()) if self.output_path.text() else None
            filename = current.name if current else "spritesheet.png"
            self.output_path.setText(str(Path(directory) / filename))

    def gather_settings(self, video_path: Path, metadata: Optional[VideoMetadata]) -> GenerationSettings:
        """Validate inputs and build a GenerationSettings instance."""

        if not video_path:
            raise ValidationError("Select a video before generating a spritesheet.")

        output_text = self.output_path.text().strip()
        if not output_text:
            raise ValidationError("Provide an output path for the spritesheet.")

        output_path = Path(output_text)

        width = validators.parse_optional_int(self.width_input.text().strip(), "Width")
        height = validators.parse_optional_int(self.height_input.text().strip(), "Height")
        validators.validate_dimensions(width, height)

        frame_count = validators.parse_optional_int(self.frame_count_input.text().strip(), "Frame count")
        frame_interval = validators.parse_optional_float(
            self.frame_interval_input.text().strip(), "Frame interval"
        )
        validators.validate_frame_selection(frame_count, frame_interval)

        start_time = validators.parse_optional_non_negative_float(self.start_time_input.text().strip(), "Start time")
        end_time = validators.parse_optional_non_negative_float(self.end_time_input.text().strip(), "End time")
        validators.validate_time_range(start_time, end_time)

        columns = validators.parse_optional_int(self.columns_input.text().strip(), "Columns")
        rows = validators.parse_optional_int(self.rows_input.text().strip(), "Rows")
        validators.validate_grid(columns, rows)

        max_frames_parsed = validators.parse_optional_non_negative_int(
            self.max_frames_input.text().strip(), "Max frames"
        )
        max_frames = None if max_frames_parsed == 0 else max_frames_parsed

        validators.validate_tolerance(self.chroma_tolerance.value(), "Key tolerance")

        manifest_path = output_path.with_suffix(".json") if self.generate_manifest.isChecked() else None

        settings = GenerationSettings(
            video_path=video_path,
            output_path=output_path,
            output_width=width or (metadata.width if metadata else None),
            output_height=height or (metadata.height if metadata else None),
            frame_count=frame_count,
            frame_interval=frame_interval,
            start_time=start_time,
            end_time=end_time,
            columns=columns,
            rows=rows,
            generate_manifest=self.generate_manifest.isChecked(),
            manifest_path=manifest_path,
            max_frames=max_frames,
            background_color=self._background_color,
            remove_black_background=self.remove_black_checkbox.isChecked(),
            chroma_key_color=self._chroma_color,
            chroma_key_tolerance=self.chroma_tolerance.value(),
            auto_edge_cutout=self.auto_edge_checkbox.isChecked(),
            padding=self.padding_input.value(),
            output_pattern=self.output_pattern_input.text().strip() or None,
        )
        logger.debug("Collected settings: %s", settings)
        return settings

    def _choose_color(self) -> None:
        """Open a color dialog for background selection."""

        color = QColorDialog.getColor(parent=self)
        if color.isValid():
            self._background_color = (color.red(), color.green(), color.blue(), color.alpha())
            self.background_color_label.setText(
                f"Current: rgba({color.red()},{color.green()},{color.blue()},{color.alpha()})"
            )
        else:
            self._background_color = None
            self.background_color_label.setText("Current: transparent")

    def _choose_chroma_color(self) -> None:
        """Open a color dialog for key color selection."""

        color = QColorDialog.getColor(parent=self)
        if color.isValid():
            self._chroma_color = (color.red(), color.green(), color.blue(), color.alpha())
            self._update_chroma_label()
        else:
            self._chroma_color = (0, 0, 0, 255)
            self._update_chroma_label()
        self.changed.emit()

    def set_chroma_color(self, color: tuple[int, int, int, int]) -> None:
        """Allow external callers to set the key color (e.g., eyedropper)."""

        self._chroma_color = color
        self._update_chroma_label()
        self.changed.emit()

    def _update_chroma_label(self) -> None:
        r, g, b, a = self._chroma_color or (0, 0, 0, 255)
        self.chroma_color_label.setText(f"Key: rgba({r},{g},{b},{a})")

    def sizeHint(self) -> QSize:  # pragma: no cover - Qt paints this
        return QSize(360, 400)

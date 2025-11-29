"""Detached preview window for spritesheet and frames."""

from __future__ import annotations

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QDialog


class PreviewDialog(QDialog):
    """Simple dialog to show a larger preview."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preview")
        self._base_size = QSize(800, 600)
        self.resize(self._base_size)
        self.label = QLabel("Preview", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMouseTracking(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        self._last_pixmap: QPixmap | None = None

    def mousePressEvent(self, event) -> None:  # pragma: no cover - UI callback
        self.parent().handle_preview_click(event) if hasattr(self.parent(), "handle_preview_click") else None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # pragma: no cover - UI callback
        self.parent().handle_preview_hover(event, self._last_pixmap) if hasattr(self.parent(), "handle_preview_hover") else None
        super().mouseMoveEvent(event)

    def update_pixmap(self, pixmap: QPixmap) -> None:
        target = self._base_size
        scaled = pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._last_pixmap = scaled
        self.label.setPixmap(scaled)

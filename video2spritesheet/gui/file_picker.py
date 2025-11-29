"""Native file picker helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QFileDialog, QWidget


def open_video_file_dialog(parent: QWidget) -> Optional[Path]:
    """Open a native file dialog and return the selected video path."""

    dialog = QFileDialog(parent, caption="Select Video")
    dialog.setFileMode(QFileDialog.ExistingFile)
    dialog.setNameFilters([
        "Video Files (*.mp4 *.mov *.avi *.mkv *.webm)",
        "All Files (*.*)",
    ])
    if dialog.exec():
        selected = dialog.selectedFiles()
        if selected:
            return Path(selected[0])
    return None

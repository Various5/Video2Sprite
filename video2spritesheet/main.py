"""Entry point for the Video2SpriteSheet GUI application."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from .gui.main_window import MainWindow


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )


def run() -> int:
    """Start the Qt event loop."""

    configure_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("Video2SpriteSheet")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(run())

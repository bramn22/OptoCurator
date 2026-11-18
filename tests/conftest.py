"""Pytest fixtures for OptoCurator tests."""

import os
import sys
import pytest
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox

# Run Qt headlessly in tests
os.environ["QT_QPA_PLATFORM"] = "offscreen"


@pytest.fixture(scope="session")
def qapp():
    """Ensure a single QApplication instance exists for GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    yield app


@pytest.fixture(autouse=True)
def no_gui_dialogs(monkeypatch):
    """Prevent GUI modal dialogs from blocking headless test execution."""
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "about", lambda *args, **kwargs: None)
    monkeypatch.setattr(QDialog, "exec", lambda self: 1)


@pytest.fixture(autouse=True)
def _cleanup_qt_widgets():
    """Close leftover top-level Qt widgets and flush deferred deletions after each test."""
    yield
    app = QApplication.instance()
    if app is not None:
        for w in list(app.topLevelWidgets()):
            try:
                w.close()
                w.deleteLater()
            except (RuntimeError, AttributeError):
                pass
        app.processEvents()


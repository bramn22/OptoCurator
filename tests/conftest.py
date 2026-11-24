"""Pytest fixtures for OptoCurator tests."""

import os
import sys
import time
from typing import Any, Callable, Optional
import pytest
from PyQt6.QtCore import QThread, QThreadPool, QTimer
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox

# Run Qt headlessly in tests
os.environ["QT_QPA_PLATFORM"] = "offscreen"


class _SignalBlocker:
    """Context manager / waiter that captures a Qt signal emission."""

    def __init__(self, signal: Any, timeout: int = 10000, raising: bool = True) -> None:
        self.signal = signal
        self.timeout = int(timeout)
        self.raising = bool(raising)
        self.args: list[Any] = []
        self.signal_triggered: bool = False
        self._connected: bool = False
        self.signal.connect(self._on_signal)
        self._connected = True

    def _on_signal(self, *args: Any) -> None:
        self.args = list(args)
        self.signal_triggered = True

    def __enter__(self) -> "_SignalBlocker":
        if not self._connected:
            self.signal.connect(self._on_signal)
            self._connected = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_type is not None:
            self._disconnect()
            return False
        self.wait()
        return False

    def _disconnect(self) -> None:
        if self._connected:
            try:
                self.signal.disconnect(self._on_signal)
            except (TypeError, RuntimeError):
                pass
            self._connected = False

    def wait(self) -> None:
        """Wait until the connected signal fires or timeout expires."""
        app = QApplication.instance()
        deadline = time.monotonic() + max(0, self.timeout) / 1000.0
        try:
            while not self.signal_triggered:
                if app is not None:
                    app.processEvents()
                if self.signal_triggered:
                    break
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.01)
            if app is not None:
                app.processEvents()
        finally:
            self._disconnect()

        if not self.signal_triggered and self.raising:
            raise TimeoutError(f"Signal did not fire within {self.timeout} ms")


class QtBot:
    """Minimal pytest-qt compatible helper for headless PyQt6 tests."""

    def __init__(self, app: Optional[QApplication] = None) -> None:
        self._app = app or QApplication.instance()

    def _process_events(self) -> None:
        app = self._app or QApplication.instance()
        if app is not None:
            app.processEvents()

    def wait(self, ms: int) -> None:
        """Process Qt events for `ms` milliseconds."""
        deadline = time.monotonic() + max(0, ms) / 1000.0
        while time.monotonic() < deadline:
            self._process_events()
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(0.01, remaining))
        self._process_events()

    def waitUntil(self, callback: Callable[[], Any], timeout: int = 10000) -> None:
        """Poll `QApplication.processEvents()` until `callback()` returns truthy or timeout expires."""
        deadline = time.monotonic() + max(0, timeout) / 1000.0
        last_assertion: Optional[AssertionError] = None
        while True:
            self._process_events()
            try:
                result = callback()
                if result:
                    return
                last_assertion = None
            except AssertionError as exc:
                last_assertion = exc

            if time.monotonic() >= deadline:
                if last_assertion is not None:
                    raise last_assertion
                raise TimeoutError(f"Condition not met within {timeout} ms")
            time.sleep(0.01)

    def waitSignal(self, signal: Any, timeout: int = 10000, raising: bool = True) -> _SignalBlocker:
        """Return a context manager that waits for `signal` to emit within `timeout` ms."""
        return _SignalBlocker(signal=signal, timeout=timeout, raising=raising)


@pytest.fixture(scope="session")
def qapp():
    """Ensure a single QApplication instance exists for GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    yield app


@pytest.fixture
def qtbot(qapp):
    """Provide a minimal pytest-qt compatible `qtbot` fixture for async Qt tests."""
    return QtBot(qapp)


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
    """Close leftover top-level Qt widgets and stop background workers/timers after each test."""
    yield
    app = QApplication.instance()
    if app is not None:
        had_widgets = False
        for w in list(app.topLevelWidgets()):
            try:
                if w.property("_qt_cleaned_up") and not w.isVisible():
                    continue
                had_widgets = True
                for timer in w.findChildren(QTimer):
                    if timer.isActive():
                        timer.stop()
                if hasattr(w, "_reset_background_prewarm"):
                    w._reset_background_prewarm(clear_pixmaps=False)
                if hasattr(w, "clustering_view") and hasattr(w.clustering_view, "_reset_background_prewarm"):
                    w.clustering_view._reset_background_prewarm(clear_pixmaps=False)
                for thread in list(w.findChildren(QThread)):
                    if thread.isRunning():
                        try:
                            if hasattr(thread, "results_ready"):
                                thread.results_ready.disconnect()
                        except Exception:
                            pass
                        thread.wait(5000)
                w.setProperty("_qt_cleaned_up", True)
                w.close()
                w.deleteLater()
            except (RuntimeError, AttributeError):
                pass
        if had_widgets:
            QThreadPool.globalInstance().waitForDone(5000)
            app.processEvents()


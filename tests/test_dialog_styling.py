"""Tests for dialog and button styling visibility across views."""

import pytest
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QInputDialog,
    QMessageBox,
    QPushButton,
)

from optocurator.ui.app import MainWindow
from optocurator.ui.theme import (
    COLOR_ACCENT,
    COLOR_BORDER_STRONG,
    COLOR_PANEL,
    COLOR_TEXT,
    MAIN_STYLE,
)
from optocurator.ui.view_clusters import ClusteringView
from optocurator.ui.view_detection import DetectionView
from optocurator.ui.view_launcher import LauncherDashboardWidget


def test_workspaces_no_unscoped_stylesheet(qapp):
    """Ensure workspaces do not apply un-scoped stylesheets that cascade onto children."""
    window = MainWindow()

    # centralStack
    assert window.stack.styleSheet() == "" or "background-color" not in window.stack.styleSheet()

    # Launcher view
    launcher = window.launcher_view
    assert launcher.styleSheet() == "" or "background-color" not in launcher.styleSheet()

    # Detection view
    detection = window.detection_view
    assert detection.styleSheet() == "" or "background-color" not in detection.styleSheet()

    # Clustering view
    clustering = window.clustering_view
    assert clustering.styleSheet() == "" or "background-color" not in clustering.styleSheet()


def test_qapplication_has_main_style(qapp):
    """Ensure QApplication has MAIN_STYLE set globally so modal dialogs are styled."""
    window = MainWindow()
    app = QApplication.instance()
    assert app is not None
    assert app.styleSheet() == MAIN_STYLE


def test_dialog_button_rules_in_main_style():
    """Verify MAIN_STYLE contains crisp dialog and button box styling rules."""
    assert "QDialog, QMessageBox, QInputDialog" in MAIN_STYLE
    assert "QPushButton" in MAIN_STYLE
    assert "QPushButton#primary_btn" in MAIN_STYLE
    assert "QPushButton#secondaryButton" in MAIN_STYLE
    assert COLOR_ACCENT in MAIN_STYLE
    assert COLOR_PANEL in MAIN_STYLE
    assert COLOR_TEXT in MAIN_STYLE
    assert COLOR_BORDER_STRONG in MAIN_STYLE


def test_child_dialog_buttons_parented_to_detection_view(qapp):
    """Verify buttons in child dialogs parented to DetectionView are properly configured."""
    window = MainWindow()
    detection = window.detection_view

    # DetectionView inline stylesheet must be empty so no cascade white background occurs
    assert detection.styleSheet() == ""

    # Create dialog parented to detection workspace
    dialog = QDialog(detection)
    btn_box = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        parent=dialog,
    )
    ok_btn = btn_box.button(QDialogButtonBox.StandardButton.Ok)
    cancel_btn = btn_box.button(QDialogButtonBox.StandardButton.Cancel)

    assert ok_btn is not None
    assert cancel_btn is not None
    ok_btn.setDefault(True)
    assert ok_btn.isDefault()
    assert not cancel_btn.isDefault()

    # Ensure dialog has no inherited unscoped stylesheet
    assert dialog.styleSheet() == ""


def test_message_box_parented_to_detection_view(qapp):
    """Verify QMessageBox parented to DetectionView does not inherit white-on-white button."""
    window = MainWindow()
    detection = window.detection_view

    msg_box = QMessageBox(detection)
    msg_box.setText("Session Export Complete")
    msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg_box.setDefaultButton(QMessageBox.StandardButton.Ok)

    ok_btn = msg_box.button(QMessageBox.StandardButton.Ok)
    assert ok_btn is not None
    assert ok_btn.isDefault()
    assert msg_box.styleSheet() == ""


def test_detection_view_export_button_style(qapp):
    """Verify the detection view Save button is styled with DETECTION_SAVE_BTN_STYLE (primary accent, § 13.6)."""
    from optocurator.ui.theme import DETECTION_SAVE_BTN_STYLE

    detection = DetectionView()
    assert detection.btn_export.text() == "Save"
    assert detection.btn_export.styleSheet() == DETECTION_SAVE_BTN_STYLE
    # Ensure no hardcoded green background
    assert "#059669" not in detection.btn_export.styleSheet()
    assert "#047857" not in detection.btn_export.styleSheet()


def test_detection_view_export_progress_dialog(qapp, tmp_path, monkeypatch):
    """Verify that exporting detection session presents a QProgressDialog styled like experiment loading."""
    from PyQt6.QtWidgets import QProgressDialog

    created_dialogs = []
    original_init = QProgressDialog.__init__

    def mock_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created_dialogs.append(self)

    monkeypatch.setattr(QProgressDialog, "__init__", mock_init)

    detection = DetectionView(load_mock=True)
    export_dir = tmp_path / "export_test"
    csv_path = detection.export_curated_session(export_dir=export_dir, show_dialog=False)

    assert csv_path is not None
    assert len(created_dialogs) >= 1
    dlg = created_dialogs[0]
    assert dlg.windowTitle() == "Exporting Session"
    # Verify same style properties as load experiment loading bar
    assert "QProgressDialog" in dlg.styleSheet()
    assert COLOR_BORDER_STRONG in dlg.styleSheet()
    assert "QProgressBar::chunk" in dlg.styleSheet()
    assert COLOR_ACCENT in dlg.styleSheet()

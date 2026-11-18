"""Tests for Work Package 10 (WP10 — Launcher and cross-mode consistency) from UI_OVERHAUL_SPEC.md (§ 1.1, § 6, § 14 WP10)."""

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QMessageBox

from optocurator.ui.app import MainWindow
from optocurator.ui.plots import SiliconProbeWidget
from optocurator.ui.splitters import ResettableSplitter
from optocurator.ui.view_launcher import LauncherDashboardWidget


def test_wp10_launcher_restyle_and_layout(qapp):
    """Verify launcher initial state, card navigation, and design-system styling."""
    win = MainWindow()
    win.resize(1280, 800)

    # App launches on VIEW_LAUNCHER with header_bar and status bar hidden
    assert win.stack.currentIndex() == MainWindow.VIEW_LAUNCHER
    assert win.header_bar.isHidden()
    assert win.status.isHidden()

    # LauncherDashboardWidget has detection_card and clustering_card
    launcher = win.launcher_view
    assert isinstance(launcher, LauncherDashboardWidget)
    assert hasattr(launcher, "detection_card")
    assert hasattr(launcher, "clustering_card")

    # Launcher uses design-system styles (no hardcoded background-color override on launcher_view)
    assert launcher.objectName() == "launcherDashboard"
    assert "background-color" not in launcher.styleSheet().lower()

    # Clicking detection_card via mouse press switches to VIEW_DETECTION
    press_ev = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(10.0, 10.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    launcher.detection_card.mousePressEvent(press_ev)
    qapp.processEvents()
    assert win.stack.currentIndex() == MainWindow.VIEW_DETECTION
    assert not win.header_bar.isHidden()
    assert not win.status.isHidden()

    # Return home and click clustering_card -> switches to VIEW_CLUSTERING
    win.btn_home.click()
    qapp.processEvents()
    assert win.stack.currentIndex() == MainWindow.VIEW_LAUNCHER

    launcher.clustering_card.mousePressEvent(press_ev)
    qapp.processEvents()
    assert win.stack.currentIndex() == MainWindow.VIEW_CLUSTERING
    assert not win.header_bar.isHidden()
    assert not win.status.isHidden()

    win.close()


def test_wp10_home_button_clean_and_dirty_save_prompt_detection(qapp, monkeypatch):
    """Verify Home button tooltip, clean return, and dirty save prompt (Cancel/Discard/Save) in Detection."""
    win = MainWindow()
    assert win.btn_home.toolTip() == "Home"

    # Clean return from Detection
    win.switch_to_detection()
    win.detection_view._set_dirty(False)
    win.btn_home.click()
    qapp.processEvents()
    assert win.stack.currentIndex() == MainWindow.VIEW_LAUNCHER

    # Dirty state -> Cancel keeps user in Detection
    win.switch_to_detection()
    win.detection_view._set_dirty(True)
    prompted = []

    def mock_cancel(*args, **kwargs):
        prompted.append("cancel")
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", mock_cancel)
    win.btn_home.click()
    qapp.processEvents()
    assert prompted == ["cancel"]
    assert win.stack.currentIndex() == MainWindow.VIEW_DETECTION
    assert win.detection_view._is_dirty is True

    # Dirty state -> Discard switches to VIEW_LAUNCHER
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Discard,
    )
    win.btn_home.click()
    qapp.processEvents()
    assert win.stack.currentIndex() == MainWindow.VIEW_LAUNCHER

    # Dirty state -> Save triggers _on_save_session(); stays if still dirty (e.g. Save As cancelled)
    win.switch_to_detection()
    win.detection_view._set_dirty(True)
    save_calls = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Save,
    )
    monkeypatch.setattr(
        win,
        "_on_save_session",
        lambda *args, **kwargs: save_calls.append("still_dirty"),
    )
    win.btn_home.click()
    qapp.processEvents()
    assert save_calls == ["still_dirty"]
    assert win.stack.currentIndex() == MainWindow.VIEW_DETECTION

    # Dirty state -> Save clears dirty -> switches to VIEW_LAUNCHER
    def save_and_clean(*args, **kwargs):
        save_calls.append("saved")
        win.detection_view._set_dirty(False)

    monkeypatch.setattr(win, "_on_save_session", save_and_clean)
    win.btn_home.click()
    qapp.processEvents()
    assert save_calls == ["still_dirty", "saved"]
    assert win.stack.currentIndex() == MainWindow.VIEW_LAUNCHER

    win.close()


def test_wp10_home_button_clean_and_dirty_save_prompt_clustering(qapp, monkeypatch):
    """Verify Home button clean return and dirty save prompt (Cancel/Discard/Save) in Clustering."""
    win = MainWindow()

    # Clean return from Clustering
    win.switch_to_clustering()
    win.clustering_view._set_dirty(False)
    win.btn_home.click()
    qapp.processEvents()
    assert win.stack.currentIndex() == MainWindow.VIEW_LAUNCHER

    # Dirty state -> Cancel keeps user in Clustering
    win.switch_to_clustering()
    win.clustering_view._set_dirty(True)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Cancel,
    )
    win.btn_home.click()
    qapp.processEvents()
    assert win.stack.currentIndex() == MainWindow.VIEW_CLUSTERING
    assert win.clustering_view._is_dirty is True

    # Dirty state -> Discard switches to VIEW_LAUNCHER
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Discard,
    )
    win.btn_home.click()
    qapp.processEvents()
    assert win.stack.currentIndex() == MainWindow.VIEW_LAUNCHER

    # Dirty state -> Save triggers _on_save_session(); stays if still dirty
    win.switch_to_clustering()
    win.clustering_view._set_dirty(True)
    save_calls = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Save,
    )
    monkeypatch.setattr(
        win,
        "_on_save_session",
        lambda *args, **kwargs: save_calls.append("still_dirty"),
    )
    win.btn_home.click()
    qapp.processEvents()
    assert save_calls == ["still_dirty"]
    assert win.stack.currentIndex() == MainWindow.VIEW_CLUSTERING

    # Dirty state -> Save clears dirty -> switches to VIEW_LAUNCHER
    def save_and_clean(*args, **kwargs):
        save_calls.append("saved")
        win.clustering_view._set_dirty(False)

    monkeypatch.setattr(win, "_on_save_session", save_and_clean)
    win.btn_home.click()
    qapp.processEvents()
    assert save_calls == ["still_dirty", "saved"]
    assert win.stack.currentIndex() == MainWindow.VIEW_LAUNCHER

    win.close()


def test_wp10_shared_components_across_modes(qapp):
    """Verify both Detection and Clustering share SiliconProbeWidget and ResettableSplitter."""
    win = MainWindow()

    assert isinstance(win.detection_view.probe_widget, SiliconProbeWidget)
    assert isinstance(win.clustering_view.probe_widget, SiliconProbeWidget)
    assert win.detection_view.probe_widget is not win.clustering_view.probe_widget

    assert isinstance(win.detection_view.main_splitter, ResettableSplitter)
    assert isinstance(win.clustering_view.main_splitter, ResettableSplitter)
    assert isinstance(win.clustering_view.right_splitter, ResettableSplitter)

    win.close()


def test_wp10_cross_mode_state_isolation(qapp):
    """Verify Detection and Clustering maintain independent unit, focus, and dirty state across mode switches."""
    win = MainWindow()
    dv = win.detection_view
    cv = win.clustering_view

    # Load mock data into both views independently
    win.switch_to_detection()
    dv._load_mock_data()
    qapp.processEvents()

    det_row_count = dv.table_units.rowCount()
    assert det_row_count == 5

    # Focus unit at row 2 (u0003, initially untagged) and toggle its tag -> sets dirty=True
    dv.table_units.selectRow(2)
    qapp.processEvents()
    focused_uid = dv._focused_unit_id
    assert focused_uid is not None
    was_tagged = dv._is_row_tagged(2)
    dv.btn_toggle_tag.click()
    qapp.processEvents()

    assert dv._is_dirty is True
    assert dv._is_row_tagged(2) is (not was_tagged)

    # Switch to Clustering via header mode switch button and load mock data there
    win.btn_mode_clustering.click()
    qapp.processEvents()
    assert win.stack.currentIndex() == MainWindow.VIEW_CLUSTERING

    cv._load_mock_data()
    qapp.processEvents()
    clust_row_count = cv.table_pooled_units.rowCount()
    clust_unit_ids = list(cv._pooled_units_map.keys())
    assert clust_row_count == 9
    assert len(clust_unit_ids) == 9
    assert cv._is_dirty is False

    # Switch back to Detection via header mode switch button
    win.btn_mode_detection.click()
    qapp.processEvents()
    assert win.stack.currentIndex() == MainWindow.VIEW_DETECTION

    # Verify Detection state is completely preserved
    assert dv.table_units.rowCount() == det_row_count
    assert dv._focused_unit_id == focused_uid
    assert dv._is_dirty is True
    assert dv._is_row_tagged(2) is (not was_tagged)

    # Switch to Clustering again and verify its state is completely preserved and isolated
    win.btn_mode_clustering.click()
    qapp.processEvents()
    assert win.stack.currentIndex() == MainWindow.VIEW_CLUSTERING
    assert cv.table_pooled_units.rowCount() == clust_row_count
    assert list(cv._pooled_units_map.keys()) == clust_unit_ids
    assert cv._is_dirty is False
    assert dv.units_by_id is not cv._pooled_units_map

    win.close()

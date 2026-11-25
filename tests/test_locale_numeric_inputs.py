"""Regression tests for strict C-locale decimal-separator enforcement across Qt inputs."""

import pytest
from PyQt6.QtCore import QEvent, QLocale, Qt
from PyQt6.QtGui import QFocusEvent, QKeyEvent
from PyQt6.QtWidgets import QApplication, QDoubleSpinBox, QLineEdit, QVBoxLayout, QWidget

from optocurator.ui.app import MainWindow, configure_c_locale
from optocurator.ui.detection_widgets import DetectionCriterionRow


def _send_key(widget: QWidget, key: Qt.Key, text: str = "") -> None:
    """Send a KeyPress + KeyRelease pair to a widget headlessly."""
    press = QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier, text)
    release = QKeyEvent(QEvent.Type.KeyRelease, key, Qt.KeyboardModifier.NoModifier, text)
    QApplication.sendEvent(widget, press)
    QApplication.sendEvent(widget, release)


def _type_text(widget: QWidget, text: str) -> None:
    """Simulate typing each character of `text` into a QLineEdit via key events."""
    for ch in text:
        if ch == ".":
            key = Qt.Key.Key_Period
        elif ch == ",":
            key = Qt.Key.Key_Comma
        elif ch == "-":
            key = Qt.Key.Key_Minus
        elif "0" <= ch <= "9":
            key = Qt.Key(int(Qt.Key.Key_0) + int(ch))
        else:
            key = Qt.Key.Key_unknown
        _send_key(widget, key, ch)


def test_detection_criterion_row_delete_leading_digit_under_comma_locale(qapp: QApplication) -> None:
    """Under nl_BE (comma decimal), deleting '8' at index 0 of '8.0' succeeds and leaves '.0'."""
    orig_locale = QLocale()
    try:
        QLocale.setDefault(QLocale("nl_BE"))

        row = DetectionCriterionRow(
            label="Latency",
            operator="≤",
            unit="ms",
            axis_range=(0.0, 20.0),
            default_threshold=8.0,
            min_val=0.0,
            max_val=50.0,
            decimals=1,
        )
        assert row.edit_threshold.text() == "8.0"

        row.edit_threshold.setCursorPosition(0)
        _send_key(row.edit_threshold, Qt.Key.Key_Delete)
        qapp.processEvents()

        assert row.edit_threshold.text() == ".0"
    finally:
        QLocale.setDefault(orig_locale)


def test_detection_criterion_row_typing_decimal_and_focus_out_under_comma_locale(
    qapp: QApplication,
) -> None:
    """Under nl_BE, typing '3.0' and moving focus preserves '3.0' and threshold == 3.0 (not 30.0)."""
    orig_locale = QLocale()
    try:
        QLocale.setDefault(QLocale("nl_BE"))

        container = QWidget()
        layout = QVBoxLayout(container)
        row = DetectionCriterionRow(
            label="Latency",
            operator="≤",
            unit="ms",
            axis_range=(0.0, 20.0),
            default_threshold=8.0,
            min_val=0.0,
            max_val=50.0,
            decimals=1,
            parent=container,
        )
        other_input = QLineEdit(container)
        layout.addWidget(row)
        layout.addWidget(other_input)
        container.show()
        qapp.processEvents()

        row.edit_threshold.setFocus()
        qapp.processEvents()
        row.edit_threshold.selectAll()
        _send_key(row.edit_threshold, Qt.Key.Key_Backspace)
        _type_text(row.edit_threshold, "3.0")
        assert row.edit_threshold.text() == "3.0"

        # Move focus to another widget (triggers editingFinished)
        other_input.setFocus()
        QApplication.sendEvent(row.edit_threshold, QFocusEvent(QEvent.Type.FocusOut))
        qapp.processEvents()

        assert row.edit_threshold.text() == "3.0"
        assert row.threshold() == pytest.approx(3.0)
        assert row.threshold() != pytest.approx(30.0)
    finally:
        QLocale.setDefault(orig_locale)


def test_detection_criterion_row_rejects_comma_under_comma_locale(qapp: QApplication) -> None:
    """Under nl_BE, typing a comma (e.g. '4,5') strictly rejects ',' (periods only)."""
    orig_locale = QLocale()
    try:
        QLocale.setDefault(QLocale("nl_BE"))

        row = DetectionCriterionRow(
            label="Latency",
            operator="≤",
            unit="ms",
            axis_range=(0.0, 20.0),
            default_threshold=8.0,
            min_val=0.0,
            max_val=50.0,
            decimals=1,
        )
        row.edit_threshold.selectAll()
        _send_key(row.edit_threshold, Qt.Key.Key_Backspace)
        _type_text(row.edit_threshold, "4,5")
        qapp.processEvents()

        assert "," not in row.edit_threshold.text()
        assert row.edit_threshold.text() == "45"
    finally:
        QLocale.setDefault(orig_locale)


def test_configure_c_locale_sets_c_locale_and_double_spinbox_uses_period(qapp: QApplication) -> None:
    """configure_c_locale() sets C locale with RejectGroupSeparator and QDoubleSpinBox uses '.'."""
    orig_locale = QLocale()
    try:
        QLocale.setDefault(QLocale("nl_BE"))
        assert QLocale().decimalPoint() == ","

        configure_c_locale()

        default_loc = QLocale()
        assert default_loc.language() == QLocale.Language.C
        assert default_loc.decimalPoint() == "."
        assert bool(default_loc.numberOptions() & QLocale.NumberOption.RejectGroupSeparator)

        spin = QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setRange(0.0, 100.0)
        spin.setValue(3.25)
        qapp.processEvents()

        assert spin.textFromValue(3.25) == "3.25"
        assert spin.cleanText() == "3.25"
        assert "," not in spin.cleanText()
        assert spin.valueFromText("4.75") == pytest.approx(4.75)
    finally:
        QLocale.setDefault(orig_locale)


def test_main_window_enforces_c_locale_for_spinboxes_under_comma_locale(qapp: QApplication) -> None:
    """Instantiating MainWindow under QLocale('nl_BE') enforces C locale so its spinboxes use '.'."""
    orig_locale = QLocale()
    try:
        QLocale.setDefault(QLocale("nl_BE"))
        assert QLocale().decimalPoint() == ","

        win = MainWindow()
        qapp.processEvents()

        assert QLocale().language() == QLocale.Language.C
        assert QLocale().decimalPoint() == "."
        assert bool(QLocale().numberOptions() & QLocale.NumberOption.RejectGroupSeparator)

        spinboxes = win.findChildren(QDoubleSpinBox)
        assert len(spinboxes) > 0, "Expected MainWindow to contain at least one QDoubleSpinBox"
        for sb in spinboxes:
            assert sb.locale().decimalPoint() == "."
            assert "," not in sb.cleanText()
            assert sb.textFromValue(1.5).startswith("1.5")
    finally:
        QLocale.setDefault(orig_locale)

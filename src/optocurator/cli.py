"""Terminal CLI entry point for OptoCurator."""

import argparse
import sys
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from optocurator import __version__
from optocurator.ui.app import MainWindow


def main(argv: list[str] | None = None) -> int:
    """Launch the OptoCurator desktop application."""
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="optocurator",
        description="OptoCurator: Statistical Identification & Response Clustering for Optotagged Units",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Optional path to a recording session directory (Phy/Kilosort) or .nwb file to preload into Phase 1",
    )
    args = parser.parse_args(argv)

    app = QApplication(sys.argv[:1])
    app.setApplicationName("OptoCurator")
    app.setApplicationVersion(__version__)
    app.setFont(QFont("Segoe UI", 10))

    import pyqtgraph as pg
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor

    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")

    pal = app.palette()
    pal.setColor(pal.ColorRole.Window, Qt.GlobalColor.white)
    pal.setColor(pal.ColorRole.Base, Qt.GlobalColor.white)
    pal.setColor(pal.ColorRole.AlternateBase, QColor("#F8FAFC"))
    pal.setColor(pal.ColorRole.WindowText, QColor("#0F172A"))
    pal.setColor(pal.ColorRole.Text, QColor("#0F172A"))
    pal.setColor(pal.ColorRole.Button, Qt.GlobalColor.white)
    pal.setColor(pal.ColorRole.ButtonText, QColor("#0F172A"))
    app.setPalette(pal)

    from optocurator.ui.theme import MAIN_STYLE
    app.setStyleSheet(MAIN_STYLE)

    window = MainWindow()

    if args.experiment:
        from pathlib import Path
        window.switch_to_detection()
        exp_path = Path(args.experiment)
        if exp_path.is_file() and exp_path.suffix.lower() == ".nwb":
            window.detection_view.load_nwb_file(exp_path)
        elif exp_path.is_dir():
            window.detection_view.load_experiment_folder(exp_path)
        else:
            window.detection_view.lbl_session.setText(f"Session: {args.experiment}")
        window.showMaximized()
    else:
        window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

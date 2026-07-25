"""CLI entry: launch the Streamlit UI."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Run Streamlit against infrarag.ui.app."""
    try:
        from streamlit.web import cli as stcli
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "streamlit is required to launch the UI. Install with: pip install -e ."
        ) from exc

    app_path = Path(__file__).resolve().parent / "ui" / "app.py"
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true",
    ]
    stcli.main()


if __name__ == "__main__":
    main()

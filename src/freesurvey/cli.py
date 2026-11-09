"""Command line helpers for the cleaned v01 build."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from . import __version__


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_gui(_args: argparse.Namespace) -> int:
    from .gui import launch_gui

    launch_gui()
    return 0


def _validate_data(args: argparse.Namespace) -> int:
    folder = Path(args.data_folder)
    station_files = sorted(folder.glob("*.TXT"))
    if not station_files:
        print(f"No .TXT station files found in {folder}", file=sys.stderr)
        return 2

    from .data_manager import DataManager

    manager = DataManager()
    survey_data = manager.load_station_folder(folder)
    if not manager.validate_survey_data(survey_data):
        print("Station folder validation failed.", file=sys.stderr)
        return 2

    print(f"Validation passed: {len(station_files)} station files.")
    return 0


def _audit(args: argparse.Namespace) -> int:
    script = _project_root() / "analyze_methods_data.py"
    if not script.exists():
        print(f"Audit script not found: {script}", file=sys.stderr)
        return 2

    audit_args = list(args.audit_args)
    if audit_args[:1] == ["--"]:
        audit_args = audit_args[1:]
    cmd = [sys.executable, str(script), *audit_args]
    return subprocess.call(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="georede-v01")
    parser.add_argument("--version", action="version", version=f"georede-v01 {__version__}")
    sub = parser.add_subparsers(dest="command")

    gui_parser = sub.add_parser("gui", help="open the desktop interface")
    gui_parser.set_defaults(func=_run_gui)

    validate_parser = sub.add_parser("validate-data", help="validate a station TXT folder")
    validate_parser.add_argument("data_folder")
    validate_parser.set_defaults(func=_validate_data)

    audit_parser = sub.add_parser("audit", help="run analyze_methods_data.py")
    audit_parser.add_argument("audit_args", nargs=argparse.REMAINDER)
    audit_parser.set_defaults(func=_audit)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

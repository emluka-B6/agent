#!/usr/bin/env python3
"""Launch the local web viewer/editor for generated plan artifacts.

The viewer is a small FastAPI server. In development the React frontend runs
under Vite (``cd frontend && npm run dev``); in production it is built into
``issue_planner/web/dist`` and served by this same process.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from issue_planner.config import DEFAULT_OUTPUT_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="viewer.py",
        description=(
            "Serve a browser UI for browsing, analyzing, and editing the plans "
            "produced by agent.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python viewer.py\n"
            "  python viewer.py --plans-dir plans --port 8080\n"
            "  PLANS_DIR=plans python viewer.py\n"
        ),
    )
    parser.add_argument(
        "--plans-dir",
        type=Path,
        default=Path(os.getenv("PLANS_DIR") or DEFAULT_OUTPUT_DIR),
        help=(
            "Directory that contains plan JSON files "
            f"(default: $PLANS_DIR or {DEFAULT_OUTPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print(
            "error: uvicorn is not installed. Run `pip install -r requirements.txt`.",
            file=sys.stderr,
        )
        return 2

    from issue_planner.web.server import create_app

    plans_dir = args.plans_dir.expanduser().resolve()
    app = create_app(plans_dir)
    print(f"Serving plans from {plans_dir}")
    print(f"Open http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

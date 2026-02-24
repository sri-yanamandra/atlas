"""Command-line interface for .cohort file operations.

Usage::

    cohort inspect <file> [--json]
    cohort merge <out> <file1> <file2> [...]  [--name NAME]
    cohort intersect <out> <file1> <file2> [...] --on COLUMN [--name NAME]
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cohort",
        description="CLI for .cohort binary container files",
    )
    sub = parser.add_subparsers(dest="command")

    # ── inspect ────────────────────────────────────────────────
    p_inspect = sub.add_parser("inspect", help="Print metadata for a .cohort file")
    p_inspect.add_argument("file", help="Path to the .cohort file")
    p_inspect.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output machine-readable JSON instead of human-readable text",
    )

    # ── merge (union) ─────────────────────────────────────────
    p_merge = sub.add_parser("merge", help="Union (concatenate rows) of multiple .cohort files")
    p_merge.add_argument("out", help="Output .cohort path")
    p_merge.add_argument("files", nargs="+", help="Input .cohort files (at least two)")
    p_merge.add_argument("--name", default=None, help="Override the cohort name in the output")

    # ── intersect ─────────────────────────────────────────────
    p_intersect = sub.add_parser("intersect", help="Intersect rows across multiple .cohort files")
    p_intersect.add_argument("out", help="Output .cohort path")
    p_intersect.add_argument("files", nargs="+", help="Input .cohort files (at least two)")
    p_intersect.add_argument("--on", required=True, help="Column name to intersect on")
    p_intersect.add_argument("--name", default=None, help="Override the cohort name in the output")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    import cohort_sdk

    if args.command == "inspect":
        result = cohort_sdk.inspect(args.file)
        if args.json_output:
            print(result.model_dump_json(indent=2))
        else:
            print(result)

    elif args.command == "merge":
        cohort_sdk.union(args.files, args.out, name=args.name)
        print(f"Merged {len(args.files)} files → {args.out}")

    elif args.command == "intersect":
        cohort_sdk.intersect(args.files, args.out, on=args.on, name=args.name)
        print(f"Intersected {len(args.files)} files → {args.out}")

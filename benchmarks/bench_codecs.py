"""Codec comparison benchmark for .cohort files.

Reads an existing .cohort file, re-encodes with each codec, and measures
write time, read time, file size, and SQL query performance.

Usage:
    python benchmarks/bench_codecs.py                          # largest .cohort, all codecs
    python benchmarks/bench_codecs.py --rows 1m                # first 1M rows only
    python benchmarks/bench_codecs.py --codecs parquet feather  # specific codecs
    python benchmarks/bench_codecs.py --source diag_i2.cohort  # specific file
    python benchmarks/bench_codecs.py --no-graphs              # skip chart generation
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from cohort_sdk import (
    CohortCatalog,
    CohortDefinition,
    CohortWriter,
    read as cohort_read,
    inspect as cohort_inspect,
)
from cohort_sdk.models import TagDirectory

# ── Helpers ───────────────────────────────────────────────────

ALL_CODECS = ["parquet", "arrow_ipc", "feather", "vortex"]

SCALE_SHORTCUTS = {
    "100k": 100_000, "500k": 500_000,
    "1m": 1_000_000, "5m": 5_000_000,
    "10m": 10_000_000, "20m": 20_000_000,
    "full": -1,
}


def fmt_ms(ms: float) -> str:
    if ms < 1:
        return f"{ms * 1000:.0f} us"
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{ms / 1000:.2f} s"


def fmt_rows(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def fmt_size(b: int) -> str:
    if b < 1024:
        return f"{b} B"
    if b < 1024 * 1024:
        return f"{b / 1024:.0f} KB"
    if b < 1024 * 1024 * 1024:
        return f"{b / (1024 * 1024):.1f} MB"
    return f"{b / (1024 * 1024 * 1024):.2f} GB"


def fmt_rate(rows: int, ms: float) -> str:
    if ms <= 0:
        return "-"
    per_sec = rows / (ms / 1000)
    if per_sec >= 1_000_000:
        return f"{per_sec / 1_000_000:.1f}M/s"
    if per_sec >= 1_000:
        return f"{per_sec / 1_000:.0f}K/s"
    return f"{per_sec:.0f}/s"


def timed(fn, iterations=1, warmup=0):
    """Run fn, return average elapsed ms."""
    for _ in range(warmup):
        fn()
    gc.disable()
    t0 = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = ((time.perf_counter() - t0) / iterations) * 1000
    gc.enable()
    return elapsed


# ── Load source data ──────────────────────────────────────────


def load_source(source: str | None, max_rows: int) -> tuple[pa.Table, str, str]:
    """Load source data from an existing .cohort file.

    Returns (table, dataset_label, source_name).
    """
    if source:
        path = Path(source)
    else:
        # Find the largest .cohort file in cwd
        cohort_files = sorted(
            Path(".").glob("*.cohort"),
            key=lambda f: f.stat().st_size,
            reverse=True,
        )
        if not cohort_files:
            print("  No .cohort files found in current directory.")
            print("  Use --source to specify a file, or run `make demo` first.")
            sys.exit(1)
        path = cohort_files[0]

    print(f"  Source: {path.name} ({fmt_size(path.stat().st_size)})")
    cohort = cohort_read(str(path))
    label = cohort.datasets[0]
    table = cohort[label]
    print(f"  Dataset: {label} — {fmt_rows(table.num_rows)} rows, {table.num_columns} cols")

    if 0 < max_rows < table.num_rows:
        table = table.slice(0, max_rows)
        print(f"  Sliced to {fmt_rows(max_rows)} rows (use --rows full for everything)")

    return table, label, path.name


# ── Benchmark runner ──────────────────────────────────────────


def bench_codec(
    codec: str,
    table: pa.Table,
    label: str,
    tmpdir: str,
    definition: CohortDefinition,
) -> dict:
    """Benchmark a single codec. Returns results dict."""
    n = table.num_rows
    path = os.path.join(tmpdir, f"bench_{codec}.cohort")
    result = {"codec": codec, "rows": n}

    # ── Write ─────────────────────────────────────────────
    def do_write():
        CohortWriter.write(
            path,
            TagDirectory(cohort_definition=definition),
            [(label, table)],
            codec=codec,
        )

    try:
        # First write to create the file
        do_write()
        # Then benchmark (fewer iterations for large data)
        iters = 3 if n <= 5_000_000 else 1
        ms = timed(do_write, iterations=iters)
        result["write_ms"] = round(ms, 1)
        result["file_size"] = os.path.getsize(path)
    except Exception as e:
        print(f"    [{codec}] write FAILED: {e}")
        result["write_ms"] = None
        result["file_size"] = None
        return result

    # ── Read ──────────────────────────────────────────────
    try:
        def do_read():
            return cohort_read(path)[label]

        iters = 3 if n <= 5_000_000 else 1
        ms = timed(do_read, iterations=iters, warmup=1)
        result["read_ms"] = round(ms, 1)
    except Exception as e:
        print(f"    [{codec}] read FAILED: {e}")
        result["read_ms"] = None

    # ── Validate round-trip correctness ───────────────────
    result["valid"] = False
    try:
        rt = cohort_read(path)[label]
        errors = []
        if rt.num_rows != n:
            errors.append(f"row count: expected {n}, got {rt.num_rows}")
        if rt.num_columns != table.num_columns:
            errors.append(f"col count: expected {table.num_columns}, got {rt.num_columns}")
        if rt.schema.names != table.schema.names:
            errors.append(f"column names differ")
        # Spot-check first row values
        if rt.num_rows > 0 and not errors:
            for col_name in table.schema.names[:5]:
                orig = table.column(col_name)[0].as_py()
                got = rt.column(col_name)[0].as_py()
                if orig != got:
                    errors.append(f"first row '{col_name}': expected {orig!r}, got {got!r}")
        if errors:
            result["valid"] = False
            result["validation_errors"] = errors
        else:
            result["valid"] = True
    except Exception as e:
        result["validation_errors"] = [str(e)]

    # ── Inspect (metadata only) ───────────────────────────
    try:
        ms = timed(lambda: cohort_inspect(path), iterations=20, warmup=1)
        result["inspect_ms"] = round(ms, 4)
    except Exception as e:
        result["inspect_ms"] = None

    # ── SQL COUNT(*) via DataFusion (parquet only — DataFusion can't read other codecs)
    if CohortCatalog is not None and codec == "parquet":
        try:
            sql_dir = os.path.join(tmpdir, f"sql_{codec}")
            os.makedirs(sql_dir, exist_ok=True)
            dst = os.path.join(sql_dir, f"bench.cohort")
            shutil.copy2(path, dst)
            CohortCatalog.init(sql_dir)
            catalog = CohortCatalog(sql_dir)
            tables = catalog.tables()
            if tables:
                tbl_name = tables[0]

                def do_count():
                    return catalog.query(f'SELECT COUNT(*) FROM "{tbl_name}"')

                ms = timed(do_count, iterations=3, warmup=1)
                result["sql_count_ms"] = round(ms, 1)

                # SQL filter
                first_col = table.schema.field(0).name
                def do_filter():
                    return catalog.query(
                        f'SELECT * FROM "{tbl_name}" WHERE "{first_col}" IS NOT NULL LIMIT 1000'
                    )

                ms = timed(do_filter, iterations=3, warmup=1)
                result["sql_filter_ms"] = round(ms, 1)
        except Exception as e:
            print(f"    [{codec}] SQL FAILED: {e}")
            result["sql_count_ms"] = None
            result["sql_filter_ms"] = None
    else:
        result["sql_count_ms"] = None
        result["sql_filter_ms"] = None

    return result


# ── DuckDB baseline ───────────────────────────────────────────


def bench_duckdb(table: pa.Table, tmpdir: str) -> dict:
    """Benchmark DuckDB as a speed reference."""
    n = table.num_rows
    result = {"codec": "duckdb", "rows": n}

    # Write to parquet for DuckDB to read
    pq_path = os.path.join(tmpdir, "duckdb_bench.parquet")
    pq.write_table(table, pq_path)
    result["file_size"] = os.path.getsize(pq_path)

    # Write (DuckDB COPY to parquet)
    db_path = os.path.join(tmpdir, "bench.duckdb")

    duckdb_out = os.path.join(tmpdir, "duckdb_out.parquet")
    def do_write():
        con = duckdb.connect()
        con.execute(f"COPY (SELECT * FROM read_parquet('{pq_path}')) TO '{duckdb_out}' (FORMAT PARQUET, CODEC 'ZSTD')")
        con.close()

    ms = timed(do_write, iterations=1)
    result["write_ms"] = round(ms, 1)

    # Full read (scan entire table into Arrow)
    def do_read():
        con = duckdb.connect()
        con.execute("SELECT * FROM read_parquet(?)", [pq_path]).arrow()
        con.close()

    ms = timed(do_read, iterations=3, warmup=1)
    result["read_ms"] = round(ms, 1)

    # COUNT(*)
    def do_count():
        con = duckdb.connect()
        con.execute("SELECT COUNT(*) FROM read_parquet(?)", [pq_path]).fetchone()
        con.close()

    ms = timed(do_count, iterations=10, warmup=1)
    result["sql_count_ms"] = round(ms, 4)

    # SQL filter
    first_col = table.schema.field(0).name
    def do_filter():
        con = duckdb.connect()
        con.execute(f'SELECT * FROM read_parquet(?) WHERE "{first_col}" IS NOT NULL LIMIT 1000',
                    [pq_path]).arrow()
        con.close()

    ms = timed(do_filter, iterations=5, warmup=1)
    result["sql_filter_ms"] = round(ms, 1)

    result["inspect_ms"] = None  # N/A for DuckDB
    result["valid"] = True       # DuckDB is the reference

    return result


# ── Display ───────────────────────────────────────────────────


def print_results(results: list[dict], source_name: str):
    n = results[0]["rows"]
    print(f"\n{'=' * 78}")
    print(f"  CODEC COMPARISON — {fmt_rows(n)} rows from {source_name}")
    print(f"{'=' * 78}\n")

    # Column widths
    cw = 14

    # Header
    hdr = f"  {'':18}"
    for r in results:
        hdr += f"  {r['codec']:>{cw}}"
    print(hdr)
    print(f"  {'':18}" + "  " + ("-" * (cw)) * len(results))

    # Rows
    def row(label, key, fmt_fn=fmt_ms, extra_fn=None):
        line = f"  {label:18}"
        vals = []
        for r in results:
            v = r.get(key)
            if v is None:
                line += f"  {'FAILED':>{cw}}"
                vals.append(None)
            else:
                cell = fmt_fn(v)
                if extra_fn:
                    cell += f" ({extra_fn(r)})"
                line += f"  {cell:>{cw}}"
                vals.append(v)
        # Mark winner
        valid = [(i, v) for i, v in enumerate(vals) if v is not None]
        if len(valid) > 1:
            winner_idx = min(valid, key=lambda x: x[1])[0]
            line += f"  <- {results[winner_idx]['codec']}"
        print(line)

    row("Write", "write_ms", extra_fn=lambda r: fmt_rate(r["rows"], r["write_ms"]) if r.get("write_ms") else "")
    row("Read", "read_ms", extra_fn=lambda r: fmt_rate(r["rows"], r["read_ms"]) if r.get("read_ms") else "")
    row("Inspect", "inspect_ms")
    row("SQL COUNT(*)", "sql_count_ms")
    row("SQL Filter", "sql_filter_ms")
    row("File Size", "file_size", fmt_fn=fmt_size)

    # Compression ratio relative to parquet
    pq = next((r for r in results if r["codec"] == "parquet"), None)
    if pq and pq.get("file_size"):
        print()
        line = f"  {'Size vs Parquet':18}"
        for r in results:
            if r.get("file_size") and pq["file_size"]:
                ratio = r["file_size"] / pq["file_size"]
                line += f"  {ratio:>{cw}.2f}x"
            else:
                line += f"  {'-':>{cw}}"
        print(line)

    # Speedup relative to parquet
    if pq:
        print()
        line = f"  {'Read vs Parquet':18}"
        for r in results:
            if r.get("read_ms") and pq.get("read_ms") and r["read_ms"] > 0:
                speedup = pq["read_ms"] / r["read_ms"]
                line += f"  {speedup:>{cw}.2f}x"
            else:
                line += f"  {'-':>{cw}}"
        print(line)

        line = f"  {'Write vs Parquet':18}"
        for r in results:
            if r.get("write_ms") and pq.get("write_ms") and r["write_ms"] > 0:
                speedup = pq["write_ms"] / r["write_ms"]
                line += f"  {speedup:>{cw}.2f}x"
            else:
                line += f"  {'-':>{cw}}"
        print(line)

    # Validation
    print()
    line = f"  {'Round-trip OK?':18}"
    all_valid = True
    for r in results:
        if r.get("valid"):
            line += f"  {'YES':>{cw}}"
        else:
            line += f"  {'FAIL':>{cw}}"
            all_valid = False
    print(line)

    if not all_valid:
        for r in results:
            if not r.get("valid") and r.get("validation_errors"):
                print(f"\n  [{r['codec']}] validation errors:")
                for err in r["validation_errors"]:
                    print(f"    - {err}")

    print()


# ── Graphs ────────────────────────────────────────────────────

GRAPH_DIR = Path(__file__).parent


def generate_graphs(results: list[dict], source_name: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
    except ImportError:
        print("  [skip] matplotlib not installed — no graphs generated")
        return

    plt.rcParams.update({
        "figure.facecolor": "#0f172a",
        "axes.facecolor": "#1e293b",
        "axes.edgecolor": "#334155",
        "axes.labelcolor": "#e2e8f0",
        "text.color": "#e2e8f0",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "grid.color": "#334155",
        "grid.alpha": 0.5,
        "font.family": "monospace",
        "font.size": 11,
    })

    COLORS = {
        "parquet": "#2563eb",
        "arrow_ipc": "#f59e0b",
        "feather": "#10b981",
        "vortex": "#a855f7",
    }

    codecs = [r["codec"] for r in results]
    x = np.arange(len(codecs))

    def save(fig, name):
        path = GRAPH_DIR / f"{name}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  saved {path}")

    n = results[0]["rows"]
    title_suffix = f"{fmt_rows(n)} rows — {source_name}"

    # ── Combined: write, read, file size ──────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Write time
    ax = axes[0]
    vals = [r.get("write_ms", 0) or 0 for r in results]
    bars = ax.bar(x, vals, color=[COLORS.get(c, "#64748b") for c in codecs])
    ax.set_xticks(x)
    ax.set_xticklabels(codecs, rotation=20, ha="right")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Write Time", fontsize=13, fontweight="bold")
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt_ms(v), ha="center", va="bottom", fontsize=9)

    # Read time
    ax = axes[1]
    vals = [r.get("read_ms", 0) or 0 for r in results]
    bars = ax.bar(x, vals, color=[COLORS.get(c, "#64748b") for c in codecs])
    ax.set_xticks(x)
    ax.set_xticklabels(codecs, rotation=20, ha="right")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Read Time", fontsize=13, fontweight="bold")
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt_ms(v), ha="center", va="bottom", fontsize=9)

    # File size
    ax = axes[2]
    vals = [r.get("file_size", 0) or 0 for r in results]
    bars = ax.bar(x, [v / (1024 * 1024) for v in vals],
                  color=[COLORS.get(c, "#64748b") for c in codecs])
    ax.set_xticks(x)
    ax.set_xticklabels(codecs, rotation=20, ha="right")
    ax.set_ylabel("Size (MB)")
    ax.set_title("File Size", fontsize=13, fontweight="bold")
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt_size(v), ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"Codec Comparison — {title_suffix}", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "codec_comparison")

    print(f"\n  Graphs saved to {GRAPH_DIR}/")


# ── Main ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compare .cohort codec performance (parquet, arrow_ipc, feather, vortex)")
    parser.add_argument(
        "--source", default=None,
        help="Source .cohort file (default: largest in cwd)")
    parser.add_argument(
        "--rows", default="full",
        help="Max rows to test: 1m, 10m, full (default: full)")
    parser.add_argument(
        "--codecs", nargs="+", default=None,
        help=f"Codecs to test (default: all). Choices: {', '.join(ALL_CODECS)}")
    parser.add_argument(
        "--no-graphs", action="store_true",
        help="Skip graph generation")
    args = parser.parse_args()

    # Parse rows
    rows_str = args.rows.lower().replace(",", "").replace("_", "")
    if rows_str in SCALE_SHORTCUTS:
        max_rows = SCALE_SHORTCUTS[rows_str]
    elif rows_str == "full":
        max_rows = -1
    else:
        try:
            max_rows = int(rows_str)
        except ValueError:
            print(f"  Unknown row count: {args.rows}")
            print(f"  Use a number or shortcut: {', '.join(SCALE_SHORTCUTS.keys())}")
            sys.exit(1)

    codecs = args.codecs or ALL_CODECS
    for c in codecs:
        if c not in ALL_CODECS:
            print(f"  Unknown codec: {c}. Choose from: {', '.join(ALL_CODECS)}")
            sys.exit(1)

    print()
    print("  cohort codec benchmark")
    print(f"  Codecs: {', '.join(codecs)}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load source data
    table, label, source_name = load_source(args.source, max_rows)
    definition = CohortDefinition(name="codec-benchmark", revision=1)

    print(f"\n  Benchmarking {len(codecs)} codecs on {fmt_rows(table.num_rows)} rows...\n")

    tmpdir = tempfile.mkdtemp(prefix="cohort_codec_bench_")
    results = []

    try:
        for codec in codecs:
            print(f"  [{codec}]", end="", flush=True)
            r = bench_codec(codec, table, label, tmpdir, definition)
            results.append(r)

            parts = []
            if r.get("write_ms") is not None:
                parts.append(f"write {fmt_ms(r['write_ms'])}")
            if r.get("read_ms") is not None:
                parts.append(f"read {fmt_ms(r['read_ms'])}")
            if r.get("file_size") is not None:
                parts.append(f"size {fmt_size(r['file_size'])}")
            valid = "OK" if r.get("valid") else "FAIL"
            parts.append(f"roundtrip {valid}")
            print(f"  {' | '.join(parts)}")

        # DuckDB baseline
        print(f"  [duckdb]", end="", flush=True)
        duck = bench_duckdb(table, tmpdir)
        parts = []
        if duck.get("write_ms") is not None:
            parts.append(f"write {fmt_ms(duck['write_ms'])}")
        if duck.get("read_ms") is not None:
            parts.append(f"read {fmt_ms(duck['read_ms'])}")
        if duck.get("file_size") is not None:
            parts.append(f"size {fmt_size(duck['file_size'])}")
        if duck.get("sql_count_ms") is not None:
            parts.append(f"count {fmt_ms(duck['sql_count_ms'])}")
        print(f"  {' | '.join(parts)}")
        results.append(duck)

        print_results(results, source_name)

        if not args.no_graphs:
            generate_graphs(results, source_name)

        # Save results JSON
        results_path = GRAPH_DIR / "codec_results.json"
        with open(results_path, "w") as f:
            json.dump({
                "source": source_name,
                "timestamp": datetime.now().isoformat(),
                "results": results,
            }, f, indent=2)
        print(f"  Results saved to {results_path}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n{'=' * 78}")
    print("  Done.")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()

"""Benchmarks for cohort-sdk: scaling from 10K to 10M rows.

Usage:
    python benchmarks/bench.py                    # default: 10K → 10M
    python benchmarks/bench.py --max-scale 1M     # stop at 1M rows
    python benchmarks/bench.py --scales 50000 500000  # custom scales
    python benchmarks/bench.py --no-graphs        # skip graph generation

Shows how every core operation scales with cohort size, then compares
.cohort against raw Parquet and CSV at each tier.  Generates PNG charts
in benchmarks/ when matplotlib is available.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from cohort_sdk import (
    CohortDefinition,
    Delta,
    Lineage,
    RevisionEntry,
    Selector,
    inspect,
    intersect,
    read,
    union,
    write,
)

# ── Helpers ───────────────────────────────────────────────────


@contextmanager
def timer():
    """Context manager that yields a dict; populates elapsed_ms on exit."""
    result = {}
    gc.disable()
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_ms"] = (time.perf_counter() - t0) * 1000
        gc.enable()


def fmt_ms(ms: float) -> str:
    if ms < 1:
        return f"{ms * 1000:.0f} us"
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{ms / 1000:.2f} s"


def fmt_rows(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def fmt_size(b: int) -> str:
    if b < 1024:
        return f"{b} B"
    if b < 1024 * 1024:
        return f"{b / 1024:.0f} KB"
    return f"{b / (1024 * 1024):.1f} MB"


def fmt_rate(rows: int, ms: float) -> str:
    if ms <= 0:
        return "-"
    per_sec = rows / (ms / 1000)
    if per_sec >= 1_000_000:
        return f"{per_sec / 1_000_000:.1f}M/s"
    if per_sec >= 1_000:
        return f"{per_sec / 1_000:.0f}K/s"
    return f"{per_sec:.0f}/s"


# ── Fast synthetic data (numpy → Arrow, no Python loops) ──────


DX_CODES = ["E11.0", "E11.1", "E11.9", "I10", "J45.20", "M54.5", "K21.0"]
STATES = ["CA", "TX", "NY", "FL", "IL", "PA", "OH"]
GENDERS = ["M", "F"]


def make_patient_table(n: int, seed: int = 42) -> pa.Table:
    """Generate a realistic patient claims table — fast via numpy."""
    rng = np.random.default_rng(seed)

    patient_ids = np.arange(1, n + 1, dtype=np.int64)
    genders = rng.choice(GENDERS, size=n)
    ages = rng.integers(18, 96, size=n, dtype=np.int32)
    states = rng.choice(STATES, size=n)
    dx_codes = rng.choice(DX_CODES, size=n)
    amounts = np.round(rng.uniform(50, 50000, size=n), 2)
    months = rng.integers(1, 13, size=n)
    days = rng.integers(1, 29, size=n)
    dates = [f"2025-{m:02d}-{d:02d}" for m, d in zip(months, days)]
    npis = rng.integers(1_000_000_000, 9_999_999_999, size=n, dtype=np.int64)

    return pa.table({
        "PATIENT_NUMBER": pa.array(patient_ids),
        "PATIENT_GENDER": pa.array(genders),
        "PATIENT_AGE": pa.array(ages),
        "PATIENT_STATE": pa.array(states),
        "DIAGNOSIS_CODE": pa.array(dx_codes),
        "CLAIM_AMOUNT": pa.array(amounts),
        "SERVICE_DATE": pa.array(dates),
        "PROVIDER_NPI": pa.array(npis),
    })


def make_rx_table(n: int, seed: int = 99) -> pa.Table:
    rng = np.random.default_rng(seed)
    return pa.table({
        "PATIENT_NUMBER": rng.integers(1, n + 1, size=n, dtype=np.int64),
        "NDC_CODE": rng.integers(10_000_000_000, 99_999_999_999, size=n, dtype=np.int64),
        "QUANTITY": rng.integers(1, 91, size=n, dtype=np.int32),
    })


def make_definition(name: str = "Benchmark Cohort") -> CohortDefinition:
    now = datetime.now(timezone.utc)
    return CohortDefinition(
        name=name,
        description="Synthetic cohort for benchmarking",
        created_at=now,
        created_by="bench",
        selectors=[
            Selector(
                selector_id="sel-dx",
                field="DIAGNOSIS_CODE",
                code_system="ICD-10-CM",
                operator="in",
                values=["E11.0", "E11.1", "E11.9"],
                label="Type 2 Diabetes",
            ),
            Selector(
                selector_id="sel-age",
                field="PATIENT_AGE",
                operator="between",
                values=[40, 75],
                label="Age 40-75",
            ),
            Selector(
                selector_id="sel-state",
                field="PATIENT_STATE",
                operator="in",
                values=["CA", "TX", "NY"],
                label="Target states",
            ),
        ],
        lineage=Lineage(
            source_system=["Benchmark Warehouse"],
            source_tables=["claims", "rx"],
            extraction_date="2026-02-10",
        ),
        revision_history=[
            RevisionEntry(
                revision=1,
                timestamp=now,
                author="bench",
                message="Initial cohort",
            )
        ],
    )


# ── Scaling sweep ─────────────────────────────────────────────


def run_scale(n: int, tmpdir: str) -> dict:
    """Run all core operations at a given row count. Returns timing dict."""
    results = {"rows": n}
    label = fmt_rows(n)
    print(f"\n  [{label} rows] generating data...", end="", flush=True)

    # ── Generate ──────────────────────────────────────────────
    with timer() as t:
        claims = make_patient_table(n)
        rx = make_rx_table(n // 2)
    results["gen_ms"] = t["elapsed_ms"]
    print(f" {fmt_ms(t['elapsed_ms'])}", flush=True)

    defn = make_definition(f"Cohort-{label}")

    # ── Write .cohort ─────────────────────────────────────────
    cohort_path = os.path.join(tmpdir, f"scale_{n}.cohort")
    with timer() as t:
        write(cohort_path, defn, [("Claims", claims), ("Rx", rx)])
    results["write_ms"] = t["elapsed_ms"]
    results["file_size"] = os.path.getsize(cohort_path)
    print(f"  [{label}] write       {fmt_ms(t['elapsed_ms']):>10}  "
          f"({fmt_size(results['file_size'])})", flush=True)

    # ── Inspect ───────────────────────────────────────────────
    inspect(cohort_path)  # warm up
    iters = 100 if n <= 1_000_000 else 20
    with timer() as t:
        for _ in range(iters):
            inspect(cohort_path)
    results["inspect_ms"] = t["elapsed_ms"] / iters
    print(f"  [{label}] inspect     {fmt_ms(results['inspect_ms']):>10}", flush=True)

    # ── Read definition only ──────────────────────────────────
    iters_read = 50 if n <= 1_000_000 else 10
    with timer() as t:
        for _ in range(iters_read):
            c = read(cohort_path)
            _ = c.definition
    results["defn_ms"] = t["elapsed_ms"] / iters_read

    # ── Read single dataset ──────────────────────────────────
    read(cohort_path)["Claims"]  # warm up
    iters_data = 10 if n <= 1_000_000 else 3
    with timer() as t:
        for _ in range(iters_data):
            c = read(cohort_path)
            _ = c["Claims"]
    results["read_one_ms"] = t["elapsed_ms"] / iters_data
    print(f"  [{label}] read 1 mod  {fmt_ms(results['read_one_ms']):>10}  "
          f"({fmt_rate(n, results['read_one_ms'])})", flush=True)

    # ── Read all datasets ───────────────────────────────────
    with timer() as t:
        for _ in range(iters_data):
            c = read(cohort_path)
            for mod in c.datasets:
                _ = c[mod]
    results["read_all_ms"] = t["elapsed_ms"] / iters_data
    total = n + n // 2
    print(f"  [{label}] read all    {fmt_ms(results['read_all_ms']):>10}  "
          f"({fmt_rate(total, results['read_all_ms'])})", flush=True)

    # ── Write .cohort (single dataset, for fair format comparison) ──
    cohort_1mod_path = os.path.join(tmpdir, f"scale_{n}_1mod.cohort")
    defn_1mod = make_definition(f"Cohort-{label}-1mod")
    with timer() as t:
        write(cohort_1mod_path, defn_1mod, [("Claims", claims)])
    results["write_1mod_ms"] = t["elapsed_ms"]
    results["file_size_1mod"] = os.path.getsize(cohort_1mod_path)

    # ── Read .cohort (single dataset) ────────────────────────
    read(cohort_1mod_path)["Claims"]  # warm up
    with timer() as t:
        for _ in range(iters_data):
            _ = read(cohort_1mod_path)["Claims"]
    results["read_1mod_ms"] = t["elapsed_ms"] / iters_data

    # ── Write Parquet (for comparison) ────────────────────────
    pq_path = os.path.join(tmpdir, f"scale_{n}.parquet")
    with timer() as t:
        pq.write_table(claims, pq_path)
    results["pq_write_ms"] = t["elapsed_ms"]
    results["pq_size"] = os.path.getsize(pq_path)

    # ── Read Parquet ──────────────────────────────────────────
    _ = pq.read_table(pq_path)  # warm up
    with timer() as t:
        for _ in range(iters_data):
            _ = pq.read_table(pq_path)
    results["pq_read_ms"] = t["elapsed_ms"] / iters_data

    # ── Write CSV ─────────────────────────────────────────────
    csv_path = os.path.join(tmpdir, f"scale_{n}.csv")
    with timer() as t:
        pcsv.write_csv(claims, csv_path)
    results["csv_write_ms"] = t["elapsed_ms"]
    results["csv_size"] = os.path.getsize(csv_path)

    # ── Read CSV ──────────────────────────────────────────────
    _ = pcsv.read_csv(csv_path)  # warm up
    with timer() as t:
        for _ in range(iters_data):
            _ = pcsv.read_csv(csv_path)
    results["csv_read_ms"] = t["elapsed_ms"] / iters_data

    # ── Union 2 files ─────────────────────────────────────────
    cohort_b = os.path.join(tmpdir, f"scale_{n}_b.cohort")
    claims_b = make_patient_table(n, seed=77)
    write(cohort_b, make_definition(f"B-{label}"), [("Claims", claims_b), ("Rx", rx)])
    union_out = os.path.join(tmpdir, f"union_{n}.cohort")
    with timer() as t:
        union([cohort_path, cohort_b], union_out, name=f"Union-{label}")
    results["union_ms"] = t["elapsed_ms"]
    print(f"  [{label}] union (2x)  {fmt_ms(t['elapsed_ms']):>10}", flush=True)

    # ── Intersect 2 files ────────────────────────────────────
    int_out = os.path.join(tmpdir, f"intersect_{n}.cohort")
    with timer() as t:
        intersect([cohort_path, cohort_b], int_out, on="PATIENT_NUMBER",
                  name=f"Intersect-{label}")
    results["intersect_ms"] = t["elapsed_ms"]
    print(f"  [{label}] intersect   {fmt_ms(t['elapsed_ms']):>10}", flush=True)

    # Clean up large objects
    del claims, rx, claims_b
    gc.collect()

    return results


# ── Display ───────────────────────────────────────────────────


def print_scaling_table(all_results: list[dict]):
    """Print the main scaling table."""
    cols = [fmt_rows(r["rows"]) for r in all_results]
    w = max(12, max(len(c) for c in cols) + 2)

    def table_header():
        hdr = f"  {'Operation':<26}"
        for c in cols:
            hdr += f"  {c:>{w}}"
        print(hdr)
        print("  " + "-" * (26 + (w + 2) * len(cols)))

    def table_row(label: str, key: str, rate_key: str | None = None):
        line = f"  {label:<26}"
        for r in all_results:
            val = fmt_ms(r[key])
            if rate_key:
                val += f" ({fmt_rate(r['rows'], r[key])})"
            line += f"  {val:>{w}}"
        print(line)

    def table_row_size(label: str, key: str):
        line = f"  {label:<26}"
        for r in all_results:
            line += f"  {fmt_size(r[key]):>{w}}"
        print(line)

    # ── Main operations table ─────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  SCALING: Core Operations")
    print(f"{'=' * 72}\n")
    table_header()

    rows = [
        ("write", "write_ms"),
        ("inspect", "inspect_ms"),
        ("read (definition)", "defn_ms"),
        ("read (1 dataset)", "read_one_ms"),
        ("read (all data)", "read_all_ms"),
        ("union (2 files)", "union_ms"),
        ("intersect (2 files)", "intersect_ms"),
    ]
    for label, key in rows:
        table_row(label, key)

    # ── File sizes ────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  SCALING: File Sizes")
    print(f"{'=' * 72}\n")
    table_header()
    table_row_size(".cohort", "file_size")
    table_row_size("Parquet", "pq_size")
    table_row_size("CSV", "csv_size")

    # ── Format comparison table ───────────────────────────────
    print(f"\n{'=' * 72}")
    print("  FORMAT COMPARISON: .cohort vs Parquet vs CSV")
    print(f"{'=' * 72}")

    print("\n  Same data (Claims table only) across all three formats:\n")
    for r in all_results:
        label = fmt_rows(r["rows"])
        print(f"  --- {label} rows ---")
        print(f"  {'Write':<16} {'Read':<16} {'Size':<10} {'Format'}")
        print(f"  {'-'*16} {'-'*16} {'-'*10} {'-'*10}")
        print(f"  {fmt_ms(r['write_1mod_ms']):<16} {fmt_ms(r['read_1mod_ms']):<16} "
              f"{fmt_size(r['file_size_1mod']):<10} .cohort (+ metadata)")
        print(f"  {fmt_ms(r['pq_write_ms']):<16} {fmt_ms(r['pq_read_ms']):<16} "
              f"{fmt_size(r['pq_size']):<10} Parquet (data only)")
        print(f"  {fmt_ms(r['csv_write_ms']):<16} {fmt_ms(r['csv_read_ms']):<16} "
              f"{fmt_size(r['csv_size']):<10} CSV (data only)")
        print()

    # ── The punchline: inspect vs full read ─────────────────
    print(f"\n{'=' * 72}")
    print("  inspect() vs full data read")
    print(f"{'=' * 72}\n")
    print("  inspect() reads only the JSON tag directory — never touches")
    print("  patient data.  Compare to a full read at each scale:\n")

    line = f"  {'Rows':<12}"
    for r in all_results:
        line += f"  {fmt_rows(r['rows']):>12}"
    print(line)

    line = f"  {'inspect()':<12}"
    for r in all_results:
        line += f"  {fmt_ms(r['inspect_ms']):>12}"
    print(line)

    line = f"  {'full read':<12}"
    for r in all_results:
        line += f"  {fmt_ms(r['read_all_ms']):>12}"
    print(line)

    line = f"  {'speedup':<12}"
    for r in all_results:
        speedup = r["read_all_ms"] / r["inspect_ms"] if r["inspect_ms"] > 0 else 0
        line += f"  {speedup:>11.0f}x"
    print(line)

    line = f"  {'file size':<12}"
    for r in all_results:
        line += f"  {fmt_size(r['file_size']):>12}"
    print(line)

    print()
    print("  What you get from inspect() — without reading any patient data:")
    print("    - Cohort name, description, created_by, created_at")
    print("    - All selectors (filter criteria with ICD codes, age ranges, etc.)")
    print("    - Full data lineage (source system, tables, extraction date)")
    print("    - Complete revision history (who changed what, when, why)")
    print("    - Per-dataset stats (row counts, column schemas)")


def print_bonus_benchmarks():
    """Run the non-scaling benchmarks once."""

    # ── Revision tracking ─────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  BONUS: Revision Tracking Speed")
    print(f"{'=' * 72}\n")
    print("  .cohort tracks every change: who, what, when, why.")
    print("  Apply 100 sequential revisions to a definition:\n")

    defn = make_definition()
    now = datetime.now(timezone.utc)
    n_revisions = 100

    with timer() as t:
        for i in range(n_revisions):
            defn.apply_revision(RevisionEntry(
                revision=i + 2,
                timestamp=now,
                author="bench",
                message=f"Revision {i + 2}",
                deltas=[Delta(
                    action="modify",
                    selector_id="sel-age",
                    field="values",
                    before=[40, 75],
                    after=[40 + i, 75],
                )],
            ))

    print(f"  {n_revisions} revisions applied in {fmt_ms(t['elapsed_ms'])}")
    print(f"  Per revision: {fmt_ms(t['elapsed_ms'] / n_revisions)}")
    print(f"  Final state: r{defn.revision}, {len(defn.revision_history)} history entries")

    # ── Catalog scan ──────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  BONUS: Cohort Catalog Scan (50 files)")
    print(f"{'=' * 72}\n")
    print("  Simulate a catalog UI loading metadata for 50 cohort files:\n")

    tmpdir = tempfile.mkdtemp(prefix="cohort_catalog_")
    try:
        paths = []
        for i in range(50):
            table = make_patient_table(5000, seed=i)
            d = make_definition(f"Study-{i + 1}")
            p = os.path.join(tmpdir, f"cat_{i}.cohort")
            write(p, d, [("Claims", table)])
            paths.append(p)

        inspect(paths[0])  # warm up

        with timer() as t:
            for p in paths:
                inspect(p)

        print(f"  50 files scanned in {fmt_ms(t['elapsed_ms'])}")
        print(f"  Per file: {fmt_ms(t['elapsed_ms'] / 50)}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Graphs ────────────────────────────────────────────────────


GRAPH_DIR = Path(__file__).parent


def generate_graphs(all_results: list[dict]):
    """Generate benchmark charts as PNGs in benchmarks/."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("\n  [skip] matplotlib not installed — no graphs generated")
        return

    print(f"\n{'=' * 72}")
    print("  Generating graphs...")
    print(f"{'=' * 72}")

    scales = [r["rows"] for r in all_results]
    scale_labels = [fmt_rows(r["rows"]) for r in all_results]

    # Shared style
    COLORS = {
        "cohort": "#2563eb",   # blue
        "parquet": "#f59e0b",  # amber
        "csv": "#ef4444",      # red
        "inspect": "#10b981",  # emerald
        "defn": "#8b5cf6",     # violet
        "read_one": "#2563eb",
        "read_all": "#0ea5e9",
        "write": "#f97316",
        "union": "#22c55e",
        "intersect": "#a855f7",
    }

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

    def save(fig, name):
        path = GRAPH_DIR / f"{name}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  saved {path}")

    # ── 1. Core operations scaling (log-log) ──────────────────

    fig, ax = plt.subplots(figsize=(10, 6))
    ops = [
        ("write", "write_ms", "Write"),
        ("inspect", "inspect_ms", "Inspect"),
        ("read_one", "read_one_ms", "Read (1 dataset)"),
        ("read_all", "read_all_ms", "Read (all data)"),
        ("union", "union_ms", "Union (2 files)"),
        ("intersect", "intersect_ms", "Intersect (2 files)"),
    ]

    for color_key, data_key, label in ops:
        vals = [r[data_key] for r in all_results]
        ax.plot(scales, vals, "o-", color=COLORS[color_key], label=label,
                linewidth=2, markersize=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rows per cohort")
    ax.set_ylabel("Time (ms)")
    ax.set_title("cohort-sdk: Operation Scaling (10K → 10M rows)", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_rows(int(x))))
    ax.legend(loc="upper left", framealpha=0.8, facecolor="#1e293b", edgecolor="#475569")
    ax.grid(True, which="both", linestyle="--")
    save(fig, "scaling_operations")

    # ── 2. inspect() vs full read — the real comparison ─────

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    inspect_times = [r["inspect_ms"] for r in all_results]
    read_times = [r["read_all_ms"] for r in all_results]
    speedups = [r / i if i > 0 else 0 for r, i in zip(read_times, inspect_times)]

    x = np.arange(len(scale_labels))
    bar_w = 0.35

    # Left: grouped bars — inspect vs full read (log scale)
    b1 = ax1.bar(x - bar_w / 2, inspect_times, bar_w,
                 color=COLORS["inspect"], label="inspect()")
    b2 = ax1.bar(x + bar_w / 2, read_times, bar_w,
                 color="#475569", label="full read")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scale_labels)
    ax1.set_ylabel("Time (ms)")
    ax1.set_yscale("log")
    ax1.set_title("inspect() vs full data read", fontsize=13, fontweight="bold")
    ax1.legend(framealpha=0.8, facecolor="#1e293b", edgecolor="#475569")

    # Annotate speedup on each pair
    for i, (bar, spd) in enumerate(zip(b2, speedups)):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() * 1.3,
                 f"{spd:.0f}x", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color=COLORS["inspect"])

    # Right: what inspect gives you (text panel)
    ax2.axis("off")
    info_lines = [
        "inspect() returns all of this",
        "without reading patient data:\n",
        "  Cohort name & description",
        "  Selectors (ICD codes, age ranges...)",
        "  Data lineage (source, tables, date)",
        "  Full revision history (who/what/when)",
        "  Per-dataset row counts & schemas",
        "",
        f"  10M rows, 256 MB file:",
        f"  inspect:   {fmt_ms(inspect_times[-1])}",
        f"  full read: {fmt_ms(read_times[-1])}",
        f"  speedup:   {speedups[-1]:.0f}x faster",
    ]
    ax2.text(0.05, 0.95, "\n".join(info_lines),
             transform=ax2.transAxes, fontsize=12,
             verticalalignment="top", fontfamily="monospace",
             color="#e2e8f0",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#1e293b",
                       edgecolor="#475569", alpha=0.9))

    fig.suptitle("inspect(): full metadata without touching data",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "inspect_vs_read")

    # ── 3. Format comparison: read time ───────────────────────

    fig, (ax_write, ax_read) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(scale_labels))
    bar_w = 0.25

    # Write comparison (same data: Claims only)
    ax_write.bar(x - bar_w, [r["write_1mod_ms"] for r in all_results],
                 bar_w, color=COLORS["cohort"], label=".cohort")
    ax_write.bar(x, [r["pq_write_ms"] for r in all_results],
                 bar_w, color=COLORS["parquet"], label="Parquet")
    ax_write.bar(x + bar_w, [r["csv_write_ms"] for r in all_results],
                 bar_w, color=COLORS["csv"], label="CSV")
    ax_write.set_xticks(x)
    ax_write.set_xticklabels(scale_labels)
    ax_write.set_ylabel("Time (ms)")
    ax_write.set_title("Write time (same data)", fontsize=13, fontweight="bold")
    ax_write.legend(framealpha=0.8, facecolor="#1e293b", edgecolor="#475569")
    ax_write.set_yscale("log")

    # Read comparison (same data: Claims only)
    ax_read.bar(x - bar_w, [r["read_1mod_ms"] for r in all_results],
                bar_w, color=COLORS["cohort"], label=".cohort")
    ax_read.bar(x, [r["pq_read_ms"] for r in all_results],
                bar_w, color=COLORS["parquet"], label="Parquet")
    ax_read.bar(x + bar_w, [r["csv_read_ms"] for r in all_results],
                bar_w, color=COLORS["csv"], label="CSV")
    ax_read.set_xticks(x)
    ax_read.set_xticklabels(scale_labels)
    ax_read.set_ylabel("Time (ms)")
    ax_read.set_title("Read time", fontsize=13, fontweight="bold")
    ax_read.legend(framealpha=0.8, facecolor="#1e293b", edgecolor="#475569")
    ax_read.set_yscale("log")

    fig.suptitle(".cohort vs Parquet vs CSV", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "format_comparison")

    # ── 4. File size comparison ───────────────────────────────

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(scales, [r["file_size_1mod"] / (1024 * 1024) for r in all_results],
            "o-", color=COLORS["cohort"], linewidth=2, markersize=7, label=".cohort (+ metadata)")
    ax.plot(scales, [r["pq_size"] / (1024 * 1024) for r in all_results],
            "o-", color=COLORS["parquet"], linewidth=2, markersize=7, label="Parquet (data only)")
    ax.plot(scales, [r["csv_size"] / (1024 * 1024) for r in all_results],
            "o-", color=COLORS["csv"], linewidth=2, markersize=7, label="CSV (data only)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_rows(int(x))))
    ax.set_xlabel("Rows")
    ax.set_ylabel("File size (MB)")
    ax.set_title("File Size: .cohort adds metadata with minimal overhead", fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.8, facecolor="#1e293b", edgecolor="#475569")
    ax.grid(True, which="both", linestyle="--")
    save(fig, "file_sizes")

    # ── 5. Read throughput (rows/sec) ─────────────────────────

    fig, ax = plt.subplots(figsize=(10, 5))

    def rows_per_sec(r, key, total=None):
        n = total if total else r["rows"]
        return n / (r[key] / 1000) if r[key] > 0 else 0

    ax.plot(scales, [rows_per_sec(r, "read_1mod_ms") for r in all_results],
            "o-", color=COLORS["read_one"], linewidth=2, markersize=7,
            label=".cohort read (same data)")
    ax.plot(scales, [rows_per_sec(r, "read_all_ms", r["rows"] + r["rows"] // 2) for r in all_results],
            "o-", color=COLORS["read_all"], linewidth=2, markersize=7,
            label=".cohort read (all datasets)")
    ax.plot(scales, [rows_per_sec(r, "pq_read_ms") for r in all_results],
            "o--", color=COLORS["parquet"], linewidth=2, markersize=7,
            label="Parquet read")
    ax.plot(scales, [rows_per_sec(r, "csv_read_ms") for r in all_results],
            "o--", color=COLORS["csv"], linewidth=2, markersize=7,
            label="CSV read")

    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x / 1e6:.0f}M" if x >= 1e6 else f"{x / 1e3:.0f}K"))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_rows(int(x))))
    ax.set_xlabel("Rows")
    ax.set_ylabel("Rows / second")
    ax.set_title("Read Throughput", fontsize=14, fontweight="bold")
    ax.legend(framealpha=0.8, facecolor="#1e293b", edgecolor="#475569")
    ax.grid(True, which="both", linestyle="--")
    save(fig, "read_throughput")

    print(f"\n  All graphs saved to {GRAPH_DIR}/")


# ── Main ──────────────────────────────────────────────────────


DEFAULT_SCALES = [10_000, 100_000, 1_000_000, 10_000_000]

SCALE_SHORTCUTS = {
    "10k": 10_000, "100k": 100_000, "1m": 1_000_000, "10m": 10_000_000,
    "50k": 50_000, "500k": 500_000, "5m": 5_000_000,
}


def parse_scale(s: str) -> int:
    s = s.lower().replace(",", "").replace("_", "")
    if s in SCALE_SHORTCUTS:
        return SCALE_SHORTCUTS[s]
    return int(s)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark cohort-sdk across cohort sizes (10K → 10M rows)")
    parser.add_argument(
        "--scales", nargs="+", default=None,
        help="Custom row counts, e.g. --scales 10k 100k 1m 10m")
    parser.add_argument(
        "--max-scale", default=None,
        help="Stop at this scale, e.g. --max-scale 1m")
    parser.add_argument(
        "--no-graphs", action="store_true",
        help="Skip graph generation")
    args = parser.parse_args()

    if args.scales:
        scales = [parse_scale(s) for s in args.scales]
    else:
        scales = list(DEFAULT_SCALES)

    if args.max_scale:
        cap = parse_scale(args.max_scale)
        scales = [s for s in scales if s <= cap]

    scales.sort()

    print()
    print("  cohort-sdk benchmark")
    print(f"  Scales: {', '.join(fmt_rows(s) for s in scales)}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    tmpdir = tempfile.mkdtemp(prefix="cohort_bench_")
    all_results = []

    try:
        for n in scales:
            result = run_scale(n, tmpdir)
            all_results.append(result)

        print_scaling_table(all_results)
        print_bonus_benchmarks()

        if not args.no_graphs:
            generate_graphs(all_results)

        # Save raw results to JSON for telemetry / debugging
        results_path = GRAPH_DIR / "results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved to {results_path}")

        print(f"\n{'=' * 72}")
        print("  Done.")
        print(f"{'=' * 72}\n")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()

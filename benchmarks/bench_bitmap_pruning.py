"""Bitmap pruning benchmark — proves that bitmap indexes speed up point queries.

Creates N .cohort files with disjoint patient ID ranges, then runs
WHERE patient_id = X queries comparing:
  - With bitmap index (pruning enabled)
  - Without bitmap index (scans all files)

Usage:
    python benchmarks/bench_bitmap_pruning.py
    python benchmarks/bench_bitmap_pruning.py --files 20 --rows-per-file 100000
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pyarrow as pa

from cohort_sdk import CohortCatalog, CohortDefinition, CohortWriter
from cohort_sdk.models import TagDirectory


# ── Helpers ───────────────────────────────────────────────────


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


def timed(fn, iterations=1, warmup=0):
    for _ in range(warmup):
        fn()
    gc.disable()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    gc.enable()
    times.sort()
    # Use median to avoid outliers
    return times[len(times) // 2]


# ── Data generation ──────────────────────────────────────────


def make_table(start_id: int, num_rows: int) -> pa.Table:
    """Generate a table with patient_id range [start_id, start_id + num_rows)."""
    ids = list(range(start_id, start_id + num_rows))
    # Add some payload columns to make files non-trivial
    return pa.table({
        "patient_id": pa.array(ids, type=pa.int64()),
        "diagnosis": pa.array([f"E11.{i % 10}" for i in ids], type=pa.utf8()),
        "claim_amount": pa.array([float(i % 1000) + 0.50 for i in ids], type=pa.float64()),
        "visit_date": pa.array(["2025-01-15"] * num_rows, type=pa.utf8()),
        "provider_id": pa.array([f"PRV-{i % 500:05d}" for i in ids], type=pa.utf8()),
    })


# ── Benchmark runner ──────────────────────────────────────────


def run_benchmark(num_files: int, rows_per_file: int, iterations: int):
    total_rows = num_files * rows_per_file
    print(f"\n{'=' * 70}")
    print(f"  BITMAP PRUNING BENCHMARK")
    print(f"  {num_files} files x {fmt_rows(rows_per_file)} rows = {fmt_rows(total_rows)} total")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}\n")

    definition = CohortDefinition(name="bitmap-bench", revision=1)
    directory = TagDirectory(cohort_definition=definition)

    # Create two temp dirs: one with bitmaps, one without
    dir_with = tempfile.mkdtemp(prefix="bitmap_with_")
    dir_without = tempfile.mkdtemp(prefix="bitmap_without_")

    try:
        # ── Write files ──────────────────────────────────────
        print("  Writing files...", flush=True)

        for i in range(num_files):
            start_id = i * rows_per_file
            table = make_table(start_id, rows_per_file)

            path_with = os.path.join(dir_with, f"part_{i:04d}.cohort")
            path_without = os.path.join(dir_without, f"part_{i:04d}.cohort")

            CohortWriter.write(
                path_with, directory, [("claims", table)],
                codec="parquet", index_column="patient_id",
            )
            CohortWriter.write(
                path_without, directory, [("claims", table)],
                codec="parquet",
            )

        size_with = sum(
            os.path.getsize(os.path.join(dir_with, f))
            for f in os.listdir(dir_with) if f.endswith(".cohort")
        )
        size_without = sum(
            os.path.getsize(os.path.join(dir_without, f))
            for f in os.listdir(dir_without) if f.endswith(".cohort")
        )
        overhead_pct = ((size_with - size_without) / size_without) * 100
        print(f"    Without bitmap: {size_without / 1024:.0f} KB total")
        print(f"    With bitmap:    {size_with / 1024:.0f} KB total (+{overhead_pct:.1f}% overhead)")

        # ── Init catalogs ────────────────────────────────────
        print("  Initializing catalogs...", flush=True)
        CohortCatalog.init(dir_with)
        CohortCatalog.init(dir_without)
        cat_with = CohortCatalog(dir_with)
        cat_without = CohortCatalog(dir_without)

        tables_with = cat_with.tables()
        tables_without = cat_without.tables()
        assert tables_with, "No tables found in bitmap catalog"
        assert tables_without, "No tables found in no-bitmap catalog"
        tbl = tables_with[0]

        # ── Benchmark: point query (patient in first file) ───
        # Target is in the first file — bitmap should skip N-1 files
        target_id = rows_per_file // 2
        sql_eq = f'SELECT * FROM "{tbl}" WHERE patient_id = {target_id}'

        print(f"\n  Query: WHERE patient_id = {target_id}")
        print(f"  (target is in file 0 of {num_files})\n")

        ms_with = timed(lambda: cat_with.query(sql_eq), iterations=iterations, warmup=2)
        ms_without = timed(lambda: cat_without.query(sql_eq), iterations=iterations, warmup=2)

        # Verify correctness
        result_with = cat_with.query(sql_eq)
        result_without = cat_without.query(sql_eq)
        assert result_with.num_rows == 1, f"Expected 1 row with bitmap, got {result_with.num_rows}"
        assert result_without.num_rows == 1, f"Expected 1 row without bitmap, got {result_without.num_rows}"

        speedup_eq = ms_without / ms_with if ms_with > 0 else float("inf")
        print(f"  {'Point query (=)':30}  {'With bitmap':>14}  {'Without bitmap':>14}  {'Speedup':>10}")
        print(f"  {'-' * 30}  {'-' * 14}  {'-' * 14}  {'-' * 10}")
        print(f"  {'patient_id = X':30}  {fmt_ms(ms_with):>14}  {fmt_ms(ms_without):>14}  {speedup_eq:>9.1f}x")

        # ── Benchmark: IN list query ─────────────────────────
        # Pick 3 IDs all in the same file
        in_ids = [target_id, target_id + 1, target_id + 2]
        in_list = ", ".join(str(x) for x in in_ids)
        sql_in = f'SELECT * FROM "{tbl}" WHERE patient_id IN ({in_list})'

        ms_in_with = timed(lambda: cat_with.query(sql_in), iterations=iterations, warmup=2)
        ms_in_without = timed(lambda: cat_without.query(sql_in), iterations=iterations, warmup=2)

        result_in_with = cat_with.query(sql_in)
        result_in_without = cat_without.query(sql_in)
        assert result_in_with.num_rows == 3, f"Expected 3 rows, got {result_in_with.num_rows}"
        assert result_in_without.num_rows == 3, f"Expected 3 rows, got {result_in_without.num_rows}"

        speedup_in = ms_in_without / ms_in_with if ms_in_with > 0 else float("inf")
        print(f"  {'patient_id IN (x,y,z)':30}  {fmt_ms(ms_in_with):>14}  {fmt_ms(ms_in_without):>14}  {speedup_in:>9.1f}x")

        # ── Benchmark: non-key column (no pruning expected) ──
        sql_nokey = f'SELECT * FROM "{tbl}" WHERE diagnosis = \'E11.5\' LIMIT 100'

        ms_nk_with = timed(lambda: cat_with.query(sql_nokey), iterations=iterations, warmup=2)
        ms_nk_without = timed(lambda: cat_without.query(sql_nokey), iterations=iterations, warmup=2)

        speedup_nk = ms_nk_without / ms_nk_with if ms_nk_with > 0 else float("inf")
        print(f"  {'non-key filter (baseline)':30}  {fmt_ms(ms_nk_with):>14}  {fmt_ms(ms_nk_without):>14}  {speedup_nk:>9.1f}x")

        # ── Benchmark: COUNT(*) (no pruning expected) ────────
        sql_count = f'SELECT COUNT(*) FROM "{tbl}"'

        ms_cnt_with = timed(lambda: cat_with.query(sql_count), iterations=iterations, warmup=2)
        ms_cnt_without = timed(lambda: cat_without.query(sql_count), iterations=iterations, warmup=2)

        speedup_cnt = ms_cnt_without / ms_cnt_with if ms_cnt_with > 0 else float("inf")
        print(f"  {'COUNT(*) (no pruning)':30}  {fmt_ms(ms_cnt_with):>14}  {fmt_ms(ms_cnt_without):>14}  {speedup_cnt:>9.1f}x")

        # ── Summary ──────────────────────────────────────────
        print(f"\n  {'=' * 70}")
        print(f"  RESULT: Bitmap pruning gives {speedup_eq:.1f}x speedup on point queries")
        print(f"          ({num_files} files, only 1 read instead of {num_files})")
        print(f"          File size overhead: {overhead_pct:.1f}%")
        print(f"  {'=' * 70}\n")

    finally:
        shutil.rmtree(dir_with, ignore_errors=True)
        shutil.rmtree(dir_without, ignore_errors=True)


# ── Main ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Benchmark bitmap pruning speedup")
    parser.add_argument("--files", type=int, default=10, help="Number of .cohort files (default: 10)")
    parser.add_argument("--rows-per-file", type=int, default=50_000, help="Rows per file (default: 50000)")
    parser.add_argument("--iterations", type=int, default=5, help="Query iterations for median (default: 5)")
    args = parser.parse_args()

    run_benchmark(args.files, args.rows_per_file, args.iterations)


if __name__ == "__main__":
    main()

"""SDK-path and raw-format benchmarks for .cohort files.

Primary: SDK benchmarks using CohortWriter.write(codec=...) and CohortReader[label].
Secondary: Raw format-variant benchmarks using Python-side writers/readers.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pyarrow as pa

from cohort_sdk import (
    CohortCatalog,
    CohortDefinition,
    CohortWriter,
    read as cohort_read,
)

from cohort_viz.format_variants import (
    FORMATS,
    write_cohort_parquet,
    write_cohort_arrow,
    write_cohort_feather,
    write_cohort_roaring,
    read_cohort_any,
    inspect_cohort_any,
    read_roaring_ids,
)


def _time_it(fn, iterations: int = 1, warmup: int = 0) -> float:
    """Time a function, return average ms."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iterations):
        fn()
    return ((time.perf_counter() - t0) / iterations) * 1000


# ── SDK benchmark definitions ──────────────────────────────────

SDK_CODECS = ["parquet", "arrow_ipc", "feather"]

SDK_USE_CASES = [
    {
        "id": "sdk_write",
        "name": "Write",
        "description": "CohortWriter.write(codec=...) — Rust encode + disk I/O",
        "unit": "ms",
    },
    {
        "id": "sdk_read",
        "name": "Full Read",
        "description": "CohortReader[label] — Rust decode via Arrow FFI",
        "unit": "ms",
    },
    {
        "id": "sdk_inspect",
        "name": "Inspect",
        "description": "CohortDefinition.inspect() — metadata only, no data I/O",
        "unit": "ms",
    },
    {
        "id": "sql_count",
        "name": "SQL COUNT(*)",
        "description": "CohortCatalog.query(\"SELECT COUNT(*) ...\") — DataFusion path",
        "unit": "ms",
    },
    {
        "id": "sql_filter",
        "name": "SQL Filter",
        "description": "CohortCatalog.query(\"SELECT ... WHERE ...\") — predicate pushdown",
        "unit": "ms",
    },
    {
        "id": "file_size",
        "name": "File Size",
        "description": "Compressed size on disk",
        "unit": "MB",
    },
]


def _run_sdk_benchmarks(
    source_file: Path,
    test_table: pa.Table,
    dataset_label: str,
) -> dict[str, Any]:
    """Run SDK-path benchmarks across all codecs."""
    from cohort_sdk.models import TagDirectory

    sdk = {
        "codecs": SDK_CODECS,
        "use_cases": SDK_USE_CASES,
        "grid": {},
        "winners": {},
        "speedups": {},
    }

    definition = CohortDefinition(name="benchmark", revision=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_files: dict[str, str] = {}

        for codec in SDK_CODECS:
            path = os.path.join(tmpdir, f"test_{codec}.cohort")
            grid: dict[str, float] = {}

            # Pre-write so all benchmarks have a file to work with
            CohortWriter.write(
                path,
                TagDirectory(cohort_definition=definition),
                [(dataset_label, test_table)],
                codec=codec,
            )
            test_files[codec] = path

            # 1. SDK Write benchmark
            def do_write(p=path, c=codec):
                CohortWriter.write(
                    p,
                    TagDirectory(cohort_definition=definition),
                    [(dataset_label, test_table)],
                    codec=c,
                )

            ms = _time_it(do_write, iterations=3, warmup=1)
            grid["sdk_write"] = round(ms, 1)

            # 2. SDK Read benchmark
            def do_read(p=path, lbl=dataset_label):
                return cohort_read(p)[lbl]

            ms = _time_it(do_read, iterations=3, warmup=1)
            grid["sdk_read"] = round(ms, 1)

            # 3. Inspect benchmark
            def do_inspect(p=path):
                return CohortDefinition.inspect(p)

            ms = _time_it(do_inspect, iterations=20, warmup=1)
            grid["sdk_inspect"] = round(ms, 4)

            # 4. File size
            grid["file_size"] = round(os.path.getsize(path) / (1024 * 1024), 2)

            sdk["grid"][codec] = grid

        # 5 & 6. SQL benchmarks — need CohortCatalog
        if CohortCatalog is not None:
            sql_dir = os.path.join(tmpdir, "sql_bench")
            os.makedirs(sql_dir)

            for codec in SDK_CODECS:
                # Copy test file into sql_dir so catalog can find it
                src = test_files[codec]
                dst = os.path.join(sql_dir, f"bench_{codec}.cohort")
                import shutil
                shutil.copy2(src, dst)

            CohortCatalog.init(sql_dir)
            catalog = CohortCatalog(sql_dir)
            tables = catalog.tables()

            if tables:
                for codec in SDK_CODECS:
                    # Find the table name that corresponds to this codec's file
                    table_name = None
                    for t in tables:
                        if codec in t:
                            table_name = t
                            break
                    if table_name is None:
                        table_name = tables[0]

                    # SQL COUNT(*)
                    def do_count(tbl=table_name):
                        return catalog.query(f'SELECT COUNT(*) FROM "{tbl}"')

                    ms = _time_it(do_count, iterations=3, warmup=1)
                    sdk["grid"][codec]["sql_count"] = round(ms, 1)

                    # SQL Filter — use first column for a WHERE clause
                    # Column names in CohortCatalog are prefixed with modality and lowercased
                    # Format: {normalized_label}.{column_name_lowercase}
                    # Normalize: "Gen_Patients-v1" -> "gen_patients_v1"
                    # Note: Don't quote the column - the dot syntax works unquoted in DataFusion
                    raw_col = test_table.schema.field(0).name
                    normalized_label = dataset_label.lower().replace("-", "_")
                    qualified_col = f"{normalized_label}.{raw_col.lower()}"
                    def do_filter(tbl=table_name, col=qualified_col):
                        return catalog.query(
                            f'SELECT * FROM "{tbl}" WHERE {col} IS NOT NULL LIMIT 100'
                        )

                    ms = _time_it(do_filter, iterations=3, warmup=1)
                    sdk["grid"][codec]["sql_filter"] = round(ms, 1)

    # Compute winners (lowest value wins for each use case)
    for uc in SDK_USE_CASES:
        uc_id = uc["id"]
        values = [
            (c, sdk["grid"][c].get(uc_id))
            for c in SDK_CODECS
            if sdk["grid"][c].get(uc_id) is not None
        ]
        if values:
            sdk["winners"][uc_id] = min(values, key=lambda x: x[1])[0]

    # Compute speedups relative to parquet baseline
    baseline = "parquet"
    for codec in SDK_CODECS:
        sdk["speedups"][codec] = {}
        for uc in SDK_USE_CASES:
            uc_id = uc["id"]
            base_val = sdk["grid"].get(baseline, {}).get(uc_id)
            codec_val = sdk["grid"].get(codec, {}).get(uc_id)
            if base_val and codec_val and codec_val > 0:
                sdk["speedups"][codec][uc_id] = round(base_val / codec_val, 2)

    return sdk


# ── Raw format-variant benchmark definitions (existing) ────────

RAW_USE_CASES = [
    {
        "id": "metadata",
        "name": "Read Metadata",
        "description": "Read row count and schema without loading data",
        "unit": "ms",
        "applies_to": ["parquet", "arrow", "feather", "roaring"],
    },
    {
        "id": "full_read",
        "name": "Full Read",
        "description": "Load entire table into memory",
        "unit": "ms",
        "applies_to": ["parquet", "arrow", "feather"],
    },
    {
        "id": "one_column",
        "name": "Read 1 Column",
        "description": "Read only PATIENT_NUMBER column",
        "unit": "ms",
        "applies_to": ["parquet", "arrow", "feather"],
    },
    {
        "id": "get_ids",
        "name": "Get Patient IDs",
        "description": "Extract all patient IDs as a set",
        "unit": "ms",
        "applies_to": ["parquet", "arrow", "feather", "roaring"],
    },
    {
        "id": "intersect",
        "name": "AND (Intersect)",
        "description": "Find patients in BOTH cohorts",
        "unit": "ms",
        "applies_to": ["parquet", "arrow", "feather", "roaring"],
    },
    {
        "id": "union",
        "name": "OR (Union)",
        "description": "Find patients in EITHER cohort",
        "unit": "ms",
        "applies_to": ["parquet", "arrow", "feather", "roaring"],
    },
    {
        "id": "file_size",
        "name": "File Size",
        "description": "Size on disk",
        "unit": "MB",
        "applies_to": ["parquet", "arrow", "feather", "roaring"],
    },
]


def _run_raw_format_benchmarks(
    full_table: pa.Table,
    test_table: pa.Table,
    test_rows: int,
    patient_col_name: str | None,
    patient_col_idx: int,
) -> dict[str, Any]:
    """Run raw format-variant benchmarks (Python-side decode)."""
    from pyroaring import BitMap

    raw: dict[str, Any] = {
        "formats": [],
        "use_cases": RAW_USE_CASES,
        "grid": {},
        "winners": {},
        "speedups": {},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        test_files: dict[str, str] = {}

        for fmt_id, fmt_info in FORMATS.items():
            ext = fmt_info["extension"]
            path = f"{tmpdir}/test{ext}"

            if fmt_id == "parquet":
                write_cohort_parquet(path, test_table)
            elif fmt_id == "arrow":
                write_cohort_arrow(path, test_table)
            elif fmt_id == "feather":
                write_cohort_feather(path, test_table)
            elif fmt_id == "roaring":
                write_cohort_roaring(path, test_table, id_column=patient_col_name or "PATIENT_NUMBER")

            test_files[fmt_id] = path

            raw["formats"].append({
                "id": fmt_id,
                "name": fmt_id.upper(),
                "extension": ext,
                "description": fmt_info["description"],
                "file_size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
            })
            raw["grid"][fmt_id] = {}

        # Second cohort for set operations
        test_table2 = full_table.slice(test_rows // 2, test_rows)
        test_files2: dict[str, str] = {}
        for fmt_id, fmt_info in FORMATS.items():
            ext = fmt_info["extension"]
            path2 = f"{tmpdir}/test2{ext}"
            if fmt_id == "parquet":
                write_cohort_parquet(path2, test_table2)
            elif fmt_id == "arrow":
                write_cohort_arrow(path2, test_table2)
            elif fmt_id == "feather":
                write_cohort_feather(path2, test_table2)
            elif fmt_id == "roaring":
                write_cohort_roaring(path2, test_table2, id_column=patient_col_name or "PATIENT_NUMBER")
            test_files2[fmt_id] = path2

        for fmt_id, path in test_files.items():
            path2 = test_files2[fmt_id]
            applies = [uc["id"] for uc in RAW_USE_CASES if fmt_id in uc.get("applies_to", [])]

            if "metadata" in applies:
                ms = _time_it(lambda p=path: inspect_cohort_any(p), iterations=20, warmup=1)
                raw["grid"][fmt_id]["metadata"] = round(ms, 4)

            if "full_read" in applies:
                ms = _time_it(lambda p=path: read_cohort_any(p)[0].num_rows, iterations=3, warmup=1)
                raw["grid"][fmt_id]["full_read"] = round(ms, 1)

            if "one_column" in applies:
                def read_one_col(p=path, col=patient_col_idx):
                    tbl, _ = read_cohort_any(p)
                    return tbl.column(col).length()
                ms = _time_it(read_one_col, iterations=3, warmup=1)
                raw["grid"][fmt_id]["one_column"] = round(ms, 1)

            if "get_ids" in applies:
                if fmt_id == "roaring":
                    ms = _time_it(lambda p=path: read_roaring_ids(p), iterations=10, warmup=1)
                else:
                    def get_ids_table(p=path, col=patient_col_idx):
                        tbl, _ = read_cohort_any(p)
                        return set(tbl.column(col).to_pylist())
                    ms = _time_it(get_ids_table, iterations=3, warmup=1)
                raw["grid"][fmt_id]["get_ids"] = round(ms, 2)

            if "intersect" in applies:
                if fmt_id == "roaring":
                    ids1 = read_roaring_ids(path)
                    ids2 = read_roaring_ids(path2)
                    bm1 = BitMap(ids1)
                    bm2 = BitMap(ids2)
                    ms = _time_it(lambda: bm1 & bm2, iterations=100, warmup=1)
                else:
                    tbl1, _ = read_cohort_any(path)
                    tbl2, _ = read_cohort_any(path2)
                    ids1 = set(tbl1.column(patient_col_idx).to_pylist())
                    ids2 = set(tbl2.column(patient_col_idx).to_pylist())
                    ms = _time_it(lambda: ids1 & ids2, iterations=10, warmup=1)
                raw["grid"][fmt_id]["intersect"] = round(ms, 4)

            if "union" in applies:
                if fmt_id == "roaring":
                    ms = _time_it(lambda: bm1 | bm2, iterations=100, warmup=1)
                else:
                    ms = _time_it(lambda: ids1 | ids2, iterations=10, warmup=1)
                raw["grid"][fmt_id]["union"] = round(ms, 4)

            if "file_size" in applies:
                raw["grid"][fmt_id]["file_size"] = round(
                    os.path.getsize(path) / (1024 * 1024), 2
                )

    # Winners
    for uc in RAW_USE_CASES:
        uc_id = uc["id"]
        applicable_formats = uc.get("applies_to", [])
        values = [
            (fmt_id, raw["grid"][fmt_id].get(uc_id))
            for fmt_id in raw["grid"]
            if fmt_id in applicable_formats and raw["grid"][fmt_id].get(uc_id) is not None
        ]
        if values:
            raw["winners"][uc_id] = min(values, key=lambda x: x[1])[0]

    # Speedups relative to parquet
    baseline = "parquet"
    for fmt_id in raw["grid"]:
        raw["speedups"][fmt_id] = {}
        for uc in RAW_USE_CASES:
            uc_id = uc["id"]
            base_val = raw["grid"].get(baseline, {}).get(uc_id)
            fmt_val = raw["grid"].get(fmt_id, {}).get(uc_id)
            if base_val and fmt_val and fmt_val > 0:
                raw["speedups"][fmt_id][uc_id] = round(base_val / fmt_val, 2)

    return raw


# ── Main entry point ───────────────────────────────────────────

def run_comprehensive_benchmark(cohort_dir: Path) -> dict[str, Any]:
    """Run SDK-path + raw format benchmarks and return combined results."""
    # Find source .cohort files
    cohort_files = sorted(
        cohort_dir.glob("*.cohort"),
        key=lambda f: f.stat().st_size,
        reverse=True,
    )
    if not cohort_files:
        raise ValueError("No .cohort files found")

    source_file = cohort_files[0]
    source_cohort = cohort_read(str(source_file))
    dataset_label = source_cohort.datasets[0]
    full_table = source_cohort[dataset_label]

    TEST_ROWS = min(100_000, full_table.num_rows)
    test_table = full_table.slice(0, TEST_ROWS)

    # Find patient column for raw benchmarks
    patient_col_name = None
    patient_col_idx = 0
    for i, f in enumerate(test_table.schema):
        if f.name.lower() == "patient_number":
            patient_col_name = f.name
            patient_col_idx = i
            break

    summary = {
        "source_file": source_file.name,
        "total_rows": full_table.num_rows,
        "test_rows": TEST_ROWS,
        "num_columns": len(test_table.schema),
    }

    # Run SDK benchmarks (primary)
    sdk = _run_sdk_benchmarks(source_file, test_table, dataset_label)

    # Run raw format benchmarks (secondary)
    raw = _run_raw_format_benchmarks(
        full_table, test_table, TEST_ROWS, patient_col_name, patient_col_idx,
    )

    return {
        "summary": summary,
        # SDK section (primary)
        "sdk": sdk,
        # Raw format section (secondary — kept for backwards compat)
        "formats": raw["formats"],
        "use_cases": raw["use_cases"],
        "grid": raw["grid"],
        "winners": raw["winners"],
        "speedups": raw["speedups"],
    }

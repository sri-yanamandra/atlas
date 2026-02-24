"""cohort-sdk: Python SDK for .cohort binary container files (v3).

Wraps the Rust cohort-lib via PyO3 for high-performance binary IO,
with a Pythonic API and Snowflake integration.

Quick start::

    from cohort_sdk import read, CohortDefinition

    # Write
    definition = CohortDefinition(name="My Cohort")
    definition.write("out.cohort", [("Demographics", table)])

    # Read
    cohort = read("out.cohort")
    cohort.definition          # CohortDefinition
    cohort.datasets          # ["Demographics"]
    cohort["Demographics"]     # pa.Table
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .models import (
    ByteRange,
    CohortDefinition,
    CohortInspection,
    Delta,
    IcebergBackend,
    IcebergSnapshot,
    IcebergTableRef,
    Lineage,
    RevisionEntry,
    Selector,
    Tag,
    TagDirectory,
)
from .reader import CohortReader
from .writer import CohortWriter
from .snowflake import SnowflakeConnection
from .iceberg import IcebergConnection

try:
    from ._cohort_rs import CohortCatalog
except ImportError:
    CohortCatalog = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    import pyarrow as pa


def read(path: str | Path) -> CohortReader:
    """Open a .cohort file for reading.

    Returns a CohortReader that is subscriptable::

        cohort = read("data.cohort")
        cohort.definition
        cohort.datasets
        cohort["Demographics"]   # -> pa.Table
    """
    return CohortReader.open(path)


def write(
    path: str | Path,
    definition: CohortDefinition,
    datasets: list[tuple[str, pa.Table]],
    *,
    codec: str = "parquet",
) -> None:
    """Write a .cohort file.

    Shorthand for ``definition.write(path, datasets, codec=codec)``.
    """
    definition.write(path, datasets, codec=codec)


def inspect(path: str | Path) -> CohortInspection:
    """Inspect a .cohort file and return its metadata.

    Shorthand for ``CohortDefinition.inspect(path)``.
    """
    return CohortDefinition.inspect(path)


def _materialize_iceberg_inputs(
    input_paths: list[str | Path],
) -> tuple[list[str], list[str]]:
    """Replace Iceberg-backed .cohort paths with temp files containing embedded data.

    Returns (resolved_paths, temp_paths_to_clean_up).
    """
    import tempfile

    resolved: list[str] = []
    temps: list[str] = []
    for p in input_paths:
        reader = CohortReader.open(str(p))
        backend = reader.directory.backend
        if backend is not None and any(t.codec == "iceberg" for t in reader.directory.tags):
            # Fetch data from Iceberg and write a temp .cohort with embedded data
            conn = IcebergConnection(
                catalog_name=backend.catalog_name,
                region=backend.region,
            )
            import pyarrow as pa
            data = conn.query_table(backend.database, backend.table_name)

            # Rewrite tags as embedded (drop iceberg codec)
            for tag in reader.directory.tags:
                if tag.codec == "iceberg":
                    tag.codec = "parquet"
                    tag.offset = 0
                    tag.length = 0

            # Strip backend so the temp file is self-contained
            reader.directory.backend = None

            tmp = tempfile.NamedTemporaryFile(suffix=".cohort", delete=False)
            tmp.close()
            datasets = [(reader.directory.tags[0].label, data)]
            CohortWriter.write(tmp.name, reader.directory, datasets)
            resolved.append(tmp.name)
            temps.append(tmp.name)
        else:
            resolved.append(str(p))
    return resolved, temps


def union(
    input_paths: list[str | Path],
    output_path: str | Path,
    *,
    name: str | None = None,
) -> None:
    """Union: combine all rows from multiple .cohort files into one.

    Builds a merged definition from all inputs (selectors, lineage, history).
    Pass ``name`` to override the cohort name in the output::

        union(["a.cohort", "b.cohort"], "combined.cohort", name="All Patients")
    """
    import os

    from ._cohort_rs import union_cohorts

    resolved, temps = _materialize_iceberg_inputs(input_paths)
    try:
        union_cohorts(resolved, str(output_path), name)
    finally:
        for t in temps:
            os.unlink(t)


def intersect(
    input_paths: list[str | Path],
    output_path: str | Path,
    *,
    on: str,
    name: str | None = None,
) -> None:
    """Intersect: keep only rows where ``on`` key appears in ALL input files.

    ::

        intersect(["a.cohort", "b.cohort"], "shared.cohort",
                  on="PATIENT_NUMBER", name="Shared Patients")
    """
    import os

    from ._cohort_rs import intersect_cohorts

    resolved, temps = _materialize_iceberg_inputs(input_paths)
    try:
        intersect_cohorts(resolved, str(output_path), on, name)
    finally:
        for t in temps:
            os.unlink(t)


# Keep merge as an alias for union
merge = union


def iceberg_append(
    target_path: str | Path,
    source_path: str | Path,
    *,
    dataset: str | None = None,
) -> int:
    """Append data from source .cohort into the Iceberg table of target .cohort.

    Returns the number of rows appended.
    """
    from datetime import datetime, timezone

    target_reader = CohortReader.open(target_path)
    target_dir = target_reader.directory
    target_backend = target_dir.backend

    if target_backend is None:
        raise ValueError(
            f"Target {target_path!r} has no IcebergBackend — "
            "iceberg_append requires an Iceberg-backed target .cohort"
        )

    source_reader = CohortReader.open(source_path)
    source_backend = source_reader.directory.backend

    # Read source data
    if source_backend is not None:
        if (
            source_backend.database == target_backend.database
            and source_backend.table_name == target_backend.table_name
        ):
            raise ValueError(
                f"Source and target point to the same Iceberg table: "
                f"{target_backend.database}.{target_backend.table_name}"
            )
        source_conn = IcebergConnection(
            catalog_name=source_backend.catalog_name,
            region=source_backend.region,
        )
        data = source_conn.query_table(source_backend.database, source_backend.table_name)
    else:
        labels = source_reader.datasets
        if not labels:
            return 0
        if len(labels) == 1:
            label = labels[0]
        elif dataset is not None:
            if dataset not in labels:
                raise ValueError(
                    f"Dataset {dataset!r} not found in source. Available: {labels}"
                )
            label = dataset
        else:
            raise ValueError(
                f"Source has multiple datasets {labels} — use --dataset to select one"
            )
        data = source_reader.get_dataset_data(label)

    if data.num_rows == 0:
        return 0

    target_conn = IcebergConnection(
        catalog_name=target_backend.catalog_name,
        region=target_backend.region,
    )
    iceberg_table = target_conn.write_table(
        target_backend.database,
        target_backend.table_name,
        data,
        location=target_backend.table_location,
        mode="append",
    )

    now = datetime.now(timezone.utc)
    snapshot = iceberg_table.current_snapshot()
    target_backend.snapshot_id = snapshot.snapshot_id if snapshot else None

    history = target_dir.cohort_definition.revision_history
    next_rev = (history[-1].revision + 1) if history else 1
    source_basename = Path(source_path).name
    total_rows = iceberg_table.scan().to_arrow().num_rows

    history.append(RevisionEntry(
        revision=next_rev,
        timestamp=now,
        author="cohort-sdk",
        message=f"Append from {source_basename}",
        size=total_rows,
    ))
    target_dir.cohort_definition.revision = next_rev
    CohortWriter.write(target_path, target_dir, [])

    return data.num_rows


def iceberg_upsert(
    target_path: str | Path,
    source_path: str | Path,
    *,
    on: list[str] | None = None,
    dataset: str | None = None,
) -> tuple[int, int]:
    """Upsert source data into target's Iceberg table.

    Returns (rows_updated, rows_inserted).
    """
    from datetime import datetime, timezone

    target_reader = CohortReader.open(target_path)
    target_dir = target_reader.directory
    target_backend = target_dir.backend

    if target_backend is None:
        raise ValueError(
            f"Target {target_path!r} has no IcebergBackend — "
            "iceberg_upsert requires an Iceberg-backed target .cohort"
        )

    source_reader = CohortReader.open(source_path)
    source_backend = source_reader.directory.backend

    if source_backend is not None:
        if (
            source_backend.database == target_backend.database
            and source_backend.table_name == target_backend.table_name
        ):
            raise ValueError(
                f"Source and target point to the same Iceberg table: "
                f"{target_backend.database}.{target_backend.table_name}"
            )
        source_conn = IcebergConnection(
            catalog_name=source_backend.catalog_name,
            region=source_backend.region,
        )
        data = source_conn.query_table(source_backend.database, source_backend.table_name)
    else:
        labels = source_reader.datasets
        if not labels:
            return (0, 0)
        if len(labels) == 1:
            label = labels[0]
        elif dataset is not None:
            if dataset not in labels:
                raise ValueError(
                    f"Dataset {dataset!r} not found in source. Available: {labels}"
                )
            label = dataset
        else:
            raise ValueError(
                f"Source has multiple datasets {labels} — use --dataset to select one"
            )
        data = source_reader.get_dataset_data(label)

    if data.num_rows == 0:
        return (0, 0)

    join_cols = on if on is not None else target_dir.cohort_definition.primary_keys
    if not join_cols:
        raise ValueError(
            "No primary keys defined on target .cohort and no --on columns provided. "
            "Either set primary_keys in the target cohort definition or pass --on."
        )

    target_conn = IcebergConnection(
        catalog_name=target_backend.catalog_name,
        region=target_backend.region,
    )
    outcome = target_conn.upsert_table(
        target_backend.database,
        target_backend.table_name,
        data,
        join_cols=join_cols,
        location=target_backend.table_location,
    )

    now = datetime.now(timezone.utc)
    snapshot = outcome.table.current_snapshot()
    target_backend.snapshot_id = snapshot.snapshot_id if snapshot else None

    history = target_dir.cohort_definition.revision_history
    next_rev = (history[-1].revision + 1) if history else 1
    source_basename = Path(source_path).name
    total_rows = outcome.table.scan().to_arrow().num_rows

    history.append(RevisionEntry(
        revision=next_rev,
        timestamp=now,
        author="cohort-sdk",
        message=f"Upsert from {source_basename}",
        size=total_rows,
    ))
    target_dir.cohort_definition.revision = next_rev
    CohortWriter.write(target_path, target_dir, [])

    return (outcome.rows_updated, outcome.rows_inserted)


def annotate(
    path: str | Path,
    *,
    output: str | Path | None = None,
    name: str | None = None,
    description: str | None = None,
    add_selectors: list[Selector] | None = None,
    remove_selector_ids: list[str] | None = None,
    message: str = "Annotate",
    author: str = "cohort-sdk",
) -> None:
    """Modify name, description, and/or selectors of a .cohort file."""
    from ._cohort_rs import annotate_cohort

    add_json = None
    if add_selectors:
        import json
        add_json = json.dumps([s.model_dump() for s in add_selectors])

    annotate_cohort(
        str(path),
        str(output) if output else None,
        name,
        description,
        add_json,
        remove_selector_ids,
        message,
        author,
    )


def diff(
    left_path: str | Path,
    right_path: str | Path,
) -> dict:
    """Compare two .cohort files and return a diff dict."""
    import json
    from ._cohort_rs import diff_cohorts

    return json.loads(diff_cohorts(str(left_path), str(right_path)))


def diff_revisions(
    path: str | Path,
    rev_a: int,
    rev_b: int,
) -> dict:
    """Compare two revisions within the same .cohort file."""
    import json
    from ._cohort_rs import diff_revisions_cohort

    return json.loads(diff_revisions_cohort(str(path), rev_a, rev_b))


def append(
    path: str | Path,
    data: "pa.RecordBatch | pa.Table",
    *,
    dataset_label: str | None = None,
    output: str | Path | None = None,
    message: str = "Append data",
    author: str = "cohort-sdk",
) -> None:
    """Append rows to a dataset (or add a new dataset).

    If *dataset_label* is omitted, auto-detects the target dataset by
    matching the incoming data's schema against existing datasets.
    """
    import pyarrow as pa
    from ._cohort_rs import append_cohort

    if isinstance(data, pa.Table):
        data = data.to_batches()[0] if data.num_rows > 0 else pa.record_batch([], schema=data.schema)

    append_cohort(str(path), data, dataset_label, str(output) if output else None, message, author)


def delete_rows(
    path: str | Path,
    key_column: str,
    values: list[str],
    *,
    output: str | Path | None = None,
    message: str = "Delete rows",
    author: str = "cohort-sdk",
) -> None:
    """Delete rows by key values across all datasets."""
    from ._cohort_rs import delete_rows_cohort

    delete_rows_cohort(str(path), key_column, values, str(output) if output else None, message, author)


def delete_dataset(
    path: str | Path,
    label: str,
    *,
    output: str | Path | None = None,
    message: str = "Delete dataset",
    author: str = "cohort-sdk",
) -> None:
    """Delete an entire dataset from a .cohort file."""
    from ._cohort_rs import delete_dataset_cohort

    delete_dataset_cohort(str(path), label, str(output) if output else None, message, author)


def delete_selectors(
    path: str | Path,
    selector_ids: list[str],
    *,
    output: str | Path | None = None,
    message: str = "Delete selectors",
    author: str = "cohort-sdk",
) -> None:
    """Delete selectors by ID from a .cohort file."""
    from ._cohort_rs import delete_selectors_cohort

    delete_selectors_cohort(str(path), selector_ids, str(output) if output else None, message, author)


def upsert(
    path: str | Path,
    dataset_label: str,
    key_column: str,
    data: "pa.RecordBatch | pa.Table",
    *,
    output: str | Path | None = None,
    message: str = "Upsert data",
    author: str = "cohort-sdk",
) -> None:
    """Update-or-insert rows by key column in a dataset."""
    import pyarrow as pa
    from ._cohort_rs import upsert_cohort

    if isinstance(data, pa.Table):
        data = data.to_batches()[0] if data.num_rows > 0 else pa.record_batch([], schema=data.schema)

    upsert_cohort(str(path), dataset_label, key_column, data, str(output) if output else None, message, author)


def increment(
    base: "pa.RecordBatch | pa.Table",
    key_column: str,
    increments: list[tuple[list[str], "pa.RecordBatch | pa.Table | None"]],
) -> "pa.Table":
    """Apply incremental (delete + add) steps to a base table. Returns pa.Table."""
    import pyarrow as pa
    from ._cohort_rs import increment_cohort

    if isinstance(base, pa.Table):
        base = base.combine_chunks().to_batches()[0] if base.num_rows > 0 else pa.record_batch([], schema=base.schema)

    delete_key_lists = []
    add_batches = []
    for delete_keys, adds in increments:
        delete_key_lists.append(delete_keys)
        if adds is not None:
            if isinstance(adds, pa.Table):
                adds = adds.combine_chunks().to_batches()[0] if adds.num_rows > 0 else None
            add_batches.append(adds)
        else:
            add_batches.append(None)

    result_batches = increment_cohort(base, key_column, delete_key_lists, add_batches)
    return pa.Table.from_batches(result_batches)


def partition(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    num_partitions: int | None = None,
    target_rows: int | None = None,
) -> list[str]:
    """Split a .cohort file into N smaller partition files for parallel queries.

    Returns the list of output file paths created.
    """
    from ._cohort_rs import partition_cohort

    return partition_cohort(str(input_path), str(output_dir), num_partitions, target_rows)


__all__ = [
    "annotate",
    "increment",
    "append",
    "delete_dataset",
    "delete_rows",
    "delete_selectors",
    "diff",
    "diff_revisions",
    "inspect",
    "intersect",
    "merge",
    "partition",
    "read",
    "union",
    "upsert",
    "write",
    "ByteRange",
    "CohortDefinition",
    "CohortInspection",
    "CohortReader",
    "CohortWriter",
    "Delta",
    "Lineage",
    "RevisionEntry",
    "Selector",
    "IcebergBackend",
    "IcebergConnection",
    "IcebergSnapshot",
    "SnowflakeConnection",
    "CohortCatalog",
    "Tag",
    "TagDirectory",
]

"""CohortWriter — creates .cohort v3 files using the Rust engine."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from .models import TagDirectory

from cohort_sdk._cohort_rs import write_cohort_arrow as _rs_write_cohort_arrow


class CohortWriter:
    """Write a .cohort v3 binary container file.

    RecordBatches are passed directly to Rust via Arrow's C Data Interface
    (zero-copy FFI). Parquet encoding, tag stamping, and binary layout are
    handled entirely by the Rust cohort-lib.
    """

    @staticmethod
    def write(
        path: str | Path,
        directory: TagDirectory,
        datasets: list[tuple[str, pa.Table]],
        *,
        codec: str = "parquet",
        index_column: str | None = None,
    ) -> None:
        """Write a .cohort file.

        Args:
            path: Output file path.
            directory: Tag directory with cohort_definition and patient_index.
            datasets: List of (label, pyarrow.Table) pairs to write as data blocks.
            codec: Encoding format — "parquet" (default), "arrow_ipc", "feather", or "vortex".
            index_column: Optional column name to build a Roaring bitmap patient index on.
        """
        # Stamp size on the latest revision entry if not already set
        total_rows = sum(table.num_rows for _, table in datasets)
        history = directory.cohort_definition.revision_history
        if history and history[-1].size is None:
            history[-1].size = total_rows

        # Convert tables to single RecordBatches for zero-copy FFI
        batch_datasets = []
        for label, table in datasets:
            batch = table.combine_chunks().to_batches()[0]
            batch_datasets.append((label, batch))

        # Serialize directory (without tags — Rust will compute them from block offsets)
        directory_json = directory.model_dump_json(exclude_none=True)

        _rs_write_cohort_arrow(str(path), directory_json, batch_datasets, codec, index_column)

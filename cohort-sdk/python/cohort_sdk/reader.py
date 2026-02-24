"""CohortReader — opens .cohort v3 files using the Rust engine."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from .models import CohortDefinition, RevisionEntry, TagDirectory

from cohort_sdk._cohort_rs import (
    apply_revision as _rs_apply_revision,
    open_cohort as _rs_open_cohort,
    read_dataset_arrow as _rs_read_dataset_arrow,
    read_dataset_bytes as _rs_read_dataset_bytes,
)


class CohortReader:
    """Read a .cohort v3 binary container file.

    The binary header parsing and validation is handled by the Rust cohort-lib
    via PyO3. Parquet decoding uses pyarrow on the raw bytes returned by Rust.

    Usage::

        cohort = CohortReader.open("data.cohort")
        cohort.definition        # CohortDefinition
        cohort.datasets        # ["Demographics", "Claims", ...]
        cohort["Demographics"]   # pa.Table
    """

    def __init__(self, directory: TagDirectory, path: Path):
        self.directory = directory
        self._path = path

    @classmethod
    def open(cls, path: str | Path) -> CohortReader:
        """Open a .cohort file. Validates magic bytes and version via Rust."""
        path = Path(path)
        directory_json = _rs_open_cohort(str(path))
        directory = TagDirectory.model_validate_json(directory_json)
        return cls(directory=directory, path=path)

    @property
    def definition(self) -> CohortDefinition:
        return self.directory.cohort_definition

    # Keep old name as alias for backwards compat
    @property
    def cohort_definition(self) -> CohortDefinition:
        return self.definition

    @property
    def datasets(self) -> list[str]:
        """Labels of all data blocks in the file."""
        return [t.label for t in self.directory.tags]

    def get_dataset_data(self, label: str) -> pa.Table:
        """Read a data block by label, returning a PyArrow Table.

        Uses the Rust codec-aware reader (zero-copy Arrow FFI), so this works
        for Parquet, Arrow IPC, and Feather encoded blocks.
        """
        batch = _rs_read_dataset_arrow(str(self._path), label)
        return pa.Table.from_batches([batch])

    def __getitem__(self, label: str) -> pa.Table:
        """Read a data block by label: ``cohort["Demographics"]``."""
        return self.get_dataset_data(label)

    def list_datasets(self) -> list[str]:
        """Return labels of all data blocks in the file."""
        return self.datasets


def _apply_revision_rs(
    definition: CohortDefinition, entry: RevisionEntry
) -> CohortDefinition:
    """Apply a revision via the Rust engine. Used by CohortDefinition.apply_revision()."""
    def_json = definition.model_dump_json()
    rev_json = entry.model_dump_json()
    updated_json = _rs_apply_revision(def_json, rev_json)
    return CohortDefinition.model_validate_json(updated_json)

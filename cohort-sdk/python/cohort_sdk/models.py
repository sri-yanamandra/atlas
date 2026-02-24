"""Pydantic models for the .cohort v3 tag directory and cohort definition.

These mirror the Rust structs in cohort-lib and are used for the Python-side API.
Serialization to/from JSON is handled by Pydantic to pass data across the PyO3 boundary.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import pyarrow as pa

    from .reader import CohortReader


class ByteRange(BaseModel):
    offset: int = 0
    length: int = 0


class TagColumnInfo(BaseModel):
    name: str
    data_type: str
    nullable: bool


class Tag(BaseModel):
    tag_id: int = 0
    label: str
    codec: str = "parquet"
    offset: int = 0
    length: int = 0
    schema_type: str | None = None
    num_rows: int | None = None
    num_columns: int | None = None
    columns: list[TagColumnInfo] | None = None


class Selector(BaseModel):
    """A single filter criterion in a cohort definition.

    Supported operators:
        - ``in``        — field value is in the list (values: ["E11.0", "E11.1"])
        - ``not_in``    — field value is NOT in the list
        - ``eq``        — equals a single value (values: ["CA"])
        - ``neq``       — not equal to a single value
        - ``gt``        — greater than (values: [65])
        - ``gte``       — greater than or equal
        - ``lt``        — less than
        - ``lte``       — less than or equal
        - ``between``   — inclusive range (values: [min, max])
    """

    selector_id: str
    field: str
    code_system: str | None = None
    operator: str
    values: Any
    label: str | None = None
    is_exclusion: bool = False


class Lineage(BaseModel):
    source_system: str | list[str]
    source_tables: list[str]
    query_hash: str | None = None
    extraction_date: str
    generation_job_id: str | None = None

    @classmethod
    def empty(cls) -> Lineage:
        return cls(source_system=[], source_tables=[], extraction_date="")


class Delta(BaseModel):
    action: str  # "add", "remove", "modify"
    selector_id: str
    field: str | None = None
    before: Any = None
    after: Any = None
    selector: Selector | None = None


class RevisionEntry(BaseModel):
    revision: int
    timestamp: datetime
    author: str
    message: str
    deltas: list[Delta] = Field(default_factory=list)
    size: int | None = None


class CohortDefinition(BaseModel):
    revision: int = 1
    name: str
    description: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""
    logic_operator: str = "AND"
    selectors: list[Selector] = Field(default_factory=list)
    lineage: Lineage = Field(default_factory=Lineage.empty)
    revision_history: list[RevisionEntry] = Field(default_factory=list)
    primary_keys: list[str] = Field(default_factory=list)

    @classmethod
    def read(cls, path: str | Path) -> "CohortReader":
        """Open a .cohort file for reading.

        Returns a CohortReader::

            cohort = CohortDefinition.read("f.cohort")
            cohort.definition
            cohort.datasets
            cohort["Claims-0SPG0JZ"]
        """
        from .reader import CohortReader

        return CohortReader.open(path)

    def write(
        self,
        path: str | Path,
        datasets: list[tuple[str, pa.Table]],
        *,
        codec: str = "parquet",
    ) -> None:
        """Write this cohort definition + data to a .cohort file."""
        from .writer import CohortWriter

        directory = TagDirectory(cohort_definition=self)
        CohortWriter.write(path, directory, datasets, codec=codec)

    @classmethod
    def inspect(cls, path: str | Path) -> "CohortInspection":
        """Inspect a .cohort file and return its metadata without reading data.

        Returns a CohortInspection with file metadata, definition summary,
        and per-dataset stats::

            info = CohortDefinition.inspect("f.cohort")
            info.file_metadata.file_size_bytes
            info.cohort_definition.name
            info.datasets[0].num_rows
        """
        from ._cohort_rs import inspect_cohort

        json_str = inspect_cohort(str(path))
        return CohortInspection.model_validate_json(json_str)

    def apply_revision(self, entry: RevisionEntry) -> None:
        """Apply a revision via the Rust engine and update this instance."""
        from .reader import _apply_revision_rs

        updated = _apply_revision_rs(self, entry)
        self.revision = updated.revision
        self.selectors = updated.selectors
        self.revision_history = updated.revision_history

    def __str__(self) -> str:
        src = self.lineage.source_system
        if isinstance(src, list):
            src = ", ".join(src) if src else ""
        lines = [
            f"{self.name} (r{self.revision})",
            f"  Selectors:  {[f'{s.field} {s.operator} {s.values}' for s in self.selectors]}",
            f"  Lineage:    {src} — {self.lineage.source_tables}",
            "  History:",
        ]
        for r in self.revision_history:
            size_str = f" — {r.size:,} rows" if r.size is not None else ""
            lines.append(f"    r{r.revision}: {r.message} ({r.author}){size_str}")
        if not self.revision_history:
            lines.append("    (none)")
        return "\n".join(lines)


class IcebergSnapshot(BaseModel):
    snapshot_id: int
    timestamp_ms: int
    operation: str | None = None
    total_records: int | None = None
    total_file_size: int | None = None


class IcebergTableRef(BaseModel):
    """Per-dataset Iceberg table reference for multi-dataset cohorts."""
    dataset_label: str
    table_name: str
    table_location: str
    snapshot_id: int | None = None
    snapshots: list[IcebergSnapshot] = Field(default_factory=list)


class IcebergBackend(BaseModel):
    catalog_name: str = "glue"
    region: str = "us-east-1"
    database: str
    table_name: str
    table_location: str
    snapshot_id: int | None = None
    snapshots: list[IcebergSnapshot] = Field(default_factory=list)
    tables: list[IcebergTableRef] = Field(default_factory=list)


class TagDirectory(BaseModel):
    tags: list[Tag] = Field(default_factory=list)
    patient_index: ByteRange = Field(default_factory=ByteRange)
    cohort_definition: CohortDefinition
    backend: IcebergBackend | None = None


# ── Inspection models ──────────────────────────────────────────

class ColumnInfo(BaseModel):
    name: str
    data_type: str
    nullable: bool


class DatasetInfo(BaseModel):
    label: str
    codec: str = "parquet"
    compressed_size: int
    num_rows: int
    num_columns: int
    columns: list[ColumnInfo] = Field(default_factory=list)


class SelectorSummary(BaseModel):
    selector_id: str
    field: str
    operator: str
    values: Any = None
    label: str | None = None
    is_exclusion: bool = False


class LineageSummary(BaseModel):
    source_system: str | list[str]
    source_tables: list[str] = Field(default_factory=list)
    extraction_date: str


class RevisionSummary(BaseModel):
    revision: int
    author: str
    message: str
    size: int | None = None


class CohortDefinitionSummary(BaseModel):
    name: str
    description: str | None = None
    created_at: str
    created_by: str
    revision: int
    selectors: list[SelectorSummary] = Field(default_factory=list)
    lineage: LineageSummary
    revision_history: list[RevisionSummary] = Field(default_factory=list)
    primary_keys: list[str] = Field(default_factory=list)


class FileMetadata(BaseModel):
    path: str
    file_size_bytes: int
    format_version: int
    num_datasets: int


class CohortInspection(BaseModel):
    file_metadata: FileMetadata
    cohort_definition: CohortDefinitionSummary
    datasets: list[DatasetInfo] = Field(default_factory=list)
    backend: IcebergBackend | None = None

    def __str__(self) -> str:
        defn = self.cohort_definition
        src = defn.lineage.source_system
        if isinstance(src, list):
            src = ", ".join(src) if src else ""
        lines = [
            defn.name,
            f"  File:       {self.file_metadata.path} ({self.file_metadata.file_size_bytes:,} bytes)",
            f"  v{self.file_metadata.format_version}.{defn.revision}",
            f"  Created by: {defn.created_by} at {defn.created_at}",
        ]
        if defn.description:
            lines.append(f"  Desc:       {defn.description}")
        lines.append(f"  Selectors:  {[f'{s.field} {s.operator} {s.values}' for s in defn.selectors]}")
        lines.append(f"  Lineage:    {src} — {defn.lineage.source_tables}")
        lines.append("  History:")
        for r in defn.revision_history:
            size_str = f" — {r.size:,} rows" if r.size is not None else ""
            lines.append(f"    r{r.revision}: {r.message} ({r.author}){size_str}")
        if not defn.revision_history:
            lines.append("    (none)")
        for mod in self.datasets:
            lines.append(f"  Dataset:   {mod.label} — {mod.num_rows} rows x {mod.num_columns} cols ({mod.compressed_size:,} bytes)")
            col_names = [c.name for c in mod.columns[:8]]
            suffix = "..." if len(mod.columns) > 8 else ""
            lines.append(f"  Columns:    {col_names}{suffix}")
        if not self.datasets:
            lines.append("  (no datasets)")
        if self.backend:
            from datetime import datetime, timezone

            def _fmt_snapshots(snapshots: list, indent: str = "    ") -> None:
                for snap in snapshots:
                    ts = datetime.fromtimestamp(snap.timestamp_ms / 1000, tz=timezone.utc)
                    op = snap.operation or "unknown"
                    records = f" — {snap.total_records:,} records" if snap.total_records is not None else ""
                    size = f", {snap.total_file_size:,} bytes" if snap.total_file_size is not None else ""
                    lines.append(f"{indent}{snap.snapshot_id}: {op} at {ts.isoformat()}{records}{size}")

            if self.backend.tables:
                # Multi-dataset: show per-table info
                lines.append(f"  Backend:    Iceberg — {self.backend.database} ({len(self.backend.tables)} tables)")
                for tref in self.backend.tables:
                    lines.append(f"  Table:      {self.backend.database}.{tref.table_name} ({tref.dataset_label})")
                    lines.append(f"    Location: {tref.table_location}")
                    if tref.snapshot_id is not None:
                        lines.append(f"    Snapshot: {tref.snapshot_id}")
                    if tref.snapshots:
                        lines.append("    Snapshots:")
                        _fmt_snapshots(tref.snapshots, indent="      ")
            else:
                # Single-dataset
                lines.append(f"  Backend:    Iceberg — {self.backend.database}.{self.backend.table_name}")
                lines.append(f"  Location:   {self.backend.table_location}")
                if self.backend.snapshot_id is not None:
                    lines.append(f"  Snapshot:   {self.backend.snapshot_id}")
                if self.backend.snapshots:
                    lines.append("  Snapshots:")
                    _fmt_snapshots(self.backend.snapshots)
        return "\n".join(lines)

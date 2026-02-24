"""Iceberg connector — read Iceberg tables via AWS Glue Catalog and pipe results into .cohort files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
from pyiceberg.catalog.glue import GlueCatalog
from pyiceberg.table import Table as IcebergTable

from .models import (
    CohortDefinition,
    IcebergBackend,
    IcebergSnapshot,
    Lineage,
    RevisionEntry,
    Selector,
    Tag,
    TagColumnInfo,
    TagDirectory,
)
from .writer import CohortWriter

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog


@dataclass
class UpsertOutcome:
    """Result of an upsert operation."""

    table: IcebergTable
    rows_updated: int
    rows_inserted: int


class IcebergConnection:
    """Manage an Iceberg catalog connection for extracting cohort data into .cohort files.

    Uses AWS Glue Catalog by default. Authentication follows the standard boto3
    credential chain (env vars, ``~/.aws/credentials``, IAM role).
    """

    def __init__(
        self,
        *,
        catalog_name: str = "glue",
        region: str | None = None,
        catalog: Catalog | None = None,
    ):
        """Create a connection to an Iceberg catalog.

        Args:
            catalog_name: Name for the Glue catalog (used internally by PyIceberg).
            region: AWS region. Falls back to ``AWS_DEFAULT_REGION``, then ``us-east-1``.
            catalog: Pass an existing PyIceberg ``Catalog`` to skip auto-creation.
        """
        if catalog is not None:
            self._catalog = catalog
        else:
            resolved_region = (
                region
                or os.environ.get("AWS_DEFAULT_REGION")
                or "us-east-1"
            )
            self._catalog = GlueCatalog(
                catalog_name,
                **{
                    "region_name": resolved_region,
                    "s3.region": resolved_region,
                },
            )

    def load_table(self, database: str, table: str) -> IcebergTable:
        """Load an Iceberg table object (for metadata inspection).

        Args:
            database: Glue database / Iceberg namespace.
            table: Table name.
        """
        return self._catalog.load_table(f"{database}.{table}")

    def query_table(
        self, database: str, table: str, limit: int | None = None
    ) -> pa.Table:
        """Read an Iceberg table as a PyArrow Table.

        Args:
            database: Glue database / Iceberg namespace.
            table: Table name.
            limit: Optional row limit.
        """
        iceberg_table = self.load_table(database, table)
        result = iceberg_table.scan().to_arrow()
        if limit is not None:
            result = result.slice(0, limit)
        return result

    @staticmethod
    def _collect_snapshots(iceberg_table: IcebergTable) -> list[IcebergSnapshot]:
        """Build a list of IcebergSnapshot from all snapshots on an Iceberg table."""
        snapshots = []
        for snap in iceberg_table.snapshots():
            summary = snap.summary
            snapshots.append(IcebergSnapshot(
                snapshot_id=snap.snapshot_id,
                timestamp_ms=snap.timestamp_ms,
                operation=str(summary.operation) if summary else None,
                total_records=int(summary["total-records"] or 0) if summary else None,
                total_file_size=int(summary["total-files-size"] or 0) if summary else None,
            ))
        return snapshots

    def list_tables(self, database: str) -> list[str]:
        """List all tables in a database/namespace.

        Args:
            database: Glue database / Iceberg namespace.
        """
        return [
            t[1] if isinstance(t, tuple) else str(t)
            for t in self._catalog.list_tables(database)
        ]

    def list_namespaces(self) -> list[str]:
        """List all namespaces in the catalog."""
        return [
            ns[0] if isinstance(ns, tuple) else str(ns)
            for ns in self._catalog.list_namespaces()
        ]

    def write_table(
        self,
        database: str,
        table_name: str,
        data: pa.Table,
        *,
        location: str | None = None,
        mode: str = "overwrite",
    ) -> IcebergTable:
        """Write a PyArrow Table to an Iceberg table. Creates if needed.

        Args:
            database: Glue database / Iceberg namespace.
            table_name: Table name.
            data: PyArrow Table to write.
            location: S3 location for the table (e.g. ``s3://bucket/path``).
                Required when the table does not yet exist.
            mode: ``"overwrite"`` (default) or ``"append"``.

        Returns:
            The Iceberg table object after writing.
        """
        # Iceberg requires microsecond timestamps — downcast any nanosecond columns
        new_fields = []
        needs_cast = False
        for field in data.schema:
            if pa.types.is_timestamp(field.type) and field.type.unit == "ns":
                tz = field.type.tz
                new_fields.append(field.with_type(pa.timestamp("us", tz=tz)))
                needs_cast = True
            else:
                new_fields.append(field)
        if needs_cast:
            data = data.cast(pa.schema(new_fields))

        identifier = f"{database}.{table_name}"
        try:
            table = self._catalog.load_table(identifier)
        except Exception:
            if location is None:
                raise ValueError(
                    f"Table {identifier!r} does not exist and no location was provided."
                )
            table = self._catalog.create_table(
                identifier,
                schema=data.schema,
                location=location,
            )

        if mode == "overwrite":
            table.overwrite(data)
        elif mode == "append":
            table.append(data)
        else:
            raise ValueError(f"Unsupported write mode: {mode!r}")

        return table

    def upsert_table(
        self,
        database: str,
        table_name: str,
        data: pa.Table,
        *,
        join_cols: list[str],
        location: str | None = None,
    ) -> UpsertOutcome:
        """Upsert a PyArrow Table into an Iceberg table. Creates if needed.

        Args:
            database: Glue database / Iceberg namespace.
            table_name: Table name.
            data: PyArrow Table to upsert.
            join_cols: Column(s) to match rows on (primary key columns).
            location: S3 location for the table. Required when the table
                does not yet exist.

        Returns:
            UpsertOutcome with the Iceberg table and row counts.
        """
        new_fields = []
        needs_cast = False
        for field in data.schema:
            if pa.types.is_timestamp(field.type) and field.type.unit == "ns":
                tz = field.type.tz
                new_fields.append(field.with_type(pa.timestamp("us", tz=tz)))
                needs_cast = True
            else:
                new_fields.append(field)
        if needs_cast:
            data = data.cast(pa.schema(new_fields))

        identifier = f"{database}.{table_name}"
        try:
            table = self._catalog.load_table(identifier)
        except Exception:
            if location is None:
                raise ValueError(
                    f"Table {identifier!r} does not exist and no location was provided."
                )
            table = self._catalog.create_table(
                identifier,
                schema=data.schema,
                location=location,
            )

        result = table.upsert(data, join_cols=join_cols)

        return UpsertOutcome(
            table=table,
            rows_updated=result.rows_updated,
            rows_inserted=result.rows_inserted,
        )

    def load_to_cohort(
        self,
        path: str | Path,
        cohort_name: str,
        data: pa.Table,
        *,
        database: str,
        table_name: str,
        s3_location: str,
        created_by: str = "",
        description: str | None = None,
        selectors: list[Selector] | None = None,
    ) -> None:
        """Write data to Iceberg and produce a metadata-only .cohort file.

        Args:
            path: Output .cohort file path.
            cohort_name: Name for the cohort definition.
            data: PyArrow Table to write to Iceberg.
            database: Glue database / Iceberg namespace.
            table_name: Iceberg table name.
            s3_location: S3 location for the Iceberg table.
            created_by: Author identifier.
            description: Optional cohort description.
            selectors: Optional list of selectors for the cohort definition.
        """
        now = datetime.now(timezone.utc)

        # 1. Write data to Iceberg
        table = self.write_table(database, table_name, data, location=s3_location)

        # 2. Capture snapshot ID, total file size, and full snapshot history
        snapshot = table.current_snapshot()
        snapshot_id = snapshot.snapshot_id if snapshot else None
        total_file_size = 0
        if snapshot and snapshot.summary:
            total_file_size = int(snapshot.summary["total-files-size"] or 0)

        iceberg_snapshots = self._collect_snapshots(table)

        # 3. Build cohort definition
        definition = CohortDefinition(
            name=cohort_name,
            description=description,
            created_at=now,
            created_by=created_by,
            selectors=selectors or [],
            lineage=Lineage(
                source_system=["Snowflake", "Iceberg"],
                source_tables=[f"{database}.{table_name}"],
                extraction_date=now.strftime("%Y-%m-%d"),
            ),
            revision_history=[
                RevisionEntry(
                    revision=1,
                    timestamp=now,
                    author=created_by,
                    message=f"Initial load to Iceberg: {database}.{table_name}",
                    size=data.num_rows,
                )
            ],
        )

        # 4. Build TagDirectory with backend and dataset metadata tag
        catalog_name = self._catalog.name if hasattr(self._catalog, "name") else "glue"
        region = (
            self._catalog.properties.get("region_name", "us-east-1")
            if hasattr(self._catalog, "properties")
            else "us-east-1"
        )
        backend = IcebergBackend(
            catalog_name=catalog_name,
            region=region,
            database=database,
            table_name=table_name,
            table_location=s3_location,
            snapshot_id=snapshot_id,
            snapshots=iceberg_snapshots,
        )
        # Metadata-only tag so inspect can report schema/row info
        columns = [
            TagColumnInfo(
                name=field.name,
                data_type=str(field.type),
                nullable=field.nullable,
            )
            for field in data.schema
        ]
        dataset_tag = Tag(
            label=table_name,
            codec="iceberg",
            length=total_file_size,
            num_rows=data.num_rows,
            num_columns=len(data.schema),
            columns=columns,
        )
        directory = TagDirectory(
            tags=[dataset_tag],
            cohort_definition=definition,
            backend=backend,
        )

        # 5. Write metadata-only .cohort (no embedded data)
        CohortWriter.write(path, directory, [])

    def extract_to_cohort(
        self,
        path: str | Path,
        cohort_name: str,
        tables: dict[str, tuple[str, str]],
        *,
        created_by: str = "",
        description: str | None = None,
        limit: int | None = None,
    ) -> None:
        """Read Iceberg tables and write results to a .cohort file.

        Args:
            path: Output .cohort file path.
            cohort_name: Name for the cohort definition.
            tables: Dict of ``{label: (database, table_name)}`` for each dataset.
            created_by: Author identifier.
            description: Optional cohort description.
            limit: Optional row limit applied to each table scan.
        """
        now = datetime.now(timezone.utc)

        datasets: list[tuple[str, pa.Table]] = []
        source_tables: list[str] = []
        for label, (database, table_name) in tables.items():
            arrow_table = self.query_table(database, table_name, limit=limit)
            datasets.append((label, arrow_table))
            source_tables.append(f"{database}.{table_name}")

        definition = CohortDefinition(
            name=cohort_name,
            description=description,
            created_at=now,
            created_by=created_by,
            lineage=Lineage(
                source_system="Iceberg",
                source_tables=source_tables,
                extraction_date=now.strftime("%Y-%m-%d"),
            ),
            revision_history=[
                RevisionEntry(
                    revision=1,
                    timestamp=now,
                    author=created_by,
                    message="Initial extraction from Iceberg",
                )
            ],
        )

        directory = TagDirectory(cohort_definition=definition)
        CohortWriter.write(path, directory, datasets)

    def close(self) -> None:
        """Close the connection (no-op for Glue catalog, provided for interface consistency)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

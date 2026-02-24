"""Snowflake connector — query Snowflake tables and pipe results into .cohort files."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import snowflake.connector

from .models import (
    CohortDefinition,
    Lineage,
    RevisionEntry,
    TagDirectory,
)
from .writer import CohortWriter


def _safe_identifier(name: str) -> str:
    """Validate and sanitize a Snowflake identifier to prevent injection."""
    if not re.match(r"^[A-Za-z0-9_.]+$", name):
        raise ValueError(f"Invalid Snowflake identifier: {name!r}")
    return name


class SnowflakeConnection:
    """Manage a Snowflake connection for extracting cohort data into .cohort files."""

    def __init__(
        self,
        *,
        account: str | None = None,
        user: str | None = None,
        warehouse: str | None = None,
        authenticator: str = "externalbrowser",
        conn: snowflake.connector.SnowflakeConnection | None = None,
    ):
        """Create a connection. Either pass an existing ``conn`` or provide credentials.

        Falls back to ``SNOWFLAKE_ACCOUNT``, ``SNOWFLAKE_USER``, ``SNOWFLAKE_WAREHOUSE``
        environment variables.
        """
        if conn is not None:
            self._conn = conn
            self._owned = False
        else:
            self._conn = snowflake.connector.connect(
                account=account or os.environ["SNOWFLAKE_ACCOUNT"],
                user=user or os.environ["SNOWFLAKE_USER"],
                warehouse=warehouse or os.environ.get("SNOWFLAKE_WAREHOUSE"),
                authenticator=authenticator,
            )
            self._owned = True

    def query_arrow(self, sql: str, params: dict | None = None) -> pa.Table:
        """Execute a SQL query and return results as a PyArrow Table."""
        cur = self._conn.cursor()
        try:
            if params:
                cur.execute(sql, params)
            else:
                cur.execute(sql)
            return cur.fetch_arrow_all()
        finally:
            cur.close()

    def query_table(
        self, database: str, schema: str, table: str, limit: int | None = None
    ) -> pa.Table:
        """Read a full table (or first N rows) as a PyArrow Table."""
        fqn = (
            f"{_safe_identifier(database)}"
            f".{_safe_identifier(schema)}"
            f".{_safe_identifier(table)}"
        )
        sql = f"SELECT * FROM {fqn}"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return self.query_arrow(sql)

    def extract_to_cohort(
        self,
        path: str | Path,
        cohort_name: str,
        queries: dict[str, str],
        *,
        created_by: str = "",
        description: str | None = None,
        source_tables: list[str] | None = None,
    ) -> None:
        """Run queries against Snowflake and write results to a .cohort file.

        Args:
            path: Output .cohort file path.
            cohort_name: Name for the cohort definition.
            queries: Dict of ``{label: SQL}`` for each dataset to include.
            created_by: Author identifier.
            description: Optional cohort description.
            source_tables: Tables to record in lineage (defaults to query labels).
        """
        now = datetime.now(timezone.utc)

        # Execute all queries
        datasets: list[tuple[str, pa.Table]] = []
        for label, sql in queries.items():
            table = self.query_arrow(sql)
            datasets.append((label, table))

        # Build cohort definition
        definition = CohortDefinition(
            name=cohort_name,
            description=description,
            created_at=now,
            created_by=created_by,
            lineage=Lineage(
                source_system="Snowflake",
                source_tables=source_tables or list(queries.keys()),
                extraction_date=now.strftime("%Y-%m-%d"),
            ),
            revision_history=[
                RevisionEntry(
                    revision=1,
                    timestamp=now,
                    author=created_by,
                    message="Initial extraction from Snowflake",
                )
            ],
        )

        directory = TagDirectory(cohort_definition=definition)
        CohortWriter.write(path, directory, datasets)

    def close(self) -> None:
        if self._owned:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

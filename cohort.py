"""cohort.py — CLI for .cohort files.

Subcommands:
    inspect    <file> [--json]                        Show metadata for a .cohort file
    merge      <file1> <file2> [...] -o <out> [--name]   Union multiple .cohort files
    intersect  <file1> <file2> [...] -o <out> --on COL [--name]  Intersect .cohort files
    annotate   <file> [--name] [--description] [-m]   Modify metadata
    diff       <file1> [<file2>] [--rev A B] [--json] Compare files or revisions
    append     <file> --dataset D --data F [-m]        Append rows from Parquet/CSV
    delete     <file> --key K --values V [...] [-m]   Delete rows, datasets, or selectors
    upsert     <file> --dataset D --key K --data F    Update-or-insert rows
    sonify     <file> [-o OUT] [--melody-col] [--bass-col] [--bpm]  8-bit audio from cohort data
    load       <spec.py>                              Load cohorts from a Python manifest
    discover   <database> [--schema] [--detailed]     Crawl Snowflake inventory
    generate   <config.yaml> [--output]               Generate manifests from YAML split config

Lakehouse commands:
    search     <universe> <query> [--top-k N]         Semantic search for patients
    similar    <universe> <patient-id> [--top-k N]    Find similar patients
    index      <file> [--universe] [--vector] [--bitmap]  Build vector/bitmap indexes
    cohorts    [--list|--intersect|--union|--diff]    Bitmap cohort operations
    policy     [--list|--create|--delete]             Manage ABAC policies
    audit      [--limit N] [--user U]                 View audit logs
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from cohort_sdk import (
    CohortDefinition,
    CohortReader,
    CohortWriter,
    IcebergConnection,
    Lineage,
    RevisionEntry,
    Selector,
    annotate,
    append,
    delete_dataset,
    delete_rows,
    delete_selectors,
    diff,
    diff_revisions,
    inspect,
    intersect,
    union,
    upsert,
)


# ── Spec models ───────────────────────────────────────────────


class IncrementalStep(BaseModel):
    """One incremental step: delete keys from a deletes table, add rows from an adds table."""

    deletes_table: str | None = None
    adds_table: str | None = None
    key_column: str = "patient_number"
    message: str = ""


class MergeDataset(BaseModel):
    """An additional dataset to merge into the same .cohort output."""

    table: str
    dataset: str


class CohortSpec(BaseModel):
    """Specification for a single cohort to extract from Snowflake."""

    table: str
    output: str
    name: str
    description: str | None = None
    dataset: str
    selectors: list[Selector] = []
    increments: list[IncrementalStep] = []
    merge_datasets: list[MergeDataset] = []


class IcebergDestination(BaseModel):
    """Iceberg destination config for writing cohort data to Iceberg tables."""

    region: str = "us-west-2"
    database: str
    s3_base: str


class CohortManifest(BaseModel):
    """Top-level manifest describing a set of cohorts to build."""

    database: str
    schema_name: str
    iceberg: IcebergDestination | None = None
    cohorts: list[CohortSpec]


# ── Subcommand handlers ──────────────────────────────────────


def cmd_inspect(args: argparse.Namespace) -> None:
    """Inspect a .cohort file and print its metadata."""
    info = inspect(args.file)
    if args.json_output:
        print(info.model_dump_json(indent=2))
    else:
        print(info)


def cmd_merge(args: argparse.Namespace) -> None:
    """Union multiple .cohort files into one."""
    from cohort_sdk.models import IcebergBackend, IcebergTableRef, Tag, TagColumnInfo

    # Collect Iceberg backend info from all inputs before merging
    readers = [CohortReader.open(f) for f in args.files]
    all_iceberg = all(r.directory.backend is not None for r in readers)
    first_backend = readers[0].directory.backend

    # Do the local merge for definition merging (selectors, lineage, history)
    union(args.files, args.out, name=args.name)

    if all_iceberg:
        # All inputs are Iceberg-backed — just reference existing tables,
        # no data movement needed.
        conn = IcebergConnection(
            catalog_name=first_backend.catalog_name,
            region=first_backend.region,
        )

        merged_reader = CohortReader.open(args.out)
        directory = merged_reader.directory

        new_tags = []
        table_refs = []

        for reader in readers:
            backend = reader.directory.backend
            # Collect refs from single-table or multi-table backends
            if backend.tables:
                refs_to_process = [
                    (tr.dataset_label, tr.table_name, tr.table_location)
                    for tr in backend.tables
                ]
            else:
                label = reader.directory.tags[0].label if reader.directory.tags else backend.table_name
                refs_to_process = [(label, backend.table_name, backend.table_location)]

            for label, table_name, table_location in refs_to_process:
                # Skip if we already have this table (dedup by table_name)
                if any(tr.table_name == table_name for tr in table_refs):
                    continue

                iceberg_table = conn.load_table(backend.database, table_name)
                snapshot = iceberg_table.current_snapshot()
                total_file_size = 0
                total_records = 0
                if snapshot and snapshot.summary:
                    total_file_size = int(snapshot.summary["total-files-size"] or 0)
                    total_records = int(snapshot.summary["total-records"] or 0)

                # Build tag from Iceberg table schema
                arrow_schema = iceberg_table.schema()
                columns = [
                    TagColumnInfo(
                        name=field.name,
                        data_type=str(field.field_type),
                        nullable=field.optional,
                    )
                    for field in arrow_schema.fields
                ]
                new_tags.append(Tag(
                    label=label,
                    codec="iceberg",
                    length=total_file_size,
                    num_rows=total_records,
                    num_columns=len(columns),
                    columns=columns,
                ))
                table_refs.append(IcebergTableRef(
                    dataset_label=label,
                    table_name=table_name,
                    table_location=table_location,
                    snapshot_id=snapshot.snapshot_id if snapshot else None,
                    snapshots=conn._collect_snapshots(iceberg_table),
                ))

        first_ref = table_refs[0]
        backend = IcebergBackend(
            catalog_name=first_backend.catalog_name,
            region=first_backend.region,
            database=first_backend.database,
            table_name=first_ref.table_name,
            table_location=first_ref.table_location,
            snapshot_id=first_ref.snapshot_id,
            snapshots=first_ref.snapshots,
            tables=table_refs if len(table_refs) > 1 else [],
        )
        directory.tags = new_tags
        directory.backend = backend
        CohortWriter.write(args.out, directory, [])

    print(f"Merged {len(args.files)} files → {args.out}")
    print(inspect(args.out))


def cmd_intersect(args: argparse.Namespace) -> None:
    """Intersect rows across multiple .cohort files."""
    intersect(args.files, args.out, on=args.on, name=args.name)
    print(f"Intersected {len(args.files)} files → {args.out}")
    print(inspect(args.out))


def cmd_annotate(args: argparse.Namespace) -> None:
    """Annotate a .cohort file (modify name, description, selectors)."""
    annotate(
        args.file,
        output=args.output,
        name=args.name,
        description=args.description,
        remove_selector_ids=args.remove_selectors or None,
        message=args.message,
        author=args.author,
    )
    out = args.output or args.file
    print(f"Annotated → {out}")
    print(inspect(out))


def cmd_diff(args: argparse.Namespace) -> None:
    """Compare two .cohort files or two revisions."""
    import json as json_mod

    if args.rev:
        d = diff_revisions(args.file, args.rev[0], args.rev[1])
    elif args.file2:
        d = diff(args.file, args.file2)
    else:
        print("Error: provide either two files or --rev A B", file=sys.stderr)
        sys.exit(1)

    if args.json_output:
        print(json_mod.dumps(d, indent=2))
    else:
        print(_format_diff(d))


def _format_diff(d: dict) -> str:
    """Format a diff dict for human-readable output."""
    lines = []
    lines.append(f"Comparing: {d['left_name']} (r{d['left_revision']}) vs {d['right_name']} (r{d['right_revision']})")
    lines.append("")

    if nc := d.get("name_changed"):
        lines.append(f"  Name:        '{nc['left']}' → '{nc['right']}'")
    if dc := d.get("description_changed"):
        lines.append(f"  Description: {dc['left']!r} → {dc['right']!r}")

    sel = d.get("selectors", {})
    added = sel.get("added", [])
    removed = sel.get("removed", [])
    modified = sel.get("modified", [])
    if added or removed or modified:
        lines.append("  Selectors:")
        for s in added:
            lines.append(f"    + {s['selector_id']} ({s['field']} {s['operator']} {s['values']})")
        for s in removed:
            lines.append(f"    - {s['selector_id']} ({s['field']} {s['operator']} {s['values']})")
        for c in modified:
            lines.append(f"    ~ {c['selector_id']} (changed)")
    else:
        lines.append("  Selectors:   (no changes)")

    if d.get("lineage_changed"):
        lines.append("  Lineage:     changed")

    lines.append("  Datasets:")
    for m in d.get("datasets", []):
        status = m["status"]
        if status == "same":
            lines.append(f"    = {m['label']} ({m.get('left_rows', 0)} rows)")
        elif status == "changed":
            lines.append(f"    ~ {m['label']} ({m.get('left_rows', 0)} → {m.get('right_rows', 0)} rows)")
        elif status == "added":
            lines.append(f"    + {m['label']} ({m.get('right_rows', 0)} rows)")
        elif status == "removed":
            lines.append(f"    - {m['label']} ({m.get('left_rows', 0)} rows)")

    return "\n".join(lines)


def _read_data_file(data_path: Path) -> "pa.Table":
    """Read a Parquet, CSV, or .cohort data file into a PyArrow Table."""
    import pyarrow.parquet as pq
    import pyarrow.csv as csv

    if data_path.suffix == ".parquet":
        return pq.read_table(str(data_path))
    elif data_path.suffix == ".csv":
        return csv.read_csv(str(data_path))
    elif data_path.suffix == ".cohort":
        reader = CohortReader.open(str(data_path))
        backend = reader.directory.backend
        if backend is not None:
            # Iceberg-backed: read data from Iceberg, not from embedded blocks
            conn = IcebergConnection(
                catalog_name=backend.catalog_name,
                region=backend.region,
            )
            return conn.query_table(backend.database, backend.table_name)
        labels = reader.datasets
        if not labels:
            print("Error: source .cohort file has no datasets", file=sys.stderr)
            sys.exit(1)
        if len(labels) == 1:
            return reader.get_dataset_data(labels[0])
        print(
            f"Error: source .cohort has multiple datasets {labels} — "
            "use a single-dataset source or extract first",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print(f"Error: unsupported data format '{data_path.suffix}' (use .parquet, .csv, or .cohort)", file=sys.stderr)
        sys.exit(1)


def cmd_append(args: argparse.Namespace) -> None:
    """Append rows to a dataset from a data file.

    Auto-detects whether the target is Iceberg-backed and routes accordingly.
    """
    table = _read_data_file(Path(args.data))
    if table.num_rows == 0:
        print("Error: data file is empty", file=sys.stderr)
        sys.exit(1)

    # Check if target is Iceberg-backed
    target_reader = CohortReader.open(args.file)
    backend = target_reader.directory.backend

    if backend is not None:
        # ── Iceberg path ──────────────────────────────────────
        conn = IcebergConnection(
            catalog_name=backend.catalog_name,
            region=backend.region,
        )
        iceberg_table = conn.write_table(
            backend.database,
            backend.table_name,
            table,
            location=backend.table_location,
            mode="append",
        )

        now = datetime.now(timezone.utc)
        snapshot = iceberg_table.current_snapshot()
        backend.snapshot_id = snapshot.snapshot_id if snapshot else None
        backend.snapshots = conn._collect_snapshots(iceberg_table)

        directory = target_reader.directory
        history = directory.cohort_definition.revision_history
        next_rev = (history[-1].revision + 1) if history else 1
        total_rows = iceberg_table.scan().to_arrow().num_rows

        # Update dataset tag metadata to reflect current Iceberg state
        total_file_size = 0
        if snapshot and snapshot.summary:
            total_file_size = int(snapshot.summary["total-files-size"] or 0)
        for tag in directory.tags:
            if tag.label == backend.table_name:
                tag.num_rows = total_rows
                tag.length = total_file_size

        history.append(RevisionEntry(
            revision=next_rev,
            timestamp=now,
            author=args.author,
            message=args.message,
            size=total_rows,
        ))
        directory.cohort_definition.revision = next_rev
        CohortWriter.write(args.file, directory, [])

        print(
            f"Appended {table.num_rows} rows to Iceberg "
            f"{backend.database}.{backend.table_name} → {args.file}"
        )
    else:
        # ── Local path (Rust) ─────────────────────────────────
        if not args.dataset:
            print("Error: --dataset is required for local (non-Iceberg) cohorts", file=sys.stderr)
            sys.exit(1)

        batch = table.to_batches()[0] if table.num_rows > 0 else None
        if batch is None:
            print("Error: data file is empty", file=sys.stderr)
            sys.exit(1)

        append(
            args.file,
            batch,
            dataset_label=args.dataset,
            output=args.output,
            message=args.message,
            author=args.author,
        )
        out = args.output or args.file
        print(f"Appended {table.num_rows} rows to '{args.dataset}' → {out}")


def cmd_delete(args: argparse.Namespace) -> None:
    """Delete rows, a dataset, or selectors from a .cohort file."""
    if args.dataset:
        delete_dataset(
            args.file,
            args.dataset,
            output=args.output,
            message=args.message,
            author=args.author,
        )
        out = args.output or args.file
        print(f"Deleted dataset '{args.dataset}' → {out}")
    elif args.selectors:
        delete_selectors(
            args.file,
            args.selectors,
            output=args.output,
            message=args.message,
            author=args.author,
        )
        out = args.output or args.file
        print(f"Deleted {len(args.selectors)} selector(s) → {out}")
    elif args.key:
        if not args.values:
            print("Error: --values required when using --key", file=sys.stderr)
            sys.exit(1)
        delete_rows(
            args.file,
            args.key,
            args.values,
            output=args.output,
            message=args.message,
            author=args.author,
        )
        out = args.output or args.file
        print(f"Deleted rows with {args.key} in {args.values} → {out}")
    else:
        print("Error: specify --key/--values, --dataset, or --selector", file=sys.stderr)
        sys.exit(1)


def cmd_upsert(args: argparse.Namespace) -> None:
    """Update-or-insert rows from a data file.

    Auto-detects whether the target is Iceberg-backed and routes accordingly.
    """
    table = _read_data_file(Path(args.data))
    if table.num_rows == 0:
        print("Error: data file is empty", file=sys.stderr)
        sys.exit(1)

    # Check if target is Iceberg-backed
    target_reader = CohortReader.open(args.file)
    backend = target_reader.directory.backend

    if backend is not None:
        # ── Iceberg path ──────────────────────────────────────
        # Determine join columns: --key flag, or primary_keys from definition
        join_cols = [args.key] if args.key else target_reader.directory.cohort_definition.primary_keys
        if not join_cols:
            print(
                "Error: no --key provided and target .cohort has no primary_keys defined. "
                "Either pass --key or set primary_keys in the cohort definition.",
                file=sys.stderr,
            )
            sys.exit(1)

        conn = IcebergConnection(
            catalog_name=backend.catalog_name,
            region=backend.region,
        )
        outcome = conn.upsert_table(
            backend.database,
            backend.table_name,
            table,
            join_cols=join_cols,
            location=backend.table_location,
        )

        now = datetime.now(timezone.utc)
        snapshot = outcome.table.current_snapshot()
        backend.snapshot_id = snapshot.snapshot_id if snapshot else None

        directory = target_reader.directory
        history = directory.cohort_definition.revision_history
        next_rev = (history[-1].revision + 1) if history else 1
        total_rows = outcome.table.scan().to_arrow().num_rows

        history.append(RevisionEntry(
            revision=next_rev,
            timestamp=now,
            author=args.author,
            message=args.message,
            size=total_rows,
        ))
        directory.cohort_definition.revision = next_rev
        CohortWriter.write(args.file, directory, [])

        print(
            f"Upserted into Iceberg {backend.database}.{backend.table_name}: "
            f"{outcome.rows_updated} updated, {outcome.rows_inserted} inserted → {args.file}"
        )
    else:
        # ── Local path (Rust) ─────────────────────────────────
        if not args.dataset:
            print("Error: --dataset is required for local (non-Iceberg) cohorts", file=sys.stderr)
            sys.exit(1)
        if not args.key:
            print("Error: --key is required for local (non-Iceberg) cohorts", file=sys.stderr)
            sys.exit(1)

        batch = table.to_batches()[0] if table.num_rows > 0 else None
        if batch is None:
            print("Error: data file is empty", file=sys.stderr)
            sys.exit(1)

        upsert(
            args.file,
            args.dataset,
            args.key,
            batch,
            output=args.output,
            message=args.message,
            author=args.author,
        )
        out = args.output or args.file
        print(f"Upserted {table.num_rows} rows into '{args.dataset}' → {out}")


def _fetch_merge_datasets(cur, fqn: str, merge_datasets: list) -> list:
    """Fetch additional datasets from Snowflake for merging into a .cohort file."""
    result = []
    for md in merge_datasets:
        print(f"  Merging dataset '{md.dataset}' from {md.table}...")
        cur.execute(f"SELECT * FROM {fqn}.{md.table}")
        tbl = cur.fetch_arrow_all()
        tbl = tbl.rename_columns([c.lower() for c in tbl.column_names])
        print(f"    {tbl.num_rows} rows x {tbl.num_columns} cols")
        result.append((md.dataset, tbl))
    return result


def cmd_load(args: argparse.Namespace) -> None:
    """Load a Python manifest file and build .cohort files from Snowflake."""
    spec_path = Path(args.spec).resolve()
    if not spec_path.exists():
        print(f"Error: spec file not found: {spec_path}", file=sys.stderr)
        sys.exit(1)

    # Import the spec module
    spec = importlib.util.spec_from_file_location("_user_spec", str(spec_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    manifest: CohortManifest | None = getattr(module, "manifest", None)
    if manifest is None:
        print(
            "Error: spec module must define a module-level 'manifest' variable",
            file=sys.stderr,
        )
        sys.exit(1)

    dest = getattr(args, "dest", "cohort")
    codec = getattr(args, "codec", "parquet")
    fqn = f"{manifest.database}.{manifest.schema_name}"

    # Connect to Snowflake
    from komodo.snowflake import get_snowflake_connection

    print("Connecting to Snowflake...")
    conn = get_snowflake_connection()
    cur = conn.cursor()
    cur.execute("SELECT CURRENT_ROLE()")
    role = cur.fetchone()[0]
    cur.execute("SELECT CURRENT_USER()")
    user = cur.fetchone()[0]
    print(f"  Role: {role}")
    print(f"  User: {user}")
    cur.execute(f"USE DATABASE {manifest.database}")
    cur.execute(f"USE SCHEMA {manifest.schema_name}")
    print(f"  Context: {fqn}")
    now = datetime.now(timezone.utc)

    # Connect to Iceberg if needed
    iceberg_conn = None
    if dest == "iceberg":
        if manifest.iceberg is None:
            print(
                "Error: --dest iceberg requires 'iceberg' config in manifest",
                file=sys.stderr,
            )
            sys.exit(1)
        from cohort_sdk import IcebergConnection

        print(f"Connecting to Iceberg (region={manifest.iceberg.region})...")
        iceberg_conn = IcebergConnection(region=manifest.iceberg.region)

    # Build each cohort
    for c in manifest.cohorts:
        # Detect SQL subqueries: table starts with '(' or 'SELECT'
        table_stripped = c.table.strip()
        if table_stripped.startswith("(") or table_stripped.upper().startswith("SELECT"):
            sql = f"SELECT * FROM {fqn}.{table_stripped}" if not table_stripped.startswith("(") else f"SELECT * FROM {table_stripped}"
            print(f"\nQuerying (subquery)...")
        else:
            sql = f"SELECT * FROM {fqn}.{c.table}"
            print(f"\nQuerying {fqn}.{c.table}...")
        cur.execute(sql)
        arrow_table = cur.fetch_arrow_all()
        arrow_table = arrow_table.rename_columns([c.lower() for c in arrow_table.column_names])
        print(f"  {arrow_table.num_rows} rows x {arrow_table.num_columns} cols")

        output = Path(c.output)
        output.parent.mkdir(parents=True, exist_ok=True)

        if dest == "iceberg":
            iceberg_table_name = c.table.lower()
            s3_location = (
                f"{manifest.iceberg.s3_base}/{manifest.iceberg.database}/{iceberg_table_name}"
            )
            print(f"  Writing to Iceberg: {manifest.iceberg.database}.{iceberg_table_name}")
            iceberg_conn.load_to_cohort(
                output,
                c.name,
                arrow_table,
                database=manifest.iceberg.database,
                table_name=iceberg_table_name,
                s3_location=s3_location,
                created_by=f"user:{role}",
                description=c.description,
                selectors=c.selectors,
            )
            print(f"  Written: {output} (metadata-only, {output.stat().st_size:,} bytes)")
        else:
            definition = CohortDefinition(
                name=c.name,
                description=c.description,
                created_at=now,
                created_by=f"user:{role}",
                selectors=c.selectors,
                lineage=Lineage(
                    source_system="Snowflake",
                    source_tables=[c.table if table_stripped.startswith("(") else f"{fqn}.{c.table}"]
                        + [f"{fqn}.{md.table}" for md in c.merge_datasets],
                    extraction_date=now.strftime("%Y-%m-%d"),
                ),
                revision_history=[
                    RevisionEntry(
                        revision=1,
                        timestamp=now,
                        author=f"user:{role}",
                        message=f"Initial extraction: {c.table}",
                        size=arrow_table.num_rows,
                    )
                ],
            )

            if c.increments:
                import pyarrow as pa
                from cohort_sdk import increment

                source_tables = [c.table if table_stripped.startswith("(") else f"{fqn}.{c.table}"]
                revision_history = [
                    RevisionEntry(
                        revision=1,
                        timestamp=now,
                        author=f"user:{role}",
                        message=f"Initial extraction: {c.table}",
                        size=arrow_table.num_rows,
                    )
                ]

                # Fetch all increment tables from Snowflake
                inc_tuples = []
                for i, step in enumerate(c.increments):
                    delete_keys: list[str] = []
                    adds_batch = None

                    if step.deletes_table:
                        print(f"  Fetching deletes: {step.deletes_table}...")
                        cur.execute(f"SELECT * FROM {fqn}.{step.deletes_table}")
                        del_table = cur.fetch_arrow_all()
                        del_table = del_table.rename_columns([col.lower() for col in del_table.column_names])
                        delete_keys = [str(v) for v in del_table.column(step.key_column).to_pylist() if v is not None]
                        print(f"    {len(delete_keys)} keys to delete")
                        source_tables.append(f"{fqn}.{step.deletes_table}")

                    if step.adds_table:
                        print(f"  Fetching adds: {step.adds_table}...")
                        cur.execute(f"SELECT * FROM {fqn}.{step.adds_table}")
                        adds_batch = cur.fetch_arrow_all()
                        adds_batch = adds_batch.rename_columns([col.lower() for col in adds_batch.column_names])
                        print(f"    {adds_batch.num_rows} rows to add")
                        source_tables.append(f"{fqn}.{step.adds_table}")

                    inc_tuples.append((delete_keys, adds_batch))

                # Unify schemas: cast both base and adds to the widest types
                all_tables = [arrow_table] + [ab for _, ab in inc_tuples if ab is not None]
                unified = pa.unify_schemas([t.schema for t in all_tables], promote_options="permissive")
                arrow_table = arrow_table.cast(unified)
                inc_tuples = [
                    (dk, ab.cast(unified) if ab is not None else None)
                    for dk, ab in inc_tuples
                ]

                # Apply increments one at a time (Rust) to track per-step row counts
                print(f"  Applying {len(inc_tuples)} incremental step(s) in-memory...")
                key_col = c.increments[0].key_column
                current_table = arrow_table
                for i, (step, (dk, ab)) in enumerate(zip(c.increments, inc_tuples)):
                    current_table = increment(current_table, key_col, [(dk, ab)])
                    adds_count = ab.num_rows if ab is not None else 0
                    msg = step.message or f"Increment step {i + 1}"
                    msg += f" ({len(dk):,} deleted, {adds_count:,} added)"
                    print(f"    Step {i}: {current_table.num_rows} rows — {msg}")
                    revision_history.append(
                        RevisionEntry(
                            revision=i + 2,
                            timestamp=now,
                            author=f"user:{role}",
                            message=msg,
                            size=current_table.num_rows,
                        )
                    )
                final_table = current_table
                print(f"  Final: {final_table.num_rows} rows")

                definition = CohortDefinition(
                    name=c.name,
                    description=c.description,
                    created_at=now,
                    created_by=f"user:{role}",
                    revision=len(revision_history),
                    selectors=c.selectors,
                    lineage=Lineage(
                        source_system="Snowflake",
                        source_tables=source_tables,
                        extraction_date=now.strftime("%Y-%m-%d"),
                    ),
                    revision_history=revision_history,
                )

                all_datasets = [(c.dataset, final_table)]
                all_datasets += _fetch_merge_datasets(cur, fqn, c.merge_datasets)
                definition.write(output, all_datasets, codec=codec)
                print(f"  Written: {output} ({output.stat().st_size:,} bytes)")
            else:
                all_datasets = [(c.dataset, arrow_table)]
                all_datasets += _fetch_merge_datasets(cur, fqn, c.merge_datasets)
                definition.write(output, all_datasets, codec=codec)
                print(f"  Written: {output} ({output.stat().st_size:,} bytes)")

    if iceberg_conn is not None:
        iceberg_conn.close()
    cur.close()
    conn.close()
    print("\nDone!")


def cmd_discover(args: argparse.Namespace) -> None:
    """Crawl Snowflake inventory and output a JSON discovery file."""
    from komodo.snowflake import get_snowflake_connection
    from cohort_discovery import SnowflakeDiscovery

    print("Connecting to Snowflake...")
    conn = get_snowflake_connection()

    discovery = SnowflakeDiscovery(conn)
    print(f"Discovering {args.database}...")
    inventory = discovery.discover_database(
        database=args.database,
        schema=args.schema,
        detailed=args.detailed,
    )

    conn.close()

    output = args.output or "discovery.json"
    inventory.write(output)
    print(f"\nDiscovery complete:")
    print(f"  Schemas: {len(inventory.schemas)}")
    print(f"  Tables:  {len(inventory.tables)}")
    print(f"  Output:  {output}")


def cmd_sonify(args: argparse.Namespace) -> None:
    """Convert a .cohort file into a 3-layer 8-bit MP3/WAV.

    Layers:
        1. Melody  — slow chirpy pulse wave with vibrato (OoT ocarina feel)
        2. Bass    — low triangle-wave oscillation, changes every 4 melody notes
        3. Sub     — very slow reverb-drenched sine pad, changes every 16 melody notes
    """
    import hashlib
    import math
    import subprocess
    import wave

    import numpy as np

    cohort = CohortDefinition.read(args.file)
    ds_label = cohort.datasets[0]
    table = cohort[ds_label]
    num_rows = table.num_rows

    if num_rows == 0:
        print("Error: cohort has no rows", file=sys.stderr)
        sys.exit(1)

    col_names = table.column_names

    # ── Auto-pick columns: hash 3 distinct cols from the data ─
    # Prefer high-cardinality columns (more unique values = bouncier).
    # Skip cols that are all-null or boolean.
    import pyarrow.types as pat

    def _col_cardinality(name):
        col = table.column(name)
        if pat.is_boolean(col.type):
            return -1
        if col.null_count == num_rows:
            return -1
        try:
            return len(col.value_counts())
        except Exception:
            return 0

    if not (args.melody_col and args.bass_col and args.sub_col):
        ranked = sorted(col_names, key=_col_cardinality, reverse=True)
        # Deduplicate: pick top 3 distinct
        picks = []
        for c in ranked:
            if _col_cardinality(c) > 0 and c not in picks:
                picks.append(c)
            if len(picks) >= 3:
                break
        while len(picks) < 3:
            picks.append(picks[-1] if picks else col_names[0])
    else:
        picks = [None, None, None]

    melody_col = args.melody_col or picks[0]
    bass_col = args.bass_col or picks[1]
    sub_col = args.sub_col or picks[2]

    for label, col in [("melody", melody_col), ("bass", bass_col), ("sub", sub_col)]:
        if col not in col_names:
            print(f"Error: {label} column '{col}' not found (available: {col_names})", file=sys.stderr)
            sys.exit(1)

    # ── Hash the cohort identity into ALL musical parameters ──
    SCALES = {
        "lydian":         [0, 2, 4, 6, 7, 9, 11],   # dreamy, magical
        "dorian":         [0, 2, 3, 5, 7, 9, 10],   # melancholic, groovy
        "phrygian":       [0, 1, 3, 5, 7, 8, 10],   # dark, intense
        "mixolydian":     [0, 2, 4, 5, 7, 9, 10],   # bluesy, laid-back
        "aeolian":        [0, 2, 3, 5, 7, 8, 10],   # sad, serious
        "major":          [0, 2, 4, 5, 7, 9, 11],   # bright, happy
        "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],   # exotic, dramatic
        "pentatonic":     [0, 2, 4, 7, 9],           # simple, video-gamey
    }
    scale_names = list(SCALES.keys())

    # Build cohort identity string
    defn = cohort.definition
    identity = defn.name + "|" + "|".join(
        f"{s.field}{s.operator}{s.values}" for s in defn.selectors
    )
    # Full hash — we'll slice different byte ranges for each parameter
    full_hash = hashlib.sha256(identity.encode()).hexdigest()

    def hash_slice(start, end):
        return int(full_hash[start:end], 16)

    # 1. Scale
    chosen_scale = args.scale if args.scale and args.scale in SCALES else scale_names[hash_slice(0, 8) % len(scale_names)]

    # 2. BPM: range 60–170 (slow ambient → uptempo)
    bpm = args.bpm if args.bpm != 140 else 60 + (hash_slice(8, 16) % 111)

    # 3. Melody note length: 1–4 beats
    mel_beats_options = [1, 2, 3, 4]
    mel_beats = mel_beats_options[hash_slice(16, 20) % len(mel_beats_options)]

    # 4. Bass note multiplier over melody: 2x, 4x, 6x, 8x
    bass_mult_options = [2, 4, 6, 8]
    bass_mult = bass_mult_options[hash_slice(20, 24) % len(bass_mult_options)]

    # 5. Sub note multiplier over bass: 2x, 4x, 8x
    sub_mult_options = [2, 4, 8]
    sub_mult = sub_mult_options[hash_slice(24, 28) % len(sub_mult_options)]

    # 6. Number of layers: 2, 3, or 4
    num_layers_options = [2, 3, 3, 4]  # weighted toward 3
    num_layers = num_layers_options[hash_slice(28, 32) % len(num_layers_options)]

    # 7. Melody waveform
    mel_wave_options = ["pulse", "square", "triangle", "sine"]
    mel_wave_choice = mel_wave_options[hash_slice(32, 36) % len(mel_wave_options)]

    # 8. Bass waveform (always lower-energy than melody)
    bass_wave_options = ["triangle", "sine", "square"]
    bass_wave_choice = bass_wave_options[hash_slice(36, 40) % len(bass_wave_options)]

    # 9. Melody octave offset: 0 or 1 (shifts whole melody up an octave)
    mel_octave_shift = hash_slice(40, 44) % 2

    # 10. Vibrato rate: 3–7 Hz
    vibrato_hz = 3.0 + (hash_slice(44, 48) % 50) / 10.0

    # 11. Echo character: short/long
    echo_style = "long" if hash_slice(48, 52) % 2 == 0 else "short"

    # 12. Reverb intensity: light/heavy
    reverb_style = "heavy" if hash_slice(52, 56) % 2 == 0 else "light"

    # 13. LFO oscillation period for melody: 8–40
    mel_lfo_period = 8 + (hash_slice(56, 60) % 33)

    intervals = SCALES[chosen_scale]
    C4 = 261.63

    # Build frequency tables
    mel_base = C4 * (2 ** mel_octave_shift)
    melody_freqs = [mel_base * (2 ** (((o * 12) + s) / 12)) for o in range(2) for s in intervals]
    bass_freqs = [C4 / 4 * (2 ** (((o * 12) + s) / 12)) for o in range(2) for s in intervals]
    sub_freqs = [C4 / 8 * (2 ** (s / 12)) for s in intervals]

    # 4th layer (arp/pad) if num_layers == 4: sits between melody and bass
    arp_freqs = [C4 / 2 * (2 ** (((o * 12) + s) / 12)) for o in range(2) for s in intervals]

    print(f"  Identity: {identity[:60]}...")
    print(f"  Scale:    {chosen_scale}  |  BPM: {bpm}  |  Layers: {num_layers}")
    print(f"  Melody:   {mel_wave_choice}, {mel_beats} beat(s)/note, octave +{mel_octave_shift}")
    print(f"  Bass:     {bass_wave_choice}, every {mel_beats * bass_mult} beats")
    print(f"  Sub:      sine pad, every {mel_beats * bass_mult * sub_mult} beats, reverb={reverb_style}")
    if num_layers == 4:
        print(f"  Arp:      triangle, every {mel_beats} beats (mid-register)")
    print(f"  Vibrato:  {vibrato_hz:.1f} Hz  |  Echo: {echo_style}  |  LFO period: {mel_lfo_period}")

    # ── Oscillation-enforced hash bucketing ───────────────────
    def oscillating_hash(values, num_notes, period, amplitude):
        indices = []
        for i, v in enumerate(values):
            h = int(hashlib.sha256(str(v).encode()).hexdigest()[:8], 16)
            bucket = h % num_notes
            lfo = amplitude * math.sin(2 * math.pi * i / period)
            indices.append(int(bucket + lfo) % num_notes)
        return indices

    # ── Timing ────────────────────────────────────────────────
    sample_rate = 44100
    max_dur = 180.0
    beat_dur = 60.0 / bpm

    bass_beats = mel_beats * bass_mult
    sub_beats = bass_beats * sub_mult

    mel_note_dur = mel_beats * beat_dur
    bass_note_dur = bass_beats * beat_dur
    sub_note_dur = sub_beats * beat_dur

    num_mel_notes = max(min(int(max_dur / mel_note_dur), num_rows), 1)
    total_dur = num_mel_notes * mel_note_dur
    total_samples = int(total_dur * sample_rate)
    num_bass_notes = max(int(total_dur / bass_note_dur), 1)
    num_sub_notes = max(int(total_dur / sub_note_dur), 1)

    def sample_rows(col_values, count):
        step = max(len(col_values) // count, 1)
        return [col_values[i * step % len(col_values)] for i in range(count)]

    raw_mel = table.column(melody_col).to_pylist()
    raw_bass = table.column(bass_col).to_pylist()
    raw_sub = table.column(sub_col).to_pylist()

    mel_indices = oscillating_hash(
        sample_rows(raw_mel, num_mel_notes), len(melody_freqs),
        period=mel_lfo_period, amplitude=4,
    )
    bass_indices = oscillating_hash(
        sample_rows(raw_bass, num_bass_notes), len(bass_freqs),
        period=max(mel_lfo_period // 2, 4), amplitude=3,
    )
    sub_indices = oscillating_hash(
        sample_rows(raw_sub, num_sub_notes), len(sub_freqs),
        period=max(mel_lfo_period // 4, 3), amplitude=2,
    )
    if num_layers == 4:
        arp_col = picks[min(2, len(picks) - 1)] if len(picks) > 2 else melody_col
        raw_arp = table.column(arp_col).to_pylist()
        arp_indices = oscillating_hash(
            sample_rows(raw_arp, num_mel_notes), len(arp_freqs),
            period=mel_lfo_period + 7, amplitude=3,
        )

    # ── Envelopes ─────────────────────────────────────────────
    def make_envelope(n, attack_s, release_s):
        env = np.ones(n)
        a = min(int(attack_s * sample_rate), n // 3)
        r = min(int(release_s * sample_rate), n // 3)
        if a > 0:
            env[:a] = np.linspace(0, 1, a)
        if r > 0:
            env[-r:] = np.linspace(1, 0, r)
        return env

    mel_nsamp = int(mel_note_dur * sample_rate)
    bass_nsamp = int(bass_note_dur * sample_rate)
    sub_nsamp = int(sub_note_dur * sample_rate)

    mel_env = make_envelope(mel_nsamp, 0.04, 0.20)
    bass_env = make_envelope(bass_nsamp, 0.03, 0.12)
    sub_env = make_envelope(sub_nsamp, 0.25, 0.4)

    # ── Waveforms ─────────────────────────────────────────────
    def gen_pulse(freq, n, duty=0.25):
        t = np.arange(n) / sample_rate
        vib = 1.0 + 0.005 * np.sin(2 * np.pi * vibrato_hz * t)
        phase = np.cumsum(freq * vib / sample_rate)
        return ((phase % 1.0) < duty).astype(np.float64) * 2.0 - 1.0

    def gen_square(freq, n):
        return gen_pulse(freq, n, duty=0.5)

    def gen_triangle(freq, n):
        t = np.arange(n) / sample_rate
        phase = (freq * t) % 1.0
        return 2.0 * np.abs(2.0 * phase - 1.0) - 1.0

    def gen_sine(freq, n):
        t = np.arange(n) / sample_rate
        return np.sin(2 * np.pi * freq * t)

    WAVE_FNS = {"pulse": gen_pulse, "square": gen_square, "triangle": gen_triangle, "sine": gen_sine}
    mel_wave_fn = WAVE_FNS[mel_wave_choice]
    bass_wave_fn = WAVE_FNS[bass_wave_choice]

    # ── Render layers ─────────────────────────────────────────
    melody_audio = np.zeros(total_samples)
    bass_audio = np.zeros(total_samples)
    sub_audio = np.zeros(total_samples)
    arp_audio = np.zeros(total_samples)

    for i in range(num_mel_notes):
        s = i * mel_nsamp
        e = min(s + mel_nsamp, total_samples)
        n = e - s
        if n > 0:
            melody_audio[s:e] = mel_wave_fn(melody_freqs[mel_indices[i]], n)[:n] * mel_env[:n] * 0.25

    for i in range(num_bass_notes):
        s = i * bass_nsamp
        e = min(s + bass_nsamp, total_samples)
        n = e - s
        if n > 0:
            bass_audio[s:e] = bass_wave_fn(bass_freqs[bass_indices[i]], n)[:n] * bass_env[:n] * 0.30

    for i in range(num_sub_notes):
        s = i * sub_nsamp
        e = min(s + sub_nsamp, total_samples)
        n = e - s
        if n > 0:
            sub_audio[s:e] = gen_sine(sub_freqs[sub_indices[i]], n)[:n] * sub_env[:n] * 0.25

    if num_layers >= 4:
        arp_env = make_envelope(mel_nsamp, 0.01, 0.10)
        for i in range(num_mel_notes):
            s = i * mel_nsamp
            e = min(s + mel_nsamp, total_samples)
            n = e - s
            if n > 0:
                arp_audio[s:e] = gen_triangle(arp_freqs[arp_indices[i]], n)[:n] * arp_env[:n] * 0.18

    # ── Reverb on sub ─────────────────────────────────────────
    if reverb_style == "heavy":
        tap_delays = [0.25, 0.55, 0.95, 1.4, 2.0]
        tap_gains = [0.40, 0.28, 0.18, 0.10, 0.05]
    else:
        tap_delays = [0.15, 0.35]
        tap_gains = [0.25, 0.10]

    sub_reverbed = sub_audio.copy()
    for delay_s, gain in zip(tap_delays, tap_gains):
        d = int(delay_s * sample_rate)
        if d < total_samples:
            sub_reverbed[d:] += sub_audio[:-d] * gain

    # ── Echo on melody ────────────────────────────────────────
    if echo_style == "long":
        echo_taps = [(0.22, 0.28), (0.45, 0.15), (0.7, 0.08)]
    else:
        echo_taps = [(0.1, 0.20), (0.2, 0.08)]

    mel_echo = melody_audio.copy()
    for delay_s, gain in echo_taps:
        d = int(delay_s * sample_rate)
        if d < total_samples:
            mel_echo[d:] += melody_audio[:-d] * gain

    # ── Mix ───────────────────────────────────────────────────
    mixed = mel_echo + bass_audio + sub_reverbed
    if num_layers >= 4:
        mixed = mixed + arp_audio

    # Normalize
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = mixed / peak * 0.95
    pcm = (mixed * 32767).astype(np.int16)

    # ── Write output ──────────────────────────────────────────
    output = args.output or args.file.replace(".cohort", ".mp3")

    if output.endswith(".mp3"):
        wav_tmp = output.rsplit(".", 1)[0] + "_tmp.wav"
    else:
        wav_tmp = output

    with wave.open(wav_tmp, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    if output.endswith(".mp3"):
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", wav_tmp, "-b:a", "192k", output],
                check=True, capture_output=True,
            )
            Path(wav_tmp).unlink()
        except FileNotFoundError:
            print("ffmpeg not found — keeping WAV instead")
            output = wav_tmp

    print(f"\nWrote {output}")
    print(f"  {num_rows} rows → {total_dur:.1f}s")


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate Python manifest files from a YAML split config."""
    from cohort_generator import CohortGenerator

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output or "manifests"

    generator = CohortGenerator(config_path)
    generated = generator.generate(output_dir)

    print(f"Generated {len(generated)} manifest(s):")
    for p in generated:
        print(f"  {p}")
    print(f"\nTo load a manifest:\n  uv run python cohort.py load {generated[0] if generated else '<manifest.py>'}")


# ── Lakehouse commands ────────────────────────────────────────────


def _get_lakehouse_stores(args: argparse.Namespace):
    """Initialize lakehouse stores from args."""
    from lakehouse import VectorStore, BitmapStore, PolicyStore, AuditLogger, QueryBroker

    vector_path = getattr(args, "vector_path", "./indexes/vectors")
    bitmap_path = getattr(args, "bitmap_path", "./indexes/bitmaps")
    policy_path = getattr(args, "policy_path", "./policies")
    audit_path = getattr(args, "audit_path", "./audit_logs")

    return {
        "vector": VectorStore(vector_path),
        "bitmap": BitmapStore(bitmap_path),
        "policy": PolicyStore(policy_path),
        "audit": AuditLogger(audit_path),
    }


def cmd_search(args: argparse.Namespace) -> None:
    """Semantic search for patients."""
    import time
    stores = _get_lakehouse_stores(args)
    vector_store = stores["vector"]

    t0 = time.perf_counter()
    try:
        results = vector_store.search_by_text(
            universe_id=args.universe,
            query_text=args.query,
            top_k=args.top_k,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"Search: \"{args.query}\"")
        print(f"Universe: {args.universe}")
        print(f"Found {len(results)} results in {elapsed:.2f}ms\n")

        if args.json_output:
            import json
            data = [{"patient_id": r.patient_id, "score": round(r.score, 4)} for r in results]
            print(json.dumps(data, indent=2))
        else:
            print(f"{'Patient ID':<15} {'Score':<10}")
            print("-" * 25)
            for r in results:
                print(f"{r.patient_id:<15} {r.score:.4f}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_similar(args: argparse.Namespace) -> None:
    """Find patients similar to a given patient."""
    import time
    stores = _get_lakehouse_stores(args)
    vector_store = stores["vector"]

    t0 = time.perf_counter()
    try:
        results = vector_store.search_by_patient(
            universe_id=args.universe,
            patient_id=args.patient_id,
            top_k=args.top_k,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"Similar to patient: {args.patient_id}")
        print(f"Universe: {args.universe}")
        print(f"Found {len(results)} similar patients in {elapsed:.2f}ms\n")

        if args.json_output:
            import json
            data = [{"patient_id": r.patient_id, "score": round(r.score, 4)} for r in results]
            print(json.dumps(data, indent=2))
        else:
            print(f"{'Patient ID':<15} {'Similarity':<10}")
            print("-" * 25)
            for r in results:
                print(f"{r.patient_id:<15} {r.score:.4f}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_index(args: argparse.Namespace) -> None:
    """Build vector and/or bitmap indexes from a .cohort file."""
    from lakehouse import build_index_from_cohort, extract_membership_from_cohort

    stores = _get_lakehouse_stores(args)
    cohort_path = str(Path(args.file).resolve())
    universe = args.universe or Path(args.file).stem.replace("-", "_").replace(".", "_")

    print(f"Indexing: {args.file}")
    print(f"Universe: {universe}\n")

    if args.vector:
        print("Building vector index...")
        try:
            stats = build_index_from_cohort(stores["vector"], universe, cohort_path)
            print(f"  Indexed {stats.num_vectors} patients")
            print(f"  Dimension: {stats.dimension}")
            print(f"  Elapsed: {stats.elapsed_ms:.2f}ms\n")
        except Exception as e:
            print(f"  Error: {e}\n", file=sys.stderr)

    if args.bitmap:
        print("Building bitmap index...")
        try:
            meta = extract_membership_from_cohort(
                stores["bitmap"], cohort_path, universe, "v1"
            )
            print(f"  Indexed {meta.cardinality} patients")
            print(f"  Version: {meta.version}\n")
        except Exception as e:
            print(f"  Error: {e}\n", file=sys.stderr)

    print("Done!")


def cmd_cohorts(args: argparse.Namespace) -> None:
    """List or operate on bitmap cohorts."""
    import time
    stores = _get_lakehouse_stores(args)
    bitmap_store = stores["bitmap"]

    if args.list_cohorts:
        cohort_ids = bitmap_store.list_cohorts()
        if not cohort_ids:
            print("No cohorts found.")
            return

        print(f"{'Cohort ID':<25} {'Versions':<15} {'Latest Patients':<15}")
        print("-" * 55)
        for cid in cohort_ids:
            versions = bitmap_store.list_versions(cid)
            latest = versions[-1] if versions else "none"
            try:
                bitmap = bitmap_store.read_membership(cid, latest)
                count = len(bitmap)
            except Exception:
                count = "?"
            print(f"{cid:<25} {','.join(versions):<15} {count:<15}")

    elif args.intersect_cohorts:
        if len(args.intersect_cohorts) < 2:
            print("Error: need at least 2 cohorts to intersect", file=sys.stderr)
            sys.exit(1)

        t0 = time.perf_counter()
        result = bitmap_store.intersect(args.intersect_cohorts)
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"Intersect: {' ∩ '.join(args.intersect_cohorts)}")
        print(f"Result: {result.result_cardinality} patients ({elapsed:.2f}ms)")

    elif args.union_cohorts:
        if len(args.union_cohorts) < 2:
            print("Error: need at least 2 cohorts to union", file=sys.stderr)
            sys.exit(1)

        t0 = time.perf_counter()
        result = bitmap_store.union(args.union_cohorts)
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"Union: {' ∪ '.join(args.union_cohorts)}")
        print(f"Result: {result.result_cardinality} patients ({elapsed:.2f}ms)")

    elif args.diff_cohorts:
        if len(args.diff_cohorts) != 2:
            print("Error: need exactly 2 cohorts for difference", file=sys.stderr)
            sys.exit(1)

        t0 = time.perf_counter()
        result = bitmap_store.difference(args.diff_cohorts[0], args.diff_cohorts[1])
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"Difference: {args.diff_cohorts[0]} \\ {args.diff_cohorts[1]}")
        print(f"Result: {result.result_cardinality} patients ({elapsed:.2f}ms)")

    else:
        # Default: list cohorts
        cmd_cohorts(argparse.Namespace(
            list_cohorts=True, intersect_cohorts=None, union_cohorts=None, diff_cohorts=None,
            **{k: v for k, v in vars(args).items() if k not in ['list_cohorts', 'intersect_cohorts', 'union_cohorts', 'diff_cohorts']}
        ))


def cmd_policy(args: argparse.Namespace) -> None:
    """Manage ABAC policies."""
    from lakehouse import Policy
    stores = _get_lakehouse_stores(args)
    policy_store = stores["policy"]

    if args.list_policies:
        policies = policy_store.list_policies(enabled_only=False)
        if not policies:
            print("No policies defined.")
            return

        print(f"{'Policy ID':<20} {'Name':<25} {'Type':<15} {'Enabled':<8}")
        print("-" * 68)
        for p in policies:
            enabled = "Yes" if p.enabled else "No"
            print(f"{p.policy_id:<20} {p.name:<25} {p.filter_type:<15} {enabled:<8}")

    elif args.create_policy:
        policy = Policy(
            policy_id=args.create_policy,
            name=args.policy_name or args.create_policy,
            filter_type=args.policy_type,
            predicate=args.predicate,
            max_rows=args.max_rows,
            enabled=True,
        )
        policy_store.create_policy(policy)
        print(f"Created policy: {policy.policy_id}")
        print(f"  Type: {policy.filter_type}")
        if policy.predicate:
            print(f"  Predicate: {policy.predicate}")
        if policy.max_rows:
            print(f"  Max rows: {policy.max_rows}")

    elif args.delete_policy:
        policy_store.delete_policy(args.delete_policy)
        print(f"Deleted policy: {args.delete_policy}")

    else:
        # Default: list policies
        cmd_policy(argparse.Namespace(
            list_policies=True, create_policy=None, delete_policy=None,
            **{k: v for k, v in vars(args).items() if k not in ['list_policies', 'create_policy', 'delete_policy']}
        ))


def cmd_audit(args: argparse.Namespace) -> None:
    """View audit logs."""
    stores = _get_lakehouse_stores(args)
    audit_logger = stores["audit"]

    if args.stats:
        stats = audit_logger.get_stats()
        print("Audit Statistics:")
        print(f"  Total queries:    {stats['total_queries']}")
        print(f"  Successful:       {stats['successful_queries']}")
        print(f"  Failed:           {stats['failed_queries']}")
        print(f"  Success rate:     {stats['success_rate']}%")
        print(f"  Avg latency:      {stats['avg_latency_ms']}ms")
        print(f"  Total rows:       {stats['total_rows_returned']}")

        if stats.get('queries_by_type'):
            print("\n  By type:")
            for qtype, count in stats['queries_by_type'].items():
                print(f"    {qtype}: {count}")
    else:
        entries = audit_logger.query_logs(
            limit=args.limit,
            user_id=args.user if args.user else None,
        )

        if not entries:
            print("No audit entries found.")
            return

        print(f"{'Timestamp':<20} {'User':<15} {'Type':<12} {'Rows':<8} {'Latency':<10} {'Status':<8}")
        print("-" * 73)
        for e in entries:
            status = "OK" if e.success else "FAIL"
            print(f"{e.timestamp[:19]:<20} {e.user_id:<15} {e.query_type:<12} {e.rows_returned:<8} {e.elapsed_ms:<10.2f} {status:<8}")


# ── CLI entry point ───────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cohort",
        description="CLI for .cohort files — inspect, merge, intersect, and load.",
    )
    sub = parser.add_subparsers(dest="command")

    # inspect
    p_inspect = sub.add_parser("inspect", help="Print metadata for a .cohort file")
    p_inspect.add_argument("file", help="Path to the .cohort file")
    p_inspect.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output machine-readable JSON instead of human-readable text",
    )

    # merge (union)
    p_merge = sub.add_parser("merge", help="Union (concatenate rows) of multiple .cohort files")
    p_merge.add_argument("files", nargs="+", help="Input .cohort files (at least two)")
    p_merge.add_argument("-o", "--output", required=True, dest="out", help="Output .cohort path")
    p_merge.add_argument("--name", default=None, help="Override the cohort name in the output")

    # intersect
    p_intersect = sub.add_parser("intersect", help="Intersect rows across multiple .cohort files")
    p_intersect.add_argument("files", nargs="+", help="Input .cohort files (at least two)")
    p_intersect.add_argument("-o", "--output", required=True, dest="out", help="Output .cohort path")
    p_intersect.add_argument("--on", required=True, help="Column name to intersect on")
    p_intersect.add_argument("--name", default=None, help="Override the cohort name in the output")

    # annotate
    p_annotate = sub.add_parser("annotate", help="Modify name, description, or selectors")
    p_annotate.add_argument("file", help="Path to the .cohort file")
    p_annotate.add_argument("-o", "--output", default=None, help="Output path (default: overwrite in place)")
    p_annotate.add_argument("--name", default=None, help="Set the cohort name")
    p_annotate.add_argument("--description", default=None, help="Set the cohort description")
    p_annotate.add_argument("--remove-selector", dest="remove_selectors", action="append", default=[], help="Remove a selector by ID (repeatable)")
    p_annotate.add_argument("-m", "--message", default="Annotate", help="Revision message")
    p_annotate.add_argument("--author", default="cohort-cli", help="Author")

    # diff
    p_diff = sub.add_parser("diff", help="Compare two .cohort files or two revisions")
    p_diff.add_argument("file", help="First .cohort file (or sole file with --rev)")
    p_diff.add_argument("file2", nargs="?", default=None, help="Second .cohort file")
    p_diff.add_argument("--rev", nargs=2, type=int, metavar=("FROM", "TO"), help="Compare two revisions within the same file")
    p_diff.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")

    # append
    p_append = sub.add_parser("append", help="Append rows to a dataset from a data file")
    p_append.add_argument("file", help="Path to the .cohort file")
    p_append.add_argument("-o", "--output", default=None, help="Output path (default: overwrite in place)")
    p_append.add_argument("--dataset", default=None, help="Dataset label to append to (required for local cohorts)")
    p_append.add_argument("--data", required=True, help="Path to a Parquet, CSV, or .cohort file")
    p_append.add_argument("-m", "--message", default="Append data", help="Revision message")
    p_append.add_argument("--author", default="cohort-cli", help="Author")

    # delete
    p_delete = sub.add_parser("delete", help="Delete rows, a dataset, or selectors")
    p_delete.add_argument("file", help="Path to the .cohort file")
    p_delete.add_argument("-o", "--output", default=None, help="Output path (default: overwrite in place)")
    p_delete.add_argument("--key", default=None, help="Key column name (for deleting rows)")
    p_delete.add_argument("--values", nargs="+", default=[], help="Key values to delete")
    p_delete.add_argument("--dataset", default=None, help="Delete an entire dataset by label")
    p_delete.add_argument("--selector", dest="selectors", action="append", default=[], help="Delete a selector by ID (repeatable)")
    p_delete.add_argument("-m", "--message", default="Delete", help="Revision message")
    p_delete.add_argument("--author", default="cohort-cli", help="Author")

    # upsert
    p_upsert = sub.add_parser("upsert", help="Update-or-insert rows by key column")
    p_upsert.add_argument("file", help="Path to the .cohort file")
    p_upsert.add_argument("-o", "--output", default=None, help="Output path (default: overwrite in place)")
    p_upsert.add_argument("--dataset", default=None, help="Dataset label (required for local cohorts)")
    p_upsert.add_argument("--key", default=None, help="Key column for matching (defaults to primary_keys for Iceberg targets)")
    p_upsert.add_argument("--data", required=True, help="Path to a Parquet, CSV, or .cohort file")
    p_upsert.add_argument("-m", "--message", default="Upsert data", help="Revision message")
    p_upsert.add_argument("--author", default="cohort-cli", help="Author")

    # load
    p_load = sub.add_parser("load", help="Load cohorts from a Python manifest (requires Snowflake)")
    p_load.add_argument("spec", help="Path to a Python file defining 'manifest'")
    p_load.add_argument(
        "--dest",
        choices=["cohort", "iceberg"],
        default="cohort",
        help="Destination: 'cohort' (embedded data, default) or 'iceberg' (Iceberg table + metadata-only .cohort)",
    )
    p_load.add_argument(
        "--codec",
        choices=["parquet", "arrow_ipc", "feather", "vortex"],
        default="parquet",
        help="Data block encoding: 'parquet' (default), 'arrow_ipc', 'feather', or 'vortex'",
    )

    # discover
    p_discover = sub.add_parser("discover", help="Crawl Snowflake inventory for a database")
    p_discover.add_argument("database", help="Snowflake database name to crawl")
    p_discover.add_argument("--schema", default=None, help="Limit to a specific schema")
    p_discover.add_argument(
        "--detailed", action="store_true",
        help="Run sampling queries for column stats (min/max, distinct values)",
    )
    p_discover.add_argument("--output", default=None, help="Output JSON path (default: discovery.json)")

    # sonify
    p_sonify = sub.add_parser("sonify", help="Convert a .cohort file into an 8-bit MP3/WAV")
    p_sonify.add_argument("file", help="Path to the .cohort file")
    p_sonify.add_argument("-o", "--output", default=None, help="Output path (.mp3 or .wav, default: <file>.mp3)")
    p_sonify.add_argument("--melody-col", default=None, help="Column for melody (default: first column)")
    p_sonify.add_argument("--bass-col", default=None, help="Column for bass line (default: second column)")
    p_sonify.add_argument("--sub-col", default=None, help="Column for reverb sub bass (default: third column)")
    p_sonify.add_argument("--scale", default=None, help="Force a scale (lydian, dorian, phrygian, mixolydian, aeolian, major, harmonic_minor, pentatonic). Default: auto from cohort definition")
    p_sonify.add_argument("--bpm", type=int, default=140, help="Tempo in BPM (default: 140)")

    # generate
    p_generate = sub.add_parser("generate", help="Generate manifests from a YAML split config")
    p_generate.add_argument("config", help="Path to a YAML split config file")
    p_generate.add_argument("--output", default=None, help="Output directory for manifests (default: manifests/)")

    # ── Lakehouse commands ────────────────────────────────────

    # search
    p_search = sub.add_parser("search", help="Semantic search for patients")
    p_search.add_argument("universe", help="Universe ID (index name)")
    p_search.add_argument("query", help="Natural language query")
    p_search.add_argument("--top-k", type=int, default=20, help="Number of results")
    p_search.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
    p_search.add_argument("--vector-path", default="./indexes/vectors", help="Vector store path")

    # similar
    p_similar = sub.add_parser("similar", help="Find similar patients")
    p_similar.add_argument("universe", help="Universe ID (index name)")
    p_similar.add_argument("patient_id", type=int, help="Reference patient ID")
    p_similar.add_argument("--top-k", type=int, default=10, help="Number of results")
    p_similar.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
    p_similar.add_argument("--vector-path", default="./indexes/vectors", help="Vector store path")

    # index
    p_index = sub.add_parser("index", help="Build vector/bitmap indexes from a cohort file")
    p_index.add_argument("file", help="Path to .cohort file")
    p_index.add_argument("--universe", default=None, help="Universe ID (default: filename)")
    p_index.add_argument("--vector", action="store_true", default=True, help="Build vector index")
    p_index.add_argument("--no-vector", action="store_false", dest="vector", help="Skip vector index")
    p_index.add_argument("--bitmap", action="store_true", default=True, help="Build bitmap index")
    p_index.add_argument("--no-bitmap", action="store_false", dest="bitmap", help="Skip bitmap index")
    p_index.add_argument("--vector-path", default="./indexes/vectors", help="Vector store path")
    p_index.add_argument("--bitmap-path", default="./indexes/bitmaps", help="Bitmap store path")

    # cohorts
    p_cohorts = sub.add_parser("cohorts", help="Bitmap cohort operations")
    p_cohorts.add_argument("--list", action="store_true", dest="list_cohorts", help="List all cohorts")
    p_cohorts.add_argument("--intersect", nargs="+", dest="intersect_cohorts", help="Intersect cohorts")
    p_cohorts.add_argument("--union", nargs="+", dest="union_cohorts", help="Union cohorts")
    p_cohorts.add_argument("--diff", nargs=2, dest="diff_cohorts", help="Difference of two cohorts")
    p_cohorts.add_argument("--bitmap-path", default="./indexes/bitmaps", help="Bitmap store path")

    # policy
    p_policy = sub.add_parser("policy", help="Manage ABAC policies")
    p_policy.add_argument("--list", action="store_true", dest="list_policies", help="List policies")
    p_policy.add_argument("--create", dest="create_policy", help="Create a new policy with this ID")
    p_policy.add_argument("--name", dest="policy_name", help="Policy name")
    p_policy.add_argument("--type", dest="policy_type", default="predicate", choices=["predicate", "row_limit", "bitmap", "column_mask"])
    p_policy.add_argument("--predicate", help="SQL predicate for predicate policies")
    p_policy.add_argument("--max-rows", type=int, help="Max rows for row_limit policies")
    p_policy.add_argument("--delete", dest="delete_policy", help="Delete a policy by ID")
    p_policy.add_argument("--policy-path", default="./policies", help="Policy store path")

    # audit
    p_audit = sub.add_parser("audit", help="View audit logs")
    p_audit.add_argument("--limit", type=int, default=50, help="Max entries to show")
    p_audit.add_argument("--user", default=None, help="Filter by user ID")
    p_audit.add_argument("--stats", action="store_true", help="Show statistics instead of entries")
    p_audit.add_argument("--audit-path", default="./audit_logs", help="Audit log path")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "merge":
        cmd_merge(args)
    elif args.command == "intersect":
        cmd_intersect(args)
    elif args.command == "annotate":
        cmd_annotate(args)
    elif args.command == "diff":
        cmd_diff(args)
    elif args.command == "append":
        cmd_append(args)
    elif args.command == "delete":
        cmd_delete(args)
    elif args.command == "upsert":
        cmd_upsert(args)
    elif args.command == "load":
        cmd_load(args)
    elif args.command == "discover":
        cmd_discover(args)
    elif args.command == "sonify":
        cmd_sonify(args)
    elif args.command == "generate":
        cmd_generate(args)
    # Lakehouse commands
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "similar":
        cmd_similar(args)
    elif args.command == "index":
        cmd_index(args)
    elif args.command == "cohorts":
        cmd_cohorts(args)
    elif args.command == "policy":
        cmd_policy(args)
    elif args.command == "audit":
        cmd_audit(args)


if __name__ == "__main__":
    main()

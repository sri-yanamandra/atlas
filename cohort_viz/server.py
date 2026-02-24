"""Cohort Visualizer — FastAPI server for .cohort files.

Provides a web UI and REST API for inspecting, previewing, and operating
on .cohort files using the cohort-sdk.

Usage:
    uv run python -m cohort_viz.server [directory]
    # or
    make viz
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from cohort_sdk import (
    CohortDefinition,
    CohortCatalog,
    inspect as cohort_inspect,
    union as cohort_union,
    intersect as cohort_intersect,
    read as cohort_read,
    partition as cohort_partition,
)

# Directory to scan for .cohort files (configurable via CLI arg)
COHORT_DIR = Path(".")

app = FastAPI(title="Cohort Visualizer")


# ── Models ───────────────────────────────────────────────────


class UnionRequest(BaseModel):
    files: list[str]
    output: str
    name: str | None = None


class IntersectRequest(BaseModel):
    files: list[str]
    output: str
    on: str
    name: str | None = None


class OverlapRequest(BaseModel):
    files: list[str]
    on: str


class QueryRequest(BaseModel):
    sql: str
    limit: int = 500
    use_partitions: bool = False
    engine: str = "datafusion"


class PartitionRequest(BaseModel):
    file: str
    num_partitions: int | None = None
    target_rows: int | None = None


class SimilaritySearchRequest(BaseModel):
    cohort_file: str
    patient_id: int | None = None
    query_text: str | None = None
    filters: dict[str, str] | None = None
    limit: int = 10


class LakehouseSearchRequest(BaseModel):
    universe_id: str
    query_text: str
    top_k: int = 20


class LakehouseSimilarRequest(BaseModel):
    universe_id: str
    patient_id: int
    top_k: int = 10


class LakehouseIndexRequest(BaseModel):
    cohort_file: str
    universe_id: str | None = None
    build_vector: bool = True
    build_bitmap: bool = True


class LakehouseBitmapOpRequest(BaseModel):
    operation: str  # intersect, union, difference
    cohort_ids: list[str]


class LakehousePolicyRequest(BaseModel):
    policy_id: str
    name: str
    filter_type: str = "predicate"
    predicate: str | None = None
    max_rows: int | None = None


# ── Catalog (lazy init) ──────────────────────────────────────

_catalog = None
_catalog_partitioned = None


def _get_catalog(partitioned: bool = False):
    global _catalog, _catalog_partitioned
    if CohortCatalog is None:
        raise HTTPException(
            501, "CohortCatalog not available — rebuild cohort-sdk with maturin"
        )
    if partitioned:
        parts_dir = COHORT_DIR / "partitions"
        if not parts_dir.exists() or not any(parts_dir.glob("*.cohort")):
            raise HTTPException(404, "No partitions found — run partition first")
        if _catalog_partitioned is None:
            CohortCatalog.init(str(parts_dir))
            _catalog_partitioned = CohortCatalog(str(parts_dir))
        return _catalog_partitioned
    if _catalog is None:
        CohortCatalog.init(str(COHORT_DIR))
        _catalog = CohortCatalog(str(COHORT_DIR))
    return _catalog



_duckdb_conn = None


def _get_duckdb_conn():
    """Get or create a DuckDB connection with catalog tables registered."""
    global _duckdb_conn
    if _duckdb_conn is not None:
        return _duckdb_conn
    import duckdb

    cat = _get_catalog()
    conn = duckdb.connect(":memory:")
    for table_name in cat.tables():
        try:
            pa_table = cat.query(f'SELECT * FROM "{table_name}"')
            conn.register(table_name, pa_table)
        except Exception:
            pass
    _duckdb_conn = conn
    return conn


# ── API routes ───────────────────────────────────────────────


@app.get("/")
async def index():
    """Serve the single-page visualizer UI."""
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text())


@app.get("/api/files")
async def list_files():
    """List all .cohort files in the directory with catalog metadata."""
    import json

    files = sorted(COHORT_DIR.glob("*.cohort"))

    # Load catalog metadata keyed by file_path
    catalog_map: dict[str, dict] = {}
    cat_path = COHORT_DIR / ".kohort-catalog.json"
    if cat_path.exists():
        try:
            data = json.loads(cat_path.read_text())
            for entry in data.get("files", []):
                catalog_map[entry["file_path"]] = entry
        except Exception:
            pass

    results = []
    for f in files:
        rel = f.relative_to(COHORT_DIR)
        item: dict = {
            "name": str(rel),
            "path": str(f),
            "size_bytes": f.stat().st_size,
        }
        cat = catalog_map.get(str(rel))
        if cat:
            item["cohort_name"] = cat.get("cohort_name", "")
            item["selectors"] = [
                {"field": s.get("field", ""), "operator": s.get("operator", ""),
                 "values": s.get("values", [])}
                for s in cat.get("selectors", [])
            ]
            total_rows = sum(m.get("num_rows", 0) for m in cat.get("datasets", []))
            item["num_rows"] = total_rows
            item["datasets"] = [m.get("label", "") for m in cat.get("datasets", [])]
            columns: list[str] = []
            for m in cat.get("datasets", []):
                for col in m.get("schema", []):
                    cname = col.get("name", "")
                    if cname and cname not in columns:
                        columns.append(cname)
            item["columns"] = columns
        results.append(item)
    return {"files": results, "directory": str(COHORT_DIR.resolve())}


@app.get("/api/inspect/{filename:path}")
async def inspect_file(filename: str):
    """Inspect a .cohort file — returns full metadata without reading data."""
    path = COHORT_DIR / filename
    if not path.exists() or not path.suffix == ".cohort":
        raise HTTPException(404, f"File not found: {filename}")
    info = cohort_inspect(str(path))
    return info.model_dump()


@app.get("/api/definition/{filename:path}")
async def definition_file(filename: str):
    """Return the full CohortDefinition including revision deltas."""
    path = COHORT_DIR / filename
    if not path.exists() or not path.suffix == ".cohort":
        raise HTTPException(404, f"File not found: {filename}")
    cohort = cohort_read(str(path))
    return cohort.definition.model_dump(mode="json")


@app.get("/api/data/{filename:path}/{dataset}")
async def read_dataset(
    filename: str,
    dataset: str,
    limit: int = Query(default=100, le=10000),
):
    """Read rows from a specific dataset, returned as columnar JSON."""
    path = COHORT_DIR / filename
    if not path.exists():
        raise HTTPException(404, f"File not found: {filename}")
    cohort = cohort_read(str(path))
    if dataset not in cohort.datasets:
        raise HTTPException(404, f"Dataset not found: {dataset}")
    table = cohort[dataset]
    # Slice to limit
    if table.num_rows > limit:
        table = table.slice(0, limit)
    # Convert to list-of-dicts for JSON (vectorized, ~100x faster than row-by-row)
    columns = [col.name for col in table.schema]
    rows = table.to_pylist()
    return {
        "dataset": dataset,
        "columns": columns,
        "rows": rows,
        "total_rows": cohort[dataset].num_rows,
        "returned_rows": len(rows),
    }


@app.post("/api/union")
async def do_union(req: UnionRequest):
    """Union multiple .cohort files into one."""
    paths = [COHORT_DIR / f for f in req.files]
    for p in paths:
        if not p.exists():
            raise HTTPException(404, f"File not found: {p.name}")
    out = COHORT_DIR / req.output
    cohort_union([str(p) for p in paths], str(out), name=req.name)
    info = cohort_inspect(str(out))
    return {"status": "ok", "output": req.output, "inspection": info.model_dump()}


@app.post("/api/overlap")
async def compute_overlap(req: OverlapRequest):
    """Compute set overlaps between cohort files on a key column.

    Returns per-file sets and all intersection sizes for Venn/UpSet rendering.
    """
    from itertools import combinations

    file_sets: dict[str, set] = {}
    for fname in req.files:
        path = COHORT_DIR / fname
        if not path.exists():
            raise HTTPException(404, f"File not found: {fname}")
        cohort = cohort_read(str(path))
        dataset = cohort.datasets[0]
        table = cohort[dataset]
        if req.on not in table.column_names:
            raise HTTPException(400, f"Column '{req.on}' not found in {fname}")
        col = table.column(req.on)
        file_sets[fname] = set(v.as_py() for v in col)

    names = list(file_sets.keys())
    sizes = {n: len(file_sets[n]) for n in names}

    # Compute all pairwise and higher-order intersections
    # Encode subsets as bitmasks for UpSet-style data
    regions = []
    for r in range(1, len(names) + 1):
        for combo in combinations(range(len(names)), r):
            s = file_sets[names[combo[0]]]
            for i in combo[1:]:
                s = s & file_sets[names[i]]
            # Exclusive count: in exactly these sets, not in others
            others = set(range(len(names))) - set(combo)
            exclusive = s
            for i in others:
                exclusive = exclusive - file_sets[names[i]]
            mask = sum(1 << i for i in combo)
            regions.append({
                "mask": mask,
                "sets": [names[i] for i in combo],
                "intersection_size": len(s),
                "exclusive_size": len(exclusive),
            })

    # Pairwise matrix
    matrix = []
    for i, a in enumerate(names):
        row = []
        for j, b in enumerate(names):
            row.append(len(file_sets[a] & file_sets[b]))
        matrix.append(row)

    return {
        "files": names,
        "sizes": sizes,
        "regions": regions,
        "matrix": matrix,
        "key_column": req.on,
    }


@app.post("/api/intersect")
async def do_intersect(req: IntersectRequest):
    """Intersect .cohort files on a column."""
    paths = [COHORT_DIR / f for f in req.files]
    for p in paths:
        if not p.exists():
            raise HTTPException(404, f"File not found: {p.name}")
    out = COHORT_DIR / req.output
    cohort_intersect([str(p) for p in paths], str(out), on=req.on, name=req.name)
    info = cohort_inspect(str(out))
    return {"status": "ok", "output": req.output, "inspection": info.model_dump()}


@app.get("/api/tables")
async def list_tables():
    """List available SQL table names from the catalog."""
    cat = _get_catalog()
    return {"tables": cat.tables()}


@app.get("/api/schema")
async def get_schema():
    """Get table schemas for the query console schema browser."""
    cat = _get_catalog()
    tables = cat.tables()
    schema = []
    for t in tables:
        try:
            result = cat.query(f"SELECT * FROM {t} LIMIT 0")
            columns = []
            for field in result.schema:
                columns.append({
                    "name": field.name,
                    "type": str(field.type),
                })
            schema.append({"table": t, "columns": columns})
        except Exception:
            schema.append({"table": t, "columns": []})
    return {"schema": schema}


def _catalog_stats(source_dir: Path | None = None) -> dict:
    """Parse .kohort-catalog.json for scan stats."""
    import json

    cat_path = (source_dir or COHORT_DIR) / ".kohort-catalog.json"
    if not cat_path.exists():
        return {}
    try:
        data = json.loads(cat_path.read_text())
    except Exception:
        return {}
    files = data.get("files", [])
    total_rows = 0
    total_bytes = 0
    for f in files:
        for m in f.get("datasets", []):
            total_rows += m.get("num_rows", 0)
            total_bytes += m.get("compressed_size", 0)
    return {
        "catalog_files": len(files),
        "catalog_rows": total_rows,
        "catalog_bytes": total_bytes,
    }


_polars_ctx = None


def _get_polars_ctx():
    """Get or create a Polars SQLContext with registered tables."""
    global _polars_ctx
    if _polars_ctx is not None:
        return _polars_ctx
    import polars as pl

    ctx = pl.SQLContext()
    cat = _get_catalog()
    for table_name in cat.tables():
        try:
            pa_table = cat.query(f'SELECT * FROM "{table_name}"')
            df = pl.from_arrow(pa_table)
            ctx.register(table_name, df)
        except Exception:
            pass
    _polars_ctx = ctx
    return ctx


@app.post("/api/query")
async def run_query(req: QueryRequest):
    """Run a SQL query against .cohort files via DataFusion, DuckDB, or Polars."""
    engine = req.engine or "datafusion"
    stats = _catalog_stats()
    t0 = time.perf_counter()

    try:
        if engine == "duckdb":
            conn = _get_duckdb_conn()
            table = conn.execute(req.sql).fetch_arrow_table()
        elif engine == "polars":
            import polars as pl
            ctx = _get_polars_ctx()
            result = ctx.execute(req.sql)
            table = result.collect().to_arrow()
        else:  # datafusion (default)
            cat = _get_catalog(partitioned=req.use_partitions)
            if req.use_partitions:
                stats = _catalog_stats(COHORT_DIR / "partitions")
            table = cat.query(req.sql)
    except Exception as e:
        raise HTTPException(400, str(e))
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    total_rows = table.num_rows
    if table.num_rows > req.limit:
        table = table.slice(0, req.limit)

    columns = [col.name for col in table.schema]
    rows = table.to_pylist()

    return {
        "columns": columns,
        "rows": rows,
        "total_rows": total_rows,
        "returned_rows": len(rows),
        "elapsed_ms": elapsed_ms,
        "sql": req.sql,
        "engine": engine,
        "partitioned": req.use_partitions and engine == "datafusion",
        **stats,
    }


@app.get("/api/partition/status")
async def partition_status():
    """Check if partitions exist and return partition info."""
    parts_dir = COHORT_DIR / "partitions"
    parts = sorted(parts_dir.glob("*.cohort")) if parts_dir.exists() else []
    return {
        "has_partitions": len(parts) > 0,
        "num_partitions": len(parts),
        "partition_dir": str(parts_dir),
        "files": [f.name for f in parts],
    }


@app.post("/api/partition")
async def do_partition(req: PartitionRequest):
    """Partition a .cohort file into smaller files for parallel queries."""
    global _catalog_partitioned
    import asyncio

    path = COHORT_DIR / req.file
    if not path.exists():
        raise HTTPException(404, f"File not found: {req.file}")

    parts_dir = COHORT_DIR / "partitions"
    t0 = time.perf_counter()
    loop = asyncio.get_event_loop()
    try:
        output_paths = await loop.run_in_executor(
            None,
            lambda: cohort_partition(
                str(path),
                str(parts_dir),
                num_partitions=req.num_partitions,
                target_rows=req.target_rows,
            ),
        )
    except Exception as e:
        raise HTTPException(400, str(e))

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Invalidate cached partitioned catalog so it gets rebuilt
    _catalog_partitioned = None

    return {
        "status": "ok",
        "num_partitions": len(output_paths),
        "files": [Path(p).name for p in output_paths],
        "elapsed_ms": elapsed_ms,
    }


@app.post("/api/benchmark")
async def run_benchmark():
    """Run clear benchmarks comparing formats and access patterns."""
    from cohort_viz.benchmarks import run_comprehensive_benchmark

    try:
        return run_comprehensive_benchmark(COHORT_DIR)
    except Exception as e:
        raise HTTPException(500, f"Benchmark failed: {str(e)}")


# ── Similarity Engine (lazy init) ─────────────────────────────

_similarity_engines: dict[str, "SimilarityEngine"] = {}


def _get_similarity_engine(cohort_file: str):
    """Get or create similarity engine for a cohort file."""
    from cohort_viz.similarity_engine import SimilarityEngine

    if cohort_file not in _similarity_engines:
        engine = SimilarityEngine()
        path = COHORT_DIR / cohort_file
        if not path.exists():
            raise HTTPException(404, f"File not found: {cohort_file}")
        engine.build_index(str(path))
        _similarity_engines[cohort_file] = engine
    return _similarity_engines[cohort_file]


@app.get("/api/similarity/status/{filename:path}")
async def similarity_status(filename: str):
    """Check if similarity index is built for a cohort file."""
    return {
        "indexed": filename in _similarity_engines,
        "num_patients": len(_similarity_engines[filename].rows) if filename in _similarity_engines else 0,
    }


@app.post("/api/similarity/build/{filename:path}")
async def build_similarity_index(filename: str):
    """Build similarity index for a cohort file (may take a few seconds first time)."""
    import asyncio

    t0 = time.perf_counter()
    loop = asyncio.get_event_loop()
    engine = await loop.run_in_executor(None, _get_similarity_engine, filename)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "status": "ok",
        "num_patients": len(engine.rows),
        "embedding_dim": engine.embeddings.shape[1] if engine.embeddings is not None else 0,
        "elapsed_ms": elapsed_ms,
    }


@app.post("/api/similarity/search")
async def similarity_search(req: SimilaritySearchRequest):
    """Search for similar patients."""
    import asyncio

    loop = asyncio.get_event_loop()
    engine = await loop.run_in_executor(None, _get_similarity_engine, req.cohort_file)

    t0 = time.perf_counter()
    try:
        results = engine.search(
            query_patient_id=req.patient_id,
            query_text=req.query_text,
            filters=req.filters,
            limit=req.limit,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "results": results,
        "num_results": len(results),
        "elapsed_ms": elapsed_ms,
        "query_patient_id": req.patient_id,
        "query_text": req.query_text,
        "filters": req.filters,
    }


@app.get("/api/similarity/patients/{filename:path}")
async def list_patients(filename: str, limit: int = Query(default=100, le=1000)):
    """List patients in a cohort for the similarity search dropdown."""
    import asyncio

    loop = asyncio.get_event_loop()
    engine = await loop.run_in_executor(None, _get_similarity_engine, filename)
    patients = []
    for row in engine.rows[:limit]:
        pid = row.get("PATIENT_NUMBER") or row.get("patient_number")
        state = row.get("PATIENT_STATE") or row.get("patient_state") or ""
        diag = row.get("DIAGNOSIS_CODES") or row.get("diagnosis_codes") or ""
        if pid:
            patients.append({
                "id": pid,
                "state": state,
                "diagnosis_preview": diag[:60] + "..." if len(diag) > 60 else diag,
            })
    return {"patients": patients, "total": len(engine.rows)}


@app.get("/api/lineage")
async def lineage():
    """Build a DAG of cohort lineage from all .cohort files."""
    import re

    files_list = sorted(COHORT_DIR.glob("*.cohort"))
    nodes = []
    edges = []
    seen_nodes: set[str] = set()

    for f in files_list:
        rel = str(f.relative_to(COHORT_DIR))
        info = cohort_inspect(str(f))
        d = info.cohort_definition
        rows = sum(m.num_rows for m in info.datasets) if info.datasets else 0

        # Check revision history for union/intersect markers
        is_derived = False
        derived_op = None
        derived_sources: list[str] = []
        for rev in (d.revision_history or []):
            msg = rev.message or ""
            m_union = re.match(r"Union of (.+)", msg)
            m_inter = re.match(r"Intersect of (.+)", msg)
            if m_union:
                is_derived = True
                derived_op = "union"
                # Parse comma-separated + " ∩ " separated file names
                derived_sources = [
                    s.strip()
                    for s in re.split(r"[,∩]", m_union.group(1))
                    if s.strip()
                ]
            elif m_inter:
                is_derived = True
                derived_op = "intersect"
                derived_sources = [
                    s.strip()
                    for s in re.split(r"[,∩]", m_inter.group(1))
                    if s.strip()
                ]

        node_type = "derived" if is_derived else "cohort"
        label = d.name if d.name and d.name != rel else rel.replace(".cohort", "")
        if rel in seen_nodes:
            # Update the stub node with real data
            for n in nodes:
                if n["id"] == rel:
                    n["type"] = node_type
                    n["label"] = label
                    n["rows"] = rows
                    break
        else:
            nodes.append({
                "id": rel,
                "type": node_type,
                "label": label,
                "rows": rows,
            })
            seen_nodes.add(rel)

        if is_derived:
            for src in derived_sources:
                if src not in seen_nodes:
                    # Source may be a cohort file not yet processed
                    nodes.append({"id": src, "type": "cohort", "label": src.replace(".cohort", ""), "rows": 0})
                    seen_nodes.add(src)
                edges.append({"source": src, "target": rel, "operation": derived_op})
        else:
            # Add source system + source table nodes
            src_sys = d.lineage.source_system if d.lineage else "Unknown"
            if isinstance(src_sys, list):
                src_sys = src_sys[0] if src_sys else "Unknown"
            sys_id = f"sys:{src_sys}"
            if sys_id not in seen_nodes:
                nodes.append({"id": sys_id, "type": "source_system", "label": src_sys})
                seen_nodes.add(sys_id)

            src_tables = d.lineage.source_tables if d.lineage else []
            for tbl in (src_tables or []):
                tbl_id = f"tbl:{tbl}"
                if tbl_id not in seen_nodes:
                    nodes.append({"id": tbl_id, "type": "source_table", "label": tbl})
                    seen_nodes.add(tbl_id)
                if not any(e["source"] == sys_id and e["target"] == tbl_id for e in edges):
                    edges.append({"source": sys_id, "target": tbl_id, "operation": "provides"})
                edges.append({"source": tbl_id, "target": rel, "operation": "extract"})

            # If no source tables, link directly from source system
            if not src_tables:
                edges.append({"source": sys_id, "target": rel, "operation": "extract"})

    return {"nodes": nodes, "edges": edges}


# ── Lakehouse stores (lazy init) ─────────────────────────────

_lakehouse_stores = None


def _get_lakehouse_stores():
    """Initialize lakehouse stores."""
    global _lakehouse_stores
    if _lakehouse_stores is None:
        from lakehouse import VectorStore, BitmapStore, PolicyStore, AuditLogger

        _lakehouse_stores = {
            "vector": VectorStore(str(COHORT_DIR / "indexes" / "vectors")),
            "bitmap": BitmapStore(str(COHORT_DIR / "indexes" / "bitmaps"), kohort_path=str(COHORT_DIR)),
            "policy": PolicyStore(str(COHORT_DIR / "policies")),
            "audit": AuditLogger(str(COHORT_DIR / "audit_logs")),
        }
    return _lakehouse_stores


# ── Lakehouse API routes ─────────────────────────────────────


@app.get("/api/lakehouse/status")
async def lakehouse_status():
    """Get lakehouse status and available indexes."""
    stores = _get_lakehouse_stores()

    # List vector indexes (universes)
    vector_path = COHORT_DIR / "indexes" / "vectors"
    universes = []
    if vector_path.exists():
        for d in vector_path.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                universes.append(d.name)

    # List bitmap cohorts
    cohorts = []
    try:
        for cid in stores["bitmap"].list_cohorts():
            versions = stores["bitmap"].list_versions(cid)
            latest = versions[-1] if versions else "v1"
            try:
                bitmap = stores["bitmap"].read_membership(cid, latest)
                count = len(bitmap)
            except Exception:
                count = 0
            cohorts.append({
                "id": cid,
                "version": latest,
                "cardinality": count,
            })
    except Exception:
        pass

    # List policies
    policies = []
    try:
        for p in stores["policy"].list_policies(enabled_only=False):
            policies.append({
                "id": p.policy_id,
                "name": p.name,
                "type": p.filter_type,
                "enabled": p.enabled,
            })
    except Exception:
        pass

    return {
        "universes": universes,
        "cohorts": cohorts,
        "policies": policies,
    }


@app.post("/api/lakehouse/search")
async def lakehouse_search(req: LakehouseSearchRequest):
    """Semantic search for patients using vector embeddings, with Iceberg join."""
    import time
    from pyiceberg.catalog.sql import SqlCatalog
    from pathlib import Path

    stores = _get_lakehouse_stores()
    t0 = time.perf_counter()

    try:
        # Step 1: Vector search to get patient IDs
        results = stores["vector"].search_by_text(
            universe_id=req.universe_id,
            query_text=req.query_text,
            top_k=req.top_k,
        )
        vector_elapsed = round((time.perf_counter() - t0) * 1000, 2)

        # Get unique patient IDs with their best scores
        patient_scores = {}
        for r in results:
            if r.patient_id not in patient_scores or r.score > patient_scores[r.patient_id]:
                patient_scores[r.patient_id] = r.score
        patient_ids = list(patient_scores.keys())

        # Step 2: Join with Iceberg table to get full records
        # Uses predicate pushdown - Iceberg only reads matching rows
        iceberg_data = {}
        iceberg_elapsed = 0
        try:
            warehouse_path = Path("./lakehouse/warehouse")
            if (warehouse_path / "catalog.db").exists() and patient_ids:
                t1 = time.perf_counter()
                from pyiceberg.expressions import In, Reference

                catalog = SqlCatalog("local", **{
                    "uri": f"sqlite:///{warehouse_path}/catalog.db",
                    "warehouse": str(warehouse_path),
                })
                table = catalog.load_table("default.tx_claims")

                # Predicate pushdown - filter at Iceberg level, not pandas
                row_filter = In(Reference("PATIENT_NUMBER"), tuple(patient_ids))
                df = table.scan(row_filter=row_filter).to_pandas()

                # Group by patient and take first record
                for _, row in df.drop_duplicates(subset=["PATIENT_NUMBER"]).iterrows():
                    iceberg_data[row["PATIENT_NUMBER"]] = {
                        "procedure_code": row.get("PROCEDURE_CODE", ""),
                        "service_date": str(row.get("SERVICE_DATE", ""))[:10],
                        "place_of_service": row.get("PLACE_OF_SERVICE", ""),
                        "diagnosis_codes": row.get("DIAGNOSIS_CODES", ""),
                        "patient_state": row.get("PATIENT_STATE", ""),
                        "patient_sex": row.get("PATIENT_SEX", ""),
                        "patient_yob": str(row.get("PATIENT_YOB_DATE_UNCERTIFIED_TOP_CODED", ""))[:4],
                    }
                iceberg_elapsed = round((time.perf_counter() - t1) * 1000, 2)
        except Exception as e:
            print(f"Iceberg join error: {e}")

        total_elapsed = round((time.perf_counter() - t0) * 1000, 2)

        # Combine results
        combined = []
        for pid, score in patient_scores.items():
            record = {
                "patient_id": pid,
                "score": round(score, 4),
            }
            if pid in iceberg_data:
                record.update(iceberg_data[pid])
            else:
                # Fallback to vector metadata
                for r in results:
                    if r.patient_id == pid:
                        record["patient_state"] = r.metadata.get("state", "")
                        record["diagnosis_codes"] = r.metadata.get("diagnosis_preview", "")
                        break
            combined.append(record)

        # Sort by score descending
        combined.sort(key=lambda x: x["score"], reverse=True)

        return {
            "results": combined,
            "num_results": len(combined),
            "elapsed_ms": total_elapsed,
            "vector_ms": vector_elapsed,
            "iceberg_ms": iceberg_elapsed,
            "query": req.query_text,
            "universe_id": req.universe_id,
        }
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/lakehouse/similar")
async def lakehouse_similar(req: LakehouseSimilarRequest):
    """Find patients similar to a given patient, with Iceberg join."""
    import time
    from pyiceberg.catalog.sql import SqlCatalog
    from pathlib import Path

    stores = _get_lakehouse_stores()
    t0 = time.perf_counter()

    try:
        # Step 1: Vector search to get similar patient IDs
        results = stores["vector"].search_by_patient(
            universe_id=req.universe_id,
            patient_id=req.patient_id,
            top_k=req.top_k,
        )
        vector_elapsed = round((time.perf_counter() - t0) * 1000, 2)

        # Get unique patient IDs with their best scores
        patient_scores = {}
        for r in results:
            if r.patient_id not in patient_scores or r.score > patient_scores[r.patient_id]:
                patient_scores[r.patient_id] = r.score
        patient_ids = list(patient_scores.keys())

        # Step 2: Join with Iceberg table (predicate pushdown)
        iceberg_data = {}
        iceberg_elapsed = 0
        try:
            warehouse_path = Path("./lakehouse/warehouse")
            if (warehouse_path / "catalog.db").exists() and patient_ids:
                t1 = time.perf_counter()
                from pyiceberg.expressions import In, Reference

                catalog = SqlCatalog("local", **{
                    "uri": f"sqlite:///{warehouse_path}/catalog.db",
                    "warehouse": str(warehouse_path),
                })
                table = catalog.load_table("default.tx_claims")

                # Predicate pushdown - only read matching rows
                row_filter = In(Reference("PATIENT_NUMBER"), tuple(patient_ids))
                df = table.scan(row_filter=row_filter).to_pandas()

                for _, row in df.drop_duplicates(subset=["PATIENT_NUMBER"]).iterrows():
                    iceberg_data[row["PATIENT_NUMBER"]] = {
                        "procedure_code": row.get("PROCEDURE_CODE", ""),
                        "service_date": str(row.get("SERVICE_DATE", ""))[:10],
                        "place_of_service": row.get("PLACE_OF_SERVICE", ""),
                        "diagnosis_codes": row.get("DIAGNOSIS_CODES", ""),
                        "patient_state": row.get("PATIENT_STATE", ""),
                        "patient_sex": row.get("PATIENT_SEX", ""),
                        "patient_yob": str(row.get("PATIENT_YOB_DATE_UNCERTIFIED_TOP_CODED", ""))[:4],
                    }
                iceberg_elapsed = round((time.perf_counter() - t1) * 1000, 2)
        except Exception as e:
            print(f"Iceberg join error: {e}")

        total_elapsed = round((time.perf_counter() - t0) * 1000, 2)

        # Combine results
        combined = []
        for pid, score in patient_scores.items():
            record = {"patient_id": pid, "score": round(score, 4)}
            if pid in iceberg_data:
                record.update(iceberg_data[pid])
            else:
                for r in results:
                    if r.patient_id == pid:
                        record["patient_state"] = r.metadata.get("state", "")
                        record["diagnosis_codes"] = r.metadata.get("diagnosis_preview", "")
                        break
            combined.append(record)

        combined.sort(key=lambda x: x["score"], reverse=True)

        return {
            "results": combined,
            "num_results": len(combined),
            "elapsed_ms": total_elapsed,
            "vector_ms": vector_elapsed,
            "iceberg_ms": iceberg_elapsed,
            "reference_patient": req.patient_id,
            "universe_id": req.universe_id,
        }
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/lakehouse/index")
async def lakehouse_index(req: LakehouseIndexRequest):
    """Build vector and/or bitmap indexes from a cohort file."""
    import time
    from lakehouse import build_index_from_cohort, extract_membership_from_cohort

    stores = _get_lakehouse_stores()
    cohort_path = str((COHORT_DIR / req.cohort_file).resolve())
    universe = req.universe_id or Path(req.cohort_file).stem.replace("-", "_").replace(".", "_")

    results = {"universe_id": universe, "cohort_file": req.cohort_file}

    if req.build_vector:
        t0 = time.perf_counter()
        try:
            stats = build_index_from_cohort(stores["vector"], universe, cohort_path)
            results["vector"] = {
                "success": True,
                "num_vectors": stats.num_vectors,
                "dimension": stats.dimension,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            }
        except Exception as e:
            results["vector"] = {"success": False, "error": str(e)}

    if req.build_bitmap:
        t0 = time.perf_counter()
        try:
            meta = extract_membership_from_cohort(stores["bitmap"], cohort_path, universe, "v1")
            results["bitmap"] = {
                "success": True,
                "cardinality": meta.cardinality,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            }
        except Exception as e:
            results["bitmap"] = {"success": False, "error": str(e)}

    return results


@app.post("/api/lakehouse/bitmap/operation")
async def lakehouse_bitmap_op(req: LakehouseBitmapOpRequest):
    """Perform set operations on bitmap cohorts."""
    import time

    stores = _get_lakehouse_stores()
    t0 = time.perf_counter()

    try:
        if req.operation == "intersect":
            result = stores["bitmap"].intersect(req.cohort_ids)
        elif req.operation == "union":
            result = stores["bitmap"].union(req.cohort_ids)
        elif req.operation == "difference":
            if len(req.cohort_ids) != 2:
                raise ValueError("Difference requires exactly 2 cohorts")
            result = stores["bitmap"].difference(req.cohort_ids[0], req.cohort_ids[1])
        else:
            raise ValueError(f"Unknown operation: {req.operation}")

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "operation": req.operation,
            "cohort_ids": req.cohort_ids,
            "result_cardinality": result.result_cardinality,
            "elapsed_ms": elapsed_ms,
        }
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/lakehouse/policies")
async def lakehouse_policies():
    """List all ABAC policies."""
    stores = _get_lakehouse_stores()
    policies = stores["policy"].list_policies(enabled_only=False)

    return {
        "policies": [
            {
                "id": p.policy_id,
                "name": p.name,
                "description": getattr(p, 'description', None),
                "filter_type": p.filter_type,
                "predicate": p.predicate,
                "bitmap_cohort": getattr(p, 'bitmap_cohort', None),
                "kohort_files": getattr(p, 'kohort_files', None),
                "universe_operation": getattr(p, 'universe_operation', 'union'),
                "universe_ids": getattr(p, 'universe_ids', None),
                "max_rows": p.max_rows,
                "masked_columns": getattr(p, 'masked_columns', None),
                "mask_value": getattr(p, 'mask_value', None),
                "user_roles": getattr(p, 'user_roles', None),
                "allowed_sources": getattr(p, 'allowed_sources', None),
                "source_bitset": getattr(p, 'source_bitset', None),
                "created_at": getattr(p, 'created_at', None),
                "created_by": getattr(p, 'created_by', None),
                "priority": getattr(p, 'priority', None),
                "customer_tier": getattr(p, 'customer_tier', None),
                "patient_count": getattr(p, 'patient_count', None),
                "enabled": p.enabled,
            }
            for p in policies
        ]
    }


@app.get("/api/lakehouse/universes")
async def lakehouse_universes():
    """Compute and return customer universes from ABAC policies.

    For each policy with kohort_files, computes the union/intersection
    of the referenced kohort bitmaps to create the customer's universe.
    """
    import time

    stores = _get_lakehouse_stores()
    policies = stores["policy"].list_policies(enabled_only=True)

    universes = []
    for p in policies:
        kohort_files = getattr(p, 'kohort_files', None)
        if not kohort_files:
            continue

        operation = getattr(p, 'universe_operation', 'union')

        t0 = time.perf_counter()
        try:
            result = stores["bitmap"].compute_customer_universe(
                kohort_files=kohort_files,
                operation=operation,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

            universes.append({
                "policy_id": p.policy_id,
                "policy_name": p.name,
                "kohort_files": kohort_files,
                "operation": operation,
                "patient_count": result.result_cardinality,
                "compute_ms": elapsed_ms,
                "description": p.description if hasattr(p, 'description') else None,
            })
        except Exception as e:
            universes.append({
                "policy_id": p.policy_id,
                "policy_name": p.name,
                "kohort_files": kohort_files,
                "operation": operation,
                "patient_count": 0,
                "compute_ms": 0,
                "error": str(e),
            })

    return {
        "universes": universes,
        "total": len(universes),
    }


@app.get("/api/lakehouse/kohorts")
async def lakehouse_kohorts():
    """List all atlas_*.kohort files with full details."""
    stores = _get_lakehouse_stores()

    kohorts = []
    legacy = []

    # List kohort files
    for k in stores["bitmap"].list_kohorts():
        kohorts.append({
            "name": k.name,
            "path": k.path,
            "patient_count": k.patient_count,
            "has_membership": k.has_membership,
            "has_vector_index": k.has_vector_index,
            "vector_path": k.vector_path,
            "modalities": k.modalities,
            "type": "kohort",
        })

    # List legacy bitmaps
    for c in stores["bitmap"].list_legacy_cohorts():
        cohort_id = c.replace("_legacy", "")
        try:
            bitmap = stores["bitmap"].read_membership(cohort_id)
            count = len(bitmap)
        except Exception:
            count = 0
        legacy.append({
            "name": cohort_id,
            "path": f"indexes/bitmaps/memberships/{c}/v1.roaring",
            "patient_count": count,
            "type": "legacy",
        })

    return {
        "kohorts": kohorts,
        "legacy": legacy,
        "total_kohorts": len(kohorts),
        "total_legacy": len(legacy),
    }


@app.get("/api/lakehouse/kohort/{name}/data")
async def lakehouse_kohort_data(
    name: str,
    modality: str | None = None,
    limit: int = 100,
    policy_id: str | None = None
):
    """Query data from a specific kohort file with ABAC filtering.

    Args:
        name: The kohort filename (e.g., 'atlas_tx.kohort')
        modality: Optional modality label to query. If not provided, uses first available.
        limit: Max rows to return (default 100)
        policy_id: Optional policy ID for source bitset filtering
    """
    import time
    from cohort_viz.atlas_kohort import AtlasKohortReader
    from pathlib import Path

    t0 = time.perf_counter()

    # Find the kohort file
    stores = _get_lakehouse_stores()
    kohort_path = stores["bitmap"].kohort_path / name

    if not kohort_path.exists():
        return {"error": f"Kohort file not found: {name}"}

    try:
        reader = AtlasKohortReader(kohort_path)

        # Get available modalities
        modalities = reader.list_modalities()

        if not modalities:
            return {
                "name": name,
                "modalities": [],
                "error": "No parquet modalities in this kohort file",
                "columns": [],
                "rows": [],
                "total_rows": 0,
            }

        # Use specified modality or first available
        target_modality = modality if modality else modalities[0]

        if target_modality not in modalities:
            return {"error": f"Modality '{target_modality}' not found. Available: {modalities}"}

        # Read the data
        table = reader.get_modality(target_modality)
        if table is None:
            return {"error": f"Failed to read modality: {target_modality}"}

        # Convert to records
        total_rows = table.num_rows
        columns = table.schema.names

        # Limit rows
        if total_rows > limit:
            table = table.slice(0, limit)

        # Convert to list of dicts
        import math
        import numpy as np

        df = table.to_pandas()

        # Apply source bitset filter if policy_id specified (fast bitwise AND)
        source_filter_applied = False
        if policy_id and policy_id != 'komodo':
            policy = stores["policy"].get_policy(policy_id)
            if policy:
                source_bitset = getattr(policy, 'source_bitset', None)
                if source_bitset and 'SOURCES_BITSET' in df.columns:
                    # Bitwise filter: (row_bitset & policy_bitset) != 0
                    # This is screaming fast - O(1) per row with vectorized ops
                    original_count = len(df)
                    df = df[df['SOURCES_BITSET'].fillna(0).astype(int) & source_bitset != 0]
                    source_filter_applied = True
                    print(f"Source bitset filter: {original_count} -> {len(df)} rows (policy={policy_id}, bitset=0x{source_bitset:X})")

        rows = df.to_dict(orient="records")

        # Convert any non-JSON-serializable types
        def make_json_safe(v):
            if v is None:
                return None
            if hasattr(v, 'isoformat'):  # datetime
                return v.isoformat()
            if isinstance(v, (np.floating, float)):
                if math.isnan(v) or math.isinf(v):
                    return None
                return float(v)
            if isinstance(v, (np.integer, int)):
                return int(v)
            if isinstance(v, np.bool_):
                return bool(v)
            if str(type(v)) == "<class 'pandas._libs.missing.NAType'>":
                return None
            if str(type(v)).startswith("<class 'numpy"):
                return str(v)
            return v

        for row in rows:
            for k, v in list(row.items()):
                row[k] = make_json_safe(v)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "name": name,
            "modality": target_modality,
            "modalities": modalities,
            "columns": columns,
            "rows": rows,
            "total_rows": total_rows,
            "returned_rows": len(rows),
            "limit": limit,
            "elapsed_ms": round(elapsed_ms, 2),
            "source_filter_applied": source_filter_applied,
            "policy_id": policy_id,
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/api/lakehouse/policies")
async def lakehouse_create_policy(req: LakehousePolicyRequest):
    """Create a new ABAC policy."""
    from lakehouse import Policy

    stores = _get_lakehouse_stores()
    policy = Policy(
        policy_id=req.policy_id,
        name=req.name,
        filter_type=req.filter_type,
        predicate=req.predicate,
        max_rows=req.max_rows,
        enabled=True,
    )
    stores["policy"].create_policy(policy)

    return {"status": "created", "policy_id": req.policy_id}


@app.delete("/api/lakehouse/policies/{policy_id}")
async def lakehouse_delete_policy(policy_id: str):
    """Delete an ABAC policy."""
    stores = _get_lakehouse_stores()
    stores["policy"].delete_policy(policy_id)
    return {"status": "deleted", "policy_id": policy_id}


@app.get("/api/lakehouse/audit")
async def lakehouse_audit(limit: int = Query(default=100, le=1000), user: str | None = None):
    """Get audit log entries."""
    stores = _get_lakehouse_stores()
    entries = stores["audit"].query_logs(limit=limit, user_id=user)

    return {
        "entries": [
            {
                "timestamp": e.timestamp,
                "user_id": e.user_id,
                "query_type": e.query_type,
                "success": e.success,
                "rows_returned": e.rows_returned,
                "elapsed_ms": e.elapsed_ms,
            }
            for e in entries
        ],
        "total": len(entries),
    }


@app.get("/api/lakehouse/audit/stats")
async def lakehouse_audit_stats():
    """Get audit statistics."""
    stores = _get_lakehouse_stores()
    return stores["audit"].get_stats()


# ── Atlas Intelligent Lakehouse UI ──────────────────────────────────────

@app.get("/atlas", response_class=HTMLResponse)
async def atlas_ui():
    """Atlas - Intelligent Virtual Lakehouse landing page."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Atlas - Intelligent Virtual Lakehouse</title>
    <style>
        *{margin:0;padding:0;box-sizing:border-box}
        :root{
          --bg:#0d1117;--surface:#161b22;--border:#30363d;--text:#e6edf3;
          --muted:#8b949e;--accent:#58a6ff;--green:#3fb950;
          --red:#f85149;--yellow:#d29922;--card:#1c2128;--card-hover:#22272e;
        }
        html,body{height:100%;font-family:-apple-system,system-ui,sans-serif;background:var(--bg);color:var(--text)}
        .app-container{display:flex;min-height:100vh}
        /* Sidebar */
        .sidebar{width:260px;background:var(--surface);border-right:1px solid var(--border);display:flex;flex-direction:column}
        .logo{padding:16px 20px;border-bottom:1px solid var(--border)}
        .logo h1{font-size:28px;font-weight:700;color:var(--accent)}
        .logo span{font-size:11px;color:var(--muted);display:block;margin-top:2px}
        .nav-section{padding:0 8px;margin-bottom:16px}
        .nav-section-title{font-size:11px;text-transform:uppercase;color:var(--muted);padding:8px 12px;letter-spacing:.5px;font-weight:600}
        .nav-item{display:flex;align-items:center;padding:8px 12px;cursor:pointer;margin-bottom:2px;font-size:13px;color:var(--muted)}
        .nav-item:hover{background:var(--card);color:var(--text)}
        .nav-item.active{background:rgba(88,166,255,.12);color:var(--text)}
        .nav-item .icon{width:20px;margin-right:10px;text-align:center;font-size:14px}
        /* Main Content */
        .main-content{flex:1;padding:24px 28px;overflow-y:auto}
        .main-content::-webkit-scrollbar{width:5px}
        .main-content::-webkit-scrollbar-thumb{background:var(--border)}
        .section{display:none}
        .section.active{display:block}
        .section-header{margin-bottom:20px}
        .section-header h2{font-size:20px;font-weight:600;margin-bottom:4px}
        .section-header p{color:var(--muted);font-size:13px}
        /* Cards */
        .card{background:var(--card);border:1px solid var(--border);padding:16px;margin-bottom:16px}
        .card h3{font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid var(--border)}
        /* Stats Grid */
        .stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:20px}
        .stat-card{background:var(--card);border:1px solid var(--border);padding:16px;text-align:center}
        .stat-card .value{font-size:28px;font-weight:700;color:var(--accent)}
        .stat-card .label{font-size:11px;color:var(--muted);margin-top:4px;text-transform:uppercase}
        /* Search Box */
        .search-box{display:flex;gap:8px;margin-bottom:16px}
        .search-input{flex:1;padding:8px 12px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:13px;outline:none}
        .search-input:focus{border-color:var(--accent)}
        .search-input::placeholder{color:#2d333b}
        .btn{padding:8px 16px;border:none;cursor:pointer;font-size:13px;font-weight:500}
        .btn-primary{background:var(--accent);color:#000}
        .btn-primary:hover{background:#79c0ff}
        .btn-secondary{background:var(--card);color:var(--text);border:1px solid var(--border)}
        .btn-secondary:hover{background:var(--card-hover);border-color:#444c56}
        /* Results Table */
        .results-table{width:100%;border-collapse:collapse;margin-top:12px;font-size:13px}
        .results-table th,.results-table td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--border)}
        .results-table th{font-size:11px;text-transform:uppercase;color:var(--muted);font-weight:600}
        .results-table tr:hover{background:var(--card)}
        .score-badge{display:inline-block;padding:2px 6px;font-size:10px;font-weight:600}
        .score-high{background:rgba(63,185,80,.15);color:var(--green)}
        .score-med{background:rgba(210,153,34,.15);color:var(--yellow)}
        .score-low{background:rgba(248,81,73,.15);color:var(--red)}
        /* Loading */
        .loading{display:none;text-align:center;padding:40px}
        .loading.active{display:block}
        .spinner{width:32px;height:32px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin 1s linear infinite;margin:0 auto 12px}
        @keyframes spin{to{transform:rotate(360deg)}}
        /* Policy Cards */
        .policy-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
        .policy-card{background:var(--card);border:1px solid var(--border);padding:14px}
        .policy-card h4{font-size:14px;font-weight:500;margin-bottom:6px}
        .policy-card .desc{font-size:12px;color:var(--muted);margin-bottom:12px}
        .policy-meta{display:flex;gap:8px;flex-wrap:wrap}
        .policy-tag{font-size:10px;padding:2px 6px;background:rgba(88,166,255,.15);color:var(--accent);font-weight:600}
        /* Cohort List */
        .cohort-list{display:flex;flex-direction:column;gap:6px}
        .cohort-item{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:var(--card);border:1px solid var(--border)}
        .cohort-item .name{font-size:13px;font-weight:500}
        .cohort-item .count{font-size:12px;color:var(--green);font-weight:600}
        /* Audit Log */
        .audit-entry{display:flex;gap:12px;padding:8px 0;border-bottom:1px solid var(--border);font-size:13px}
        .audit-time{font-size:12px;color:var(--muted);width:70px}
        .audit-type{font-size:10px;padding:2px 6px;background:rgba(88,166,255,.15);color:var(--accent);font-weight:600}
        .timing-info{font-size:12px;color:var(--muted);margin-top:12px;padding:10px;background:var(--bg);border:1px solid var(--border)}
        .timing-info span{margin-right:16px}
        /* Similar patients */
        .patient-input-group{display:flex;gap:8px;margin-bottom:16px}
        select{padding:8px 12px;border:1px solid var(--border);background:var(--card);color:var(--text);font-size:13px;cursor:pointer;outline:none}
        select:focus{border-color:var(--accent)}
        .empty-state{text-align:center;padding:40px 20px;color:var(--muted);font-size:13px}
        /* Role Selector */
        .role-selector{padding:12px 16px;border-bottom:1px solid var(--border)}
        .role-selector label{font-size:11px;text-transform:uppercase;color:var(--muted);letter-spacing:.5px;display:block;margin-bottom:6px;font-weight:600}
        .role-selector select{width:100%;padding:8px 10px;font-size:12px}
        .role-info{margin-top:8px;font-size:11px;color:var(--muted);display:flex;align-items:center;gap:6px}
        .role-badge{padding:2px 6px;font-size:9px;font-weight:700;text-transform:uppercase}
        .role-badge.admin{background:rgba(63,185,80,.15);color:var(--green)}
        .role-badge.customer{background:rgba(210,153,34,.15);color:var(--yellow)}
        /* Observability */
        .metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:16px}
        .metric-card{background:var(--card);border:1px solid var(--border);padding:14px;text-align:center}
        .metric-card .metric-value{font-size:22px;font-weight:600;color:var(--accent)}
        .metric-card .metric-label{font-size:10px;color:var(--muted);margin-top:4px;text-transform:uppercase}
        .latency-chart{height:180px;background:var(--bg);border:1px solid var(--border);display:flex;align-items:flex-end;padding:16px;gap:3px
        }
        .latency-bar{flex:1;background:var(--accent);min-height:4px;position:relative}
        .latency-bar:hover::after{content:attr(data-value);position:absolute;top:-24px;left:50%;transform:translateX(-50%);font-size:10px;background:var(--bg);border:1px solid var(--border);padding:2px 6px;white-space:nowrap}
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Sidebar -->
        <nav class="sidebar">
            <div class="logo">
                <h1>◆ Atlas</h1>
                <span>Intelligent Lakehouse</span>
            </div>
            <!-- Role Selector -->
            <div class="role-selector">
                <label>Assume Role</label>
                <select id="role-select" onchange="switchRole(this.value)">
                    <option value="komodo">🔓 Komodo (Full Access)</option>
                </select>
                <div class="role-info" id="role-info">
                    <span class="role-badge admin">Admin</span>
                    <span>All datasets • All patients accessible</span>
                </div>
            </div>
            <div class="nav-section">
                <div class="nav-section-title">Overview</div>
                <div class="nav-item active" data-section="dashboard">
                    <span class="icon">📊</span>
                    Dashboard
                </div>
            </div>
            <div class="nav-section">
                <div class="nav-section-title">Discovery</div>
                <div class="nav-item" data-section="search">
                    <span class="icon">🔍</span>
                    Semantic Search
                </div>
                <div class="nav-item" data-section="similar">
                    <span class="icon">🧬</span>
                    Similar Patients
                </div>
                <div class="nav-item" data-section="sql">
                    <span class="icon">▶</span>
                    SQL Query
                </div>
                <div class="nav-item" data-section="benchmarks">
                    <span class="icon">⚡</span>
                    Benchmarks
                </div>
                <div class="nav-item" data-section="bitmap-demo">
                    <span class="icon">🗂</span>
                    Bitmap Index Demo
                </div>
                <div class="nav-item" data-section="partitions">
                    <span class="icon">📦</span>
                    Partitioning
                </div>
            </div>
            <div class="nav-section">
                <div class="nav-section-title">Data</div>
                <div class="nav-item" data-section="cohorts">
                    <span class="icon">▣</span>
                    Atlas Kohorts
                </div>
                <div class="nav-item" data-section="policies">
                    <span class="icon">🔒</span>
                    ABAC Policies
                </div>
                <div class="nav-item" data-section="universes">
                    <span class="icon">🌐</span>
                    Customer Universes
                </div>
            </div>
            <div class="nav-section">
                <div class="nav-section-title">Monitoring</div>
                <div class="nav-item" data-section="observability">
                    <span class="icon">📈</span>
                    Observability
                </div>
                <div class="nav-item" data-section="audit">
                    <span class="icon">📋</span>
                    Audit Log
                </div>
            </div>
            <div class="nav-section">
                <div class="nav-section-title">Info</div>
                <div class="nav-item" data-section="about">
                    <span class="icon">ℹ</span>
                    About
                </div>
            </div>
        </nav>

        <!-- Main Content -->
        <main class="main-content">
            <!-- Dashboard Section -->
            <section id="dashboard" class="section active">
                <div class="section-header">
                    <h2>Dashboard</h2>
                    <p>Overview of your Intelligent Virtual Lakehouse</p>
                </div>
                <div class="stats-grid" id="dashboard-stats">
                    <div class="stat-card">
                        <div class="value" id="stat-vectors">-</div>
                        <div class="label">Universes</div>
                    </div>
                    <div class="stat-card">
                        <div class="value" id="stat-cohorts">-</div>
                        <div class="label">Atlas Kohorts</div>
                    </div>
                    <div class="stat-card">
                        <div class="value" id="stat-policies">-</div>
                        <div class="label">ABAC Policies</div>
                    </div>
                    <div class="stat-card">
                        <div class="value" id="stat-largest">-</div>
                        <div class="label">Largest Cohort</div>
                    </div>
                </div>
                <div class="card">
                    <h3>Recent Activity</h3>
                    <div id="recent-activity">Loading...</div>
                </div>
            </section>

            <!-- Semantic Search Section -->
            <section id="search" class="section">
                <div class="section-header">
                    <h2>Semantic Search</h2>
                    <p>Find patients by clinical concepts using BioLORD-2023 embeddings</p>
                </div>
                <div class="card">
                    <div class="search-box">
                        <select id="search-universe">
                            <option value="all_patients">All Patients</option>
                        </select>
                        <input type="text" class="search-input" id="search-query"
                               placeholder="e.g., diabetes with heart failure, chronic kidney disease...">
                        <input type="number" class="search-input" id="search-topk"
                               value="10" min="1" max="100" style="width: 100px;">
                        <button class="btn btn-primary" onclick="runSemanticSearch()">Search</button>
                    </div>
                    <div class="loading" id="search-loading">
                        <div class="spinner"></div>
                        <div>Searching...</div>
                    </div>
                    <div id="search-results"></div>
                </div>
            </section>

            <!-- Similar Patients Section -->
            <section id="similar" class="section">
                <div class="section-header">
                    <h2>Similar Patients</h2>
                    <p>Find patients with similar clinical profiles using vector similarity</p>
                </div>
                <div class="card">
                    <div class="patient-input-group">
                        <select id="similar-universe">
                            <option value="all_patients">All Patients</option>
                        </select>
                        <input type="text" class="search-input" id="similar-patient-id"
                               placeholder="Enter Patient ID (e.g., 20491624)">
                        <input type="number" class="search-input" id="similar-topk"
                               value="10" min="1" max="100" style="width: 100px;">
                        <button class="btn btn-primary" onclick="findSimilarPatients()">Find Similar</button>
                    </div>
                    <div class="loading" id="similar-loading">
                        <div class="spinner"></div>
                        <div>Finding similar patients...</div>
                    </div>
                    <div id="similar-results"></div>
                </div>
            </section>

            <!-- SQL Query Section -->
            <section id="sql" class="section">
                <div class="section-header">
                    <h2>SQL Query Console</h2>
                    <p>Run SQL queries against cohort data — compare DataFusion vs DuckDB engines</p>
                </div>
                <div class="card">
                    <div class="sql-toolbar" style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
                        <span style="color:var(--muted);font-size:12px">Tables:</span>
                        <div id="sql-tables" style="display:flex;gap:8px;flex-wrap:wrap"></div>
                        <div style="flex:1"></div>
                        <div style="display:flex;align-items:center;gap:8px;background:var(--card);padding:6px 12px;border-radius:6px;border:1px solid var(--border)">
                            <span style="font-size:11px;color:var(--muted)">Engine:</span>
                            <label style="display:flex;align-items:center;gap:4px;cursor:pointer;font-size:12px">
                                <input type="radio" name="sql-engine" value="datafusion" checked style="accent-color:var(--accent)">
                                DataFusion
                            </label>
                            <label style="display:flex;align-items:center;gap:4px;cursor:pointer;font-size:12px">
                                <input type="radio" name="sql-engine" value="duckdb" style="accent-color:var(--green)">
                                DuckDB
                            </label>
                        </div>
                        <label style="display:flex;align-items:center;gap:4px;cursor:pointer;font-size:11px;color:var(--muted)">
                            <input type="checkbox" id="sql-use-partitions" style="accent-color:var(--accent)">
                            Use Partitions
                        </label>
                        <span class="query-shortcut" style="background:var(--card);padding:4px 8px;border-radius:4px;font-size:11px;color:var(--muted)">Cmd+Enter</span>
                        <button class="btn btn-primary" onclick="runSqlQuery()">Execute</button>
                    </div>
                    <div style="margin-bottom:12px">
                        <span style="color:var(--muted);font-size:12px;margin-right:8px">Benchmarks:</span>
                        <span class="bench-chip" style="cursor:pointer;background:var(--card);padding:4px 8px;border-radius:4px;font-size:11px;margin-right:4px" onclick="loadSqlBenchmark(0)">Full scan</span>
                        <span class="bench-chip" style="cursor:pointer;background:var(--card);padding:4px 8px;border-radius:4px;font-size:11px;margin-right:4px" onclick="loadSqlBenchmark(1)">COUNT(*)</span>
                        <span class="bench-chip" style="cursor:pointer;background:var(--card);padding:4px 8px;border-radius:4px;font-size:11px;margin-right:4px" onclick="loadSqlBenchmark(2)">GROUP BY</span>
                        <span class="bench-chip" style="cursor:pointer;background:var(--card);padding:4px 8px;border-radius:4px;font-size:11px;margin-right:4px" onclick="loadSqlBenchmark(3)">Top states</span>
                        <span class="bench-chip" style="cursor:pointer;background:var(--card);padding:4px 8px;border-radius:4px;font-size:11px;margin-right:4px;border:1px solid var(--green)" onclick="loadSqlBenchmark(4)">Point lookup</span>
                        <span class="bench-chip" style="cursor:pointer;background:var(--card);padding:4px 8px;border-radius:4px;font-size:11px;margin-right:4px;border:1px solid var(--accent)" onclick="compareEngines()">⚡ Compare Engines</span>
                    </div>
                    <textarea id="sql-editor" class="search-input" style="width:100%;min-height:120px;font-family:monospace;font-size:13px;resize:vertical" placeholder="SELECT * FROM ... LIMIT 100" spellcheck="false"></textarea>
                    <div id="sql-results" style="margin-top:16px">
                        <div class="empty-state" style="color:var(--muted)">Run a query to see results</div>
                    </div>
                </div>
            </section>

            <!-- Benchmarks Section -->
            <section id="benchmarks" class="section">
                <div class="section-header">
                    <h2>Performance Benchmarks</h2>
                    <p>The .kohort Stack: Query Engines → Codecs → Indexes</p>
                </div>

                <!-- Stack Overview -->
                <div class="card" style="margin-bottom:20px;font-family:monospace;font-size:13px;line-height:1.6">
                    <div style="border:2px solid var(--border);border-radius:8px;overflow:hidden">
                        <div style="background:var(--accent);color:#fff;padding:12px 16px;border-bottom:2px solid var(--border)">
                            <strong>QUERY ENGINES</strong> (SQL execution)
                            <div style="display:flex;gap:24px;margin-top:8px;font-size:12px;opacity:0.9">
                                <span>🦅 DataFusion — Native .kohort, bitmap pushdown</span>
                                <span>🦆 DuckDB — Full SQL, window functions</span>
                                <span>🐻‍❄️ Polars — DataFrame ops, streaming</span>
                            </div>
                        </div>
                        <div style="background:var(--green);color:#fff;padding:12px 16px;border-bottom:2px solid var(--border)">
                            <strong>CODECS</strong> (data storage format)
                            <div style="display:flex;gap:24px;margin-top:8px;font-size:12px;opacity:0.9">
                                <span>Parquet — Best compression (default)</span>
                                <span>Arrow IPC — Fastest I/O</span>
                                <span>Feather — Zero-copy reads</span>
                                <span style="opacity:0.6">Vortex — Coming soon</span>
                            </div>
                        </div>
                        <div style="background:var(--purple,#a371f7);color:#fff;padding:12px 16px">
                            <strong>INDEXES</strong> (fast filtering) — Tag 200
                            <div style="display:flex;gap:24px;margin-top:8px;font-size:12px;opacity:0.9">
                                <span>Roaring Bitmaps — 10-100x faster point queries</span>
                                <span>Source bitsets — Fast row filtering via AND operations</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Codec Benchmark -->
                <div class="card" style="margin-bottom:20px">
                    <h3 style="margin-bottom:16px">📦 Codec Comparison</h3>
                    <p style="color:var(--muted);font-size:13px;margin-bottom:16px">
                        Compare write time, read time, file size, and SQL performance across formats.
                        The .kohort format uses <strong>Parquet</strong> for data storage with <strong>Roaring bitmaps</strong> for indexes.
                    </p>
                    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px">
                        <div style="background:var(--bg);padding:16px;border-radius:8px;text-align:center">
                            <div style="font-size:11px;color:var(--muted);margin-bottom:4px">PARQUET</div>
                            <div style="font-size:18px;font-weight:600;color:var(--accent)">Baseline</div>
                            <div style="font-size:11px;color:var(--muted)">Best compression</div>
                        </div>
                        <div style="background:var(--bg);padding:16px;border-radius:8px;text-align:center">
                            <div style="font-size:11px;color:var(--muted);margin-bottom:4px">ARROW IPC</div>
                            <div style="font-size:18px;font-weight:600;color:var(--yellow)">2-5x faster read</div>
                            <div style="font-size:11px;color:var(--muted)">Larger files</div>
                        </div>
                        <div style="background:var(--bg);padding:16px;border-radius:8px;text-align:center">
                            <div style="font-size:11px;color:var(--muted);margin-bottom:4px">FEATHER</div>
                            <div style="font-size:18px;font-weight:600;color:var(--green)">Fastest I/O</div>
                            <div style="font-size:11px;color:var(--muted)">Zero-copy reads</div>
                        </div>
                        <div style="background:var(--bg);padding:16px;border-radius:8px;text-align:center;border:2px dashed var(--muted);opacity:0.6">
                            <div style="font-size:11px;color:var(--muted);margin-bottom:4px">VORTEX</div>
                            <div style="font-size:18px;font-weight:600;color:var(--muted)">Coming Soon</div>
                            <div style="font-size:11px;color:var(--muted)">Adaptive encoding</div>
                        </div>
                    </div>
                    <button class="btn btn-primary" onclick="runCodecBenchmark()" id="bench-codec-btn">Run Codec Benchmark</button>
                    <div id="codec-benchmark-results" style="margin-top:20px"></div>
                </div>
                <div class="card">
                    <h3 style="margin-bottom:16px">🦅🦆🐻‍❄️ Query Engine Comparison</h3>
                    <p style="color:var(--muted);font-size:13px;margin-bottom:16px">
                        Run the same query against all three engines to compare performance.
                        Each engine has different strengths for .kohort file queries.
                    </p>
                    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:16px">
                        <div style="background:var(--bg);padding:16px;border-radius:8px">
                            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
                                <span style="font-size:18px">🦅</span>
                                <strong>DataFusion</strong>
                            </div>
                            <ul style="margin:0;padding-left:16px;font-size:12px;color:var(--text);line-height:1.6">
                                <li>Native .kohort integration</li>
                                <li>Bitmap index pushdown</li>
                                <li>Zero-copy Arrow FFI</li>
                            </ul>
                        </div>
                        <div style="background:var(--bg);padding:16px;border-radius:8px">
                            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
                                <span style="font-size:18px">🦆</span>
                                <strong>DuckDB</strong>
                            </div>
                            <ul style="margin:0;padding-left:16px;font-size:12px;color:var(--text);line-height:1.6">
                                <li>Full SQL dialect</li>
                                <li>Window functions</li>
                                <li>In-process OLAP</li>
                            </ul>
                        </div>
                        <div style="background:var(--bg);padding:16px;border-radius:8px">
                            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
                                <span style="font-size:18px">🐻‍❄️</span>
                                <strong>Polars</strong>
                            </div>
                            <ul style="margin:0;padding-left:16px;font-size:12px;color:var(--text);line-height:1.6">
                                <li>Streaming LazyFrame</li>
                                <li>Multi-threaded Rust</li>
                                <li>Expression API</li>
                            </ul>
                        </div>
                    </div>
                    <button class="btn btn-primary" onclick="runEngineComparison()" id="engine-compare-btn">Run Engine Comparison</button>
                    <div id="engine-comparison-results" style="margin-top:20px"></div>
                </div>
            </section>

            <!-- Bitmap Index Demo Section -->
            <section id="bitmap-demo" class="section">
                <div class="section-header">
                    <h2>Bitmap Index Demo</h2>
                    <p>See how Roaring bitmap indexes accelerate point queries by 10-100x</p>
                </div>
                <div class="card" style="margin-bottom:20px">
                    <h3 style="margin-bottom:16px">How Bitmap Pruning Works</h3>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-bottom:20px">
                        <div>
                            <div style="font-size:13px;font-weight:600;color:var(--muted);margin-bottom:8px">WITHOUT BITMAP INDEX</div>
                            <div style="background:var(--bg);padding:16px;border-radius:8px;font-family:monospace;font-size:12px">
                                <div style="color:var(--red)">❌ Scan ALL files to find patient_id = 12345</div>
                                <div style="margin-top:8px;color:var(--muted)">
                                    part_0001.cohort → scan 50K rows<br>
                                    part_0002.cohort → scan 50K rows<br>
                                    part_0003.cohort → scan 50K rows<br>
                                    ...<br>
                                    part_0010.cohort → scan 50K rows
                                </div>
                                <div style="margin-top:8px;color:var(--red)">Total: 500K rows scanned</div>
                            </div>
                        </div>
                        <div>
                            <div style="font-size:13px;font-weight:600;color:var(--green);margin-bottom:8px">WITH BITMAP INDEX</div>
                            <div style="background:var(--bg);padding:16px;border-radius:8px;font-family:monospace;font-size:12px">
                                <div style="color:var(--green)">✓ Check bitmap → patient in part_0007</div>
                                <div style="margin-top:8px;color:var(--muted)">
                                    part_0001.cohort → <span style="color:var(--green)">SKIP</span><br>
                                    part_0002.cohort → <span style="color:var(--green)">SKIP</span><br>
                                    ...<br>
                                    part_0007.cohort → scan 50K rows ✓<br>
                                    part_0008.cohort → <span style="color:var(--green)">SKIP</span><br>
                                    ...
                                </div>
                                <div style="margin-top:8px;color:var(--green)">Total: 50K rows scanned (10x faster!)</div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="card">
                    <h3 style="margin-bottom:16px">Try It: Point Query Benchmark</h3>
                    <p style="color:var(--muted);font-size:13px;margin-bottom:16px">
                        Run a WHERE patient_id = X query with and without bitmap pruning to see the speedup.
                    </p>
                    <div style="display:flex;gap:12px;margin-bottom:16px">
                        <input type="text" id="bitmap-patient-id" class="search-input" placeholder="Enter Patient ID (e.g., 20491624)" style="flex:1">
                        <button class="btn btn-primary" onclick="runBitmapDemo()">Run Benchmark</button>
                    </div>
                    <div id="bitmap-demo-results"></div>
                </div>
                <div class="card" style="margin-top:20px">
                    <h3 style="margin-bottom:16px">Bitmap Index Internals</h3>
                    <p style="color:var(--muted);font-size:13px;margin-bottom:16px">
                        Each .kohort file contains a <strong>Roaring bitmap</strong> (Tag 200) that stores patient IDs as compressed bit sets.
                        This enables sub-millisecond membership checks: "Is patient X in this cohort?"
                    </p>
                    <div style="background:var(--bg);padding:20px;border-radius:8px;font-family:monospace;font-size:12px;line-height:1.8">
                        <span style="color:var(--muted)"># Check membership in O(1)</span><br>
                        <span style="color:var(--accent)">bitmap</span>.contains(<span style="color:var(--yellow)">patient_id</span>) → <span style="color:var(--green)">true</span>/<span style="color:var(--red)">false</span><br><br>
                        <span style="color:var(--muted)"># Set operations on patient universes</span><br>
                        <span style="color:var(--accent)">cohort_a</span> ∪ <span style="color:var(--accent)">cohort_b</span>  <span style="color:var(--muted)"># union</span><br>
                        <span style="color:var(--accent)">cohort_a</span> ∩ <span style="color:var(--accent)">cohort_b</span>  <span style="color:var(--muted)"># intersect</span><br>
                        <span style="color:var(--accent)">cohort_a</span> − <span style="color:var(--accent)">cohort_b</span>  <span style="color:var(--muted)"># difference</span>
                    </div>
                </div>
            </section>

            <!-- Partitioning Section -->
            <section id="partitions" class="section">
                <div class="section-header">
                    <h2>Cohort Partitioning</h2>
                    <p>Split large .kohort files into smaller partitions for parallel query execution</p>
                </div>
                <div class="card" style="margin-bottom:20px">
                    <h3 style="margin-bottom:16px">Partition Status</h3>
                    <div id="partition-status">
                        <div class="loading active"><div class="spinner"></div><div>Checking partitions...</div></div>
                    </div>
                </div>
                <div class="card">
                    <h3 style="margin-bottom:16px">Create Partitions</h3>
                    <p style="color:var(--muted);font-size:13px;margin-bottom:16px">
                        Partition a large cohort file to enable parallel query execution across multiple workers.
                    </p>
                    <div style="display:flex;gap:12px;align-items:flex-end;margin-bottom:16px">
                        <div style="flex:1">
                            <label style="display:block;font-size:12px;color:var(--muted);margin-bottom:4px">Source File</label>
                            <select id="partition-source" style="width:100%;padding:8px 12px;background:var(--card);border:1px solid var(--border);color:var(--text);font-size:13px">
                                <option value="">Select a .kohort file</option>
                            </select>
                        </div>
                        <div>
                            <label style="display:block;font-size:12px;color:var(--muted);margin-bottom:4px">Target Rows/Partition</label>
                            <input type="number" id="partition-target-rows" class="search-input" value="50000" min="1000" style="width:120px">
                        </div>
                        <button class="btn btn-primary" onclick="createPartitions()">Partition</button>
                    </div>
                    <div id="partition-create-results"></div>
                </div>
            </section>

            <!-- Cohorts Section -->
            <section id="cohorts" class="section">
                <div class="section-header">
                    <h2>Atlas Kohorts</h2>
                    <p>Unified cohort files with embedded Parquet data, Roaring bitmaps, and Lance vector pointers</p>
                </div>
                <div id="cohort-list">
                    <div class="loading active">
                        <div class="spinner"></div>
                        <div>Loading cohorts...</div>
                    </div>
                </div>
            </section>

            <!-- ABAC Policies Section -->
            <section id="policies" class="section">
                <div class="section-header">
                    <h2>ABAC Policies</h2>
                    <p>Customer universe access policies with kohort-based filtering</p>
                </div>
                <div id="policy-list">
                    <div class="loading active">
                        <div class="spinner"></div>
                        <div>Loading policies...</div>
                    </div>
                </div>
            </section>

            <!-- Customer Universes Section -->
            <section id="universes" class="section">
                <div class="section-header">
                    <h2>Customer Universes</h2>
                    <p>Computed patient sets for each customer (union/intersection of kohorts)</p>
                </div>
                <div id="universe-list">
                    <div class="loading active">
                        <div class="spinner"></div>
                        <div>Computing universes...</div>
                    </div>
                </div>
            </section>

            <!-- Observability Section -->
            <section id="observability" class="section">
                <div class="section-header">
                    <h2>Observability</h2>
                    <p>Query performance metrics and system health</p>
                </div>
                <div class="metric-grid" id="obs-metrics">
                    <div class="metric-card">
                        <div class="metric-value" id="obs-total-queries">-</div>
                        <div class="metric-label">Total Queries</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="obs-avg-latency">-</div>
                        <div class="metric-label">Avg Latency</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="obs-p95-latency">-</div>
                        <div class="metric-label">P95 Latency</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="obs-success-rate">-</div>
                        <div class="metric-label">Success Rate</div>
                    </div>
                </div>
                <div class="card">
                    <h3>Query Latency (Last 20 Queries)</h3>
                    <div class="latency-chart" id="latency-chart">
                        <div class="empty-state">No query data yet</div>
                    </div>
                </div>
                <div class="card">
                    <h3>Query Breakdown by Type</h3>
                    <div id="query-breakdown">
                        <div class="loading active">
                            <div class="spinner"></div>
                            <div>Loading metrics...</div>
                        </div>
                    </div>
                </div>
            </section>

            <!-- Audit Log Section -->
            <section id="audit" class="section">
                <div class="section-header">
                    <h2>Audit Log</h2>
                    <p>Query history and access patterns</p>
                </div>
                <div class="card">
                    <button class="btn btn-secondary" onclick="loadAuditLog()" style="margin-bottom: 20px;">
                        Refresh
                    </button>
                    <div id="audit-log">
                        <div class="loading active">
                            <div class="spinner"></div>
                            <div>Loading audit log...</div>
                        </div>
                    </div>
                </div>
            </section>

            <!-- About Section -->
            <section id="about" class="section" style="max-width:900px;margin:0 auto;padding:0 24px">
                <div class="section-header" style="margin-bottom:40px">
                    <h2 style="font-size:32px">Atlas Intelligent Virtual Lakehouse</h2>
                    <p style="font-size:18px;margin-top:8px">Architecture Overview &amp; Technical Details</p>
                </div>

                <!-- Overview Card -->
                <div class="card" style="margin-bottom:32px;padding:28px">
                    <h3 style="margin-bottom:20px;color:var(--accent);font-size:22px">What is Atlas?</h3>
                    <p style="line-height:1.8;color:var(--text);font-size:16px">
                        Atlas is an <strong>Intelligent Virtual Lakehouse</strong> for healthcare patient data that combines:
                    </p>
                    <ul style="margin:20px 0 20px 28px;line-height:2.2;color:var(--text);font-size:16px">
                        <li><strong>Self-contained .kohort files</strong> - TIFF-inspired binary containers with embedded Parquet data, Roaring bitmaps, and vector pointers</li>
                        <li><strong>BioLORD-2023 embeddings</strong> - Clinical semantic search using state-of-the-art biomedical language models</li>
                        <li><strong>ABAC Policies</strong> - Attribute-Based Access Control with source bitset filtering for O(1) row-level security</li>
                        <li><strong>Customer Universes</strong> - Multi-kohort access with union/intersection operations on bitmap memberships</li>
                    </ul>
                </div>

                <!-- Why .kohort? Section -->
                <div class="card" style="margin-bottom:32px;padding:28px">
                    <h3 style="margin-bottom:20px;color:var(--green);font-size:22px">Why .kohort?</h3>
                    <p style="line-height:1.8;color:var(--text);font-size:16px;margin-bottom:20px">
                        The <strong>.kohort file format</strong> is a self-contained binary container inspired by TIFF's tag-based architecture. It solves the "data lake sprawl" problem by packaging everything needed for patient cohort analysis into a single portable file.
                    </p>

                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-bottom:24px">
                        <div style="background:var(--bg);border-radius:8px;padding:20px">
                            <div style="font-size:14px;color:var(--muted);margin-bottom:8px;text-transform:uppercase">Original Format</div>
                            <ul style="margin:0;padding-left:20px;line-height:2;font-size:15px;color:var(--text)">
                                <li>Parquet data modalities</li>
                                <li>JSON tag directory</li>
                                <li>Cohort definition</li>
                            </ul>
                        </div>
                        <div style="background:var(--bg);border-radius:8px;padding:20px;border:2px solid var(--accent)">
                            <div style="font-size:14px;color:var(--accent);margin-bottom:8px;text-transform:uppercase">Atlas Enhanced</div>
                            <ul style="margin:0;padding-left:20px;line-height:2;font-size:15px;color:var(--text)">
                                <li>Parquet data modalities</li>
                                <li>JSON tag directory</li>
                                <li>Cohort definition</li>
                                <li style="color:var(--green)"><strong>+ Roaring bitmap indexes</strong></li>
                                <li style="color:var(--purple,#a371f7)"><strong>+ Lance vector pointers</strong></li>
                            </ul>
                        </div>
                    </div>

                    <div style="font-size:16px;line-height:1.8;color:var(--text)">
                        <p style="margin-bottom:16px"><strong>Key Benefits:</strong></p>
                        <ul style="margin:0 0 0 28px;line-height:2.2">
                            <li><strong>Single File Portability</strong> - Move entire cohorts with one file, no broken references</li>
                            <li><strong>Sub-millisecond Membership Checks</strong> - Roaring bitmaps enable O(1) "is patient in cohort?" queries</li>
                            <li><strong>Semantic Patient Search</strong> - Lance vector pointers enable "find similar patients" without copying embeddings</li>
                            <li><strong>Versioned Schema Evolution</strong> - Tag-based format allows adding new modalities without breaking readers</li>
                            <li><strong>Zero External Dependencies</strong> - No database connections, no cloud credentials, just the file</li>
                        </ul>
                    </div>
                </div>

                <!-- Architecture Sections - Single Column for Scroll -->
                <div style="display:flex;flex-direction:column;gap:32px;margin-bottom:32px">

                    <!-- .kohort File Format -->
                    <div class="card" style="padding:28px">
                        <h3 style="margin-bottom:20px;color:var(--green);font-size:22px">.kohort Binary Structure</h3>
                        <p style="margin-bottom:16px;color:var(--muted);font-size:16px">TIFF-inspired single binary container:</p>
                        <div style="background:var(--bg);border-radius:8px;padding:20px;font-family:monospace;font-size:14px;line-height:1.8">
                            <div style="color:var(--yellow)">KOHORT | Ver(3) | Dir Offset (8B)</div>
                            <div style="border-left:3px solid var(--border);margin:12px 0;padding-left:16px">
                                <div style="color:var(--accent)">Tag 101: Claims-v1 (Parquet)</div>
                                <div style="color:var(--accent)">Tag 102: Rx-v1 (Parquet)</div>
                                <div style="color:var(--green)">Tag 200: PatientMembership (Roaring)</div>
                                <div style="color:var(--purple,#a371f7)">Tag 300: VectorIndex (Lance ptr)</div>
                            </div>
                            <div style="color:var(--muted)">Tag Directory (JSON) - offsets + cohort def</div>
                        </div>
                        <div style="margin-top:20px;font-size:15px">
                            <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border)">
                                <span style="color:var(--muted)">Tags 100-199</span>
                                <span>Data Modalities (Parquet)</span>
                            </div>
                            <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border)">
                                <span style="color:var(--muted)">Tags 200-299</span>
                                <span>Indexes (Roaring bitmaps)</span>
                            </div>
                            <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border)">
                                <span style="color:var(--muted)">Tags 300-399</span>
                                <span>External Refs (Lance)</span>
                            </div>
                            <div style="display:flex;justify-content:space-between;padding:8px 0">
                                <span style="color:var(--muted)">Tags 400-499</span>
                                <span>Metadata (JSON)</span>
                            </div>
                        </div>
                    </div>

                    <!-- Embedding Model -->
                    <div class="card" style="padding:28px">
                        <h3 style="margin-bottom:20px;color:var(--accent);font-size:22px">Semantic Search Engine</h3>
                        <div style="background:var(--bg);border-radius:8px;padding:20px;margin-bottom:20px">
                            <div style="font-size:24px;font-weight:600;margin-bottom:8px">BioLORD-2023-C</div>
                            <div style="color:var(--muted);font-size:14px;margin-bottom:16px">FremyCompany/BioLORD-2023-C</div>
                            <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:16px;font-size:15px">
                                <div>
                                    <div style="color:var(--muted);font-size:12px;text-transform:uppercase">Dimensions</div>
                                    <div style="font-weight:600;font-size:28px;color:var(--accent)">768</div>
                                </div>
                                <div>
                                    <div style="color:var(--muted);font-size:12px;text-transform:uppercase">Vector Format</div>
                                    <div style="font-weight:600;font-size:18px">Lance</div>
                                </div>
                                <div>
                                    <div style="color:var(--muted);font-size:12px;text-transform:uppercase">Search</div>
                                    <div style="font-weight:600;font-size:18px">Cosine Similarity</div>
                                </div>
                                <div>
                                    <div style="color:var(--muted);font-size:12px;text-transform:uppercase">Domain</div>
                                    <div style="font-weight:600;font-size:18px">Clinical/Biomedical</div>
                                </div>
                            </div>
                        </div>
                        <p style="font-size:15px;color:var(--muted);line-height:1.8">
                            BioLORD learns representations of biomedical concepts using definitions and relational knowledge from medical ontologies (SNOMED-CT, ICD-10, MeSH). Each patient's clinical journey is encoded as a 768-dimensional vector for semantic similarity search.
                        </p>
                    </div>

                    <!-- ABAC Policies -->
                    <div class="card" style="padding:28px">
                        <h3 style="margin-bottom:20px;color:var(--yellow);font-size:22px">ABAC Policies</h3>
                        <p style="margin-bottom:16px;color:var(--muted);font-size:16px">Attribute-Based Access Control with source bitsets:</p>
                        <div style="background:var(--bg);border-radius:8px;padding:20px;font-family:monospace;font-size:14px;line-height:2">
                            <div><span style="color:var(--muted)">// Row-level filter (O(1))</span></div>
                            <div><span style="color:var(--accent)">row</span>.SOURCES_BITSET <span style="color:var(--yellow)">&amp;</span> <span style="color:var(--green)">policy</span>.source_bitset <span style="color:var(--yellow)">!=</span> <span style="color:var(--red)">0</span></div>
                        </div>
                        <div style="margin-top:20px;font-size:15px">
                            <div style="font-weight:600;margin-bottom:12px">Source Bitset Key:</div>
                            <div style="display:flex;flex-wrap:wrap;gap:8px">
                                <span class="policy-tag" style="padding:6px 12px;font-size:14px">kwazii=1</span>
                                <span class="policy-tag" style="padding:6px 12px;font-size:14px">carillon=2</span>
                                <span class="policy-tag" style="padding:6px 12px;font-size:14px">bellweather=4</span>
                                <span class="policy-tag" style="padding:6px 12px;font-size:14px">lhcg=8</span>
                                <span class="policy-tag" style="padding:6px 12px;font-size:14px">lhcg_claims=16</span>
                                <span class="policy-tag" style="padding:6px 12px;font-size:14px">rowley=32</span>
                                <span class="policy-tag" style="padding:6px 12px;font-size:14px">nimbus=64</span>
                            </div>
                        </div>
                        <div style="margin-top:20px;font-size:15px;color:var(--muted);line-height:1.8">
                            Fast bitwise AND operations enable screaming-fast row filtering. Each customer policy precomputes a source_bitset from their allowed sources list.
                        </div>
                    </div>

                    <!-- Customer Universes -->
                    <div class="card" style="padding:28px">
                        <h3 style="margin-bottom:20px;color:var(--purple,#a371f7);font-size:22px">Customer Universes</h3>
                        <p style="margin-bottom:16px;color:var(--muted);font-size:16px">Multi-kohort membership operations:</p>
                        <div style="background:var(--bg);border-radius:8px;padding:20px;font-family:monospace;font-size:14px;line-height:2">
                            <div style="color:var(--muted)">// Union of patient memberships</div>
                            <div><span style="color:var(--accent)">Universe</span> = Kohort_A.<span style="color:var(--green)">membership</span></div>
                            <div style="padding-left:48px"><span style="color:var(--yellow)">|</span> Kohort_B.<span style="color:var(--green)">membership</span></div>
                            <div style="padding-left:48px"><span style="color:var(--yellow)">|</span> Kohort_C.<span style="color:var(--green)">membership</span></div>
                        </div>
                        <div style="margin-top:20px;font-size:15px">
                            <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border)">
                                <span>Union</span>
                                <span style="color:var(--green)">A | B - Combined access</span>
                            </div>
                            <div style="display:flex;justify-content:space-between;padding:8px 0">
                                <span>Intersect</span>
                                <span style="color:var(--accent)">A & B - Overlap only</span>
                            </div>
                        </div>
                        <div style="margin-top:20px;font-size:15px;color:var(--muted);line-height:1.8">
                            Roaring bitmaps enable sub-millisecond set operations on millions of patient IDs. Customer universe is computed lazily and cached.
                        </div>
                    </div>
                </div>

                <!-- Technology Stack -->
                <div class="card" style="padding:28px">
                    <h3 style="margin-bottom:24px;font-size:22px">Technology Stack</h3>
                    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:20px;text-align:center">
                        <div style="background:var(--bg);border-radius:8px;padding:20px">
                            <div style="font-weight:600;font-size:18px;color:var(--accent)">BioLORD-2023-C</div>
                            <div style="font-size:14px;color:var(--muted);margin-top:6px">Clinical embeddings</div>
                        </div>
                        <div style="background:var(--bg);border-radius:8px;padding:20px">
                            <div style="font-weight:600;font-size:18px;color:var(--accent)">Python</div>
                            <div style="font-size:14px;color:var(--muted);margin-top:6px">FastAPI + PyArrow</div>
                        </div>
                        <div style="background:var(--bg);border-radius:8px;padding:20px">
                            <div style="font-weight:600;font-size:18px;color:var(--accent)">Rust</div>
                            <div style="font-size:14px;color:var(--muted);margin-top:6px">High-performance core</div>
                        </div>
                        <div style="background:var(--bg);border-radius:8px;padding:20px">
                            <div style="font-weight:600;font-size:18px;color:var(--accent)">Roaring</div>
                            <div style="font-size:14px;color:var(--muted);margin-top:6px">Compressed bitmaps</div>
                        </div>
                        <div style="background:var(--bg);border-radius:8px;padding:20px">
                            <div style="font-weight:600;font-size:18px;color:var(--accent)">Lance</div>
                            <div style="font-size:14px;color:var(--muted);margin-top:6px">Vector storage</div>
                        </div>
                        <div style="background:var(--bg);border-radius:8px;padding:20px">
                            <div style="font-weight:600;font-size:18px;color:var(--accent)">Parquet</div>
                            <div style="font-size:14px;color:var(--muted);margin-top:6px">Columnar data</div>
                        </div>
                        <div style="background:var(--bg);border-radius:8px;padding:20px">
                            <div style="font-weight:600;font-size:18px;color:var(--accent)">HuggingFace</div>
                            <div style="font-size:14px;color:var(--muted);margin-top:6px">Transformers</div>
                        </div>
                        <div style="background:var(--bg);border-radius:8px;padding:20px">
                            <div style="font-weight:600;font-size:18px;color:var(--accent)">DataFusion</div>
                            <div style="font-size:14px;color:var(--muted);margin-top:6px">SQL engine</div>
                        </div>
                    </div>
                </div>
            </section>
        </main>
    </div>

    <script>
        // Navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', () => {
                const section = item.dataset.section;

                // Update nav
                document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
                item.classList.add('active');

                // Update sections
                document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
                document.getElementById(section).classList.add('active');

                // Load data if needed
                if (section === 'cohorts') loadCohorts();
                if (section === 'policies') loadPolicies();
                if (section === 'universes') loadUniverses();
                if (section === 'sql') loadSqlTables();
                if (section === 'observability') loadObservability();
                if (section === 'audit') loadAuditLog();
                if (section === 'partitions') loadPartitionStatus();
                if (section === 'benchmarks') loadBenchmarkFiles();
            });
        });

        // Current role state
        let currentRole = 'komodo';
        let allPolicies = [];

        // Load dashboard on start
        loadDashboard();
        loadUniverseDropdowns();
        loadRoleSelector();

        async function loadDashboard() {
            try {
                // Load stats in parallel - use correct endpoints for Atlas Kohorts
                const [universesRes, kohortsRes, policiesRes] = await Promise.all([
                    fetch('/api/lakehouse/universes'),
                    fetch('/api/lakehouse/kohorts'),
                    fetch('/api/lakehouse/policies')
                ]);

                const universesData = await universesRes.json();
                const kohortsData = await kohortsRes.json();
                const policiesData = await policiesRes.json();

                // Get real counts from API
                const universes = universesData.universes || [];
                const kohorts = kohortsData.kohorts || [];
                const policies = policiesData.policies || [];

                // Role-specific stats
                if (currentRole === 'komodo') {
                    // Admin sees everything
                    const largestKohort = kohorts.reduce((max, k) => (k.patient_count || 0) > (max.patient_count || 0) ? k : max, {patient_count: 0});
                    document.getElementById('stat-vectors').textContent = universes.length.toLocaleString();
                    document.getElementById('stat-cohorts').textContent = kohorts.length.toLocaleString();
                    document.getElementById('stat-policies').textContent = policies.length.toLocaleString();
                    document.getElementById('stat-largest').textContent = largestKohort.patient_count?.toLocaleString() || '0';
                } else {
                    // Customer sees their policy stats
                    const policy = allPolicies.find(p => (p.id || p.policy_id) === currentRole);
                    const allowedKohorts = (policy?.kohort_files || []).length;
                    const allowedUniverses = policy?.universe_ids || [];
                    document.getElementById('stat-vectors').textContent = allowedUniverses.length || '1';
                    document.getElementById('stat-cohorts').textContent = allowedKohorts || '0';
                    document.getElementById('stat-policies').textContent = '1';
                    document.getElementById('stat-largest').textContent = (policy?.patient_count || 0).toLocaleString();
                }

                // Load recent audit
                const auditRes = await fetch('/api/lakehouse/audit?limit=5');
                const audit = await auditRes.json();

                if (audit.entries?.length) {
                    document.getElementById('recent-activity').innerHTML = audit.entries.map(e => `
                        <div class="audit-entry">
                            <span class="audit-time">${new Date(e.timestamp).toLocaleTimeString()}</span>
                            <span class="audit-type">${e.query_type}</span>
                            <span style="flex:1">${e.rows_returned} rows in ${e.elapsed_ms?.toFixed(0)}ms</span>
                        </div>
                    `).join('');
                } else {
                    document.getElementById('recent-activity').innerHTML = '<div class="empty-state">No recent activity</div>';
                }
            } catch (err) {
                console.error('Dashboard error:', err);
            }
        }

        async function loadUniverseDropdowns() {
            try {
                const res = await fetch('/api/lakehouse/status');
                const data = await res.json();
                // Handle both array and object formats
                const universes = Array.isArray(data.universes) ? data.universes : Object.keys(data.universes || {});

                const options = universes.map(u => `<option value="${u}">${u}</option>`).join('');
                document.getElementById('search-universe').innerHTML = options || '<option value="">No universes</option>';
                document.getElementById('similar-universe').innerHTML = options || '<option value="">No universes</option>';
            } catch (err) {
                console.error('Error loading universes:', err);
            }
        }

        async function runSemanticSearch() {
            const universe = document.getElementById('search-universe').value;
            const query = document.getElementById('search-query').value;
            const topK = parseInt(document.getElementById('search-topk').value) || 10;

            if (!query.trim()) {
                alert('Please enter a search query');
                return;
            }

            document.getElementById('search-loading').classList.add('active');
            document.getElementById('search-results').innerHTML = '';

            try {
                const res = await fetch('/api/lakehouse/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        universe_id: universe,
                        query_text: query,
                        top_k: topK
                    })
                });

                const data = await res.json();

                if (data.error) {
                    document.getElementById('search-results').innerHTML = `
                        <div class="empty-state" style="color: #ef4444;">
                            Error: ${data.error}
                        </div>`;
                    return;
                }

                const results = data.results || [];
                let html = `
                    <div class="timing-info">
                        <span>🔍 Vector: ${(data.vector_ms || 0).toFixed(1)}ms</span>
                        <span>🧊 Iceberg: ${(data.iceberg_ms || 0).toFixed(1)}ms</span>
                        <span>⏱️ Total: ${(data.elapsed_ms || 0).toFixed(1)}ms</span>
                        <span>📊 ${results.length} results</span>
                    </div>
                `;

                if (results.length === 0) {
                    html += '<div class="empty-state">No results found</div>';
                } else {
                    html += `
                        <table class="results-table">
                            <thead>
                                <tr>
                                    <th>Patient ID</th>
                                    <th>State</th>
                                    <th>Sex</th>
                                    <th>YOB</th>
                                    <th>Diagnosis Codes</th>
                                    <th>Score</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                    `;

                    results.forEach(r => {
                        // Handle both API formats (lowercase and uppercase field names)
                        const patientId = r.patient_id || r.PATIENT_NUMBER;
                        const state = r.patient_state || r.PATIENT_STATE || '-';
                        const sex = r.patient_sex || r.PATIENT_SEX || '-';
                        const yob = r.patient_yob || r.PATIENT_YOB_DATE_UNCERTIFIED_TOP_CODED || '-';
                        const diagCodes = (r.diagnosis_codes || r.DIAGNOSIS_CODES || '').substring(0, 50) + '...';
                        const score = Math.abs(r.score || r.similarity || 0);
                        const scoreClass = score > 0.7 ? 'score-high' : score > 0.5 ? 'score-med' : 'score-low';
                        html += `
                            <tr>
                                <td><strong>${patientId}</strong></td>
                                <td>${state}</td>
                                <td>${sex}</td>
                                <td>${yob}</td>
                                <td title="${r.diagnosis_codes || r.DIAGNOSIS_CODES || ''}">${diagCodes}</td>
                                <td><span class="score-badge ${scoreClass}">${(score * 100).toFixed(1)}%</span></td>
                                <td>
                                    <button class="btn btn-secondary" style="padding:6px 12px;font-size:12px"
                                            onclick="findSimilarFromSearch(${patientId})">
                                        Find Similar
                                    </button>
                                </td>
                            </tr>
                        `;
                    });

                    html += '</tbody></table>';
                }

                document.getElementById('search-results').innerHTML = html;
            } catch (err) {
                document.getElementById('search-results').innerHTML = `
                    <div class="empty-state" style="color: #ef4444;">
                        Error: ${err.message}
                    </div>`;
            } finally {
                document.getElementById('search-loading').classList.remove('active');
            }
        }

        function findSimilarFromSearch(patientId) {
            document.getElementById('similar-patient-id').value = patientId;
            document.querySelector('[data-section="similar"]').click();
            findSimilarPatients();
        }

        async function findSimilarPatients() {
            const universe = document.getElementById('similar-universe').value;
            const patientId = document.getElementById('similar-patient-id').value;
            const topK = parseInt(document.getElementById('similar-topk').value) || 10;

            if (!patientId.trim()) {
                alert('Please enter a patient ID');
                return;
            }

            document.getElementById('similar-loading').classList.add('active');
            document.getElementById('similar-results').innerHTML = '';

            try {
                const res = await fetch('/api/lakehouse/similar', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        universe_id: universe,
                        patient_id: parseInt(patientId),
                        top_k: topK
                    })
                });

                const data = await res.json();

                if (data.error) {
                    document.getElementById('similar-results').innerHTML = `
                        <div class="empty-state" style="color: #ef4444;">
                            Error: ${data.error}
                        </div>`;
                    return;
                }

                const results = data.results || data.similar_patients || [];
                let html = `
                    <div class="timing-info">
                        <span>🔍 Vector: ${(data.vector_ms || 0).toFixed(1)}ms</span>
                        <span>🧊 Iceberg: ${(data.iceberg_ms || 0).toFixed(1)}ms</span>
                        <span>⏱️ Total: ${(data.elapsed_ms || 0).toFixed(1)}ms</span>
                        <span>📊 ${results.length} similar patients</span>
                    </div>
                    <p style="margin: 16px 0; color: #888;">Reference patient: <strong>${patientId}</strong></p>
                `;

                if (results.length === 0) {
                    html += '<div class="empty-state">No similar patients found</div>';
                } else {
                    html += `
                        <table class="results-table">
                            <thead>
                                <tr>
                                    <th>Patient ID</th>
                                    <th>State</th>
                                    <th>Sex</th>
                                    <th>YOB</th>
                                    <th>Diagnosis Codes</th>
                                    <th>Similarity</th>
                                </tr>
                            </thead>
                            <tbody>
                    `;

                    results.forEach(r => {
                        // Handle both API formats (lowercase and uppercase field names)
                        const patientId = r.patient_id || r.PATIENT_NUMBER;
                        const state = r.patient_state || r.PATIENT_STATE || '-';
                        const sex = r.patient_sex || r.PATIENT_SEX || '-';
                        const yob = r.patient_yob || r.PATIENT_YOB_DATE_UNCERTIFIED_TOP_CODED || '-';
                        const diagCodes = (r.diagnosis_codes || r.DIAGNOSIS_CODES || '').substring(0, 50) + '...';
                        const score = Math.abs(r.score || r.similarity || 0);
                        const scoreClass = score > 0.7 ? 'score-high' : score > 0.5 ? 'score-med' : 'score-low';
                        html += `
                            <tr>
                                <td><strong>${patientId}</strong></td>
                                <td>${state}</td>
                                <td>${sex}</td>
                                <td>${yob}</td>
                                <td title="${r.diagnosis_codes || r.DIAGNOSIS_CODES || ''}">${diagCodes}</td>
                                <td><span class="score-badge ${scoreClass}">${(score * 100).toFixed(1)}%</span></td>
                            </tr>
                        `;
                    });

                    html += '</tbody></table>';
                }

                document.getElementById('similar-results').innerHTML = html;
            } catch (err) {
                document.getElementById('similar-results').innerHTML = `
                    <div class="empty-state" style="color: #ef4444;">
                        Error: ${err.message}
                    </div>`;
            } finally {
                document.getElementById('similar-loading').classList.remove('active');
            }
        }

        async function loadCohorts() {
            try {
                const res = await fetch('/api/lakehouse/kohorts');
                const data = await res.json();
                let kohorts = data.kohorts || [];
                let legacy = data.legacy || [];

                // Filter based on current role
                if (currentRole !== 'komodo') {
                    const policy = allPolicies.find(p => (p.id || p.policy_id) === currentRole);
                    if (policy && policy.kohort_files) {
                        kohorts = kohorts.filter(k => policy.kohort_files.includes(k.name));
                    }
                    if (policy && policy.bitmap_cohort) {
                        legacy = legacy.filter(c => c.name === policy.bitmap_cohort);
                    }
                }

                if (kohorts.length === 0 && legacy.length === 0) {
                    document.getElementById('cohort-list').innerHTML = '<div class="empty-state">No bitmap cohorts available for this role</div>';
                    return;
                }

                let html = '';

                // New Kohort Files Section
                if (kohorts.length > 0) {
                    html += `
                        <div class="card" style="margin-bottom:16px">
                            <h3>Atlas Kohort Files</h3>
                            <p style="color:var(--muted);font-size:12px;margin-bottom:12px">Click a row to view SELECT * FROM kohort data</p>
                            <table class="results-table">
                                <thead>
                                    <tr>
                                        <th style="width:24px"></th>
                                        <th>Name</th>
                                        <th>Modalities</th>
                                        <th style="text-align:center">Parquet</th>
                                        <th style="text-align:center">Bitmap</th>
                                        <th style="text-align:center">Vector</th>
                                        <th style="text-align:right">Patient Count</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${kohorts.map(k => {
                                        const hasParquet = (k.modalities || []).length > 0;
                                        const kohortId = k.path.replace(/\\./g, '_');
                                        return `
                                            <tr style="cursor:pointer" onclick="toggleKohortData('${k.path}', '${kohortId}')">
                                                <td style="width:24px"><span id="expand-kohort-${kohortId}" style="color:var(--muted)">&#x25B6;</span></td>
                                                <td><strong>${k.name}</strong></td>
                                                <td>${(k.modalities || []).join(', ') || '-'}</td>
                                                <td style="text-align:center">${hasParquet ? '<span style="color:var(--green)">&#x2713;</span>' : '<span style="color:var(--muted)">-</span>'}</td>
                                                <td style="text-align:center">${k.has_membership ? '<span style="color:var(--green)">&#x2713;</span>' : '<span style="color:var(--muted)">-</span>'}</td>
                                                <td style="text-align:center">${k.has_vector_index ? '<span style="color:var(--green)">&#x2713;</span>' : '<span style="color:var(--muted)">-</span>'}</td>
                                                <td style="text-align:right">${(k.patient_count || 0).toLocaleString()}</td>
                                            </tr>
                                            <tr id="kohort-detail-${kohortId}" style="display:none">
                                                <td colspan="7" style="padding:0">
                                                    <div id="kohort-data-${kohortId}" style="padding:16px;background:var(--card);border-top:1px solid var(--border)">
                                                        <div class="loading"></div> Loading data...
                                                    </div>
                                                </td>
                                            </tr>
                                        `;
                                    }).join('')}
                                </tbody>
                            </table>
                        </div>
                    `;
                }

                // Legacy Bitmaps Section
                if (legacy.length > 0) {
                    html += `
                        <div class="card">
                            <h3>Legacy Bitmaps</h3>
                            <table class="results-table">
                                <thead>
                                    <tr>
                                        <th>Cohort ID</th>
                                        <th>Path</th>
                                        <th style="text-align:right">Patient Count</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${legacy.map(c => `
                                        <tr>
                                            <td><strong>${c.name}</strong></td>
                                            <td style="font-size:11px;color:var(--muted)">${c.path}</td>
                                            <td style="text-align:right">${(c.patient_count || 0).toLocaleString()}</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    `;
                }

                document.getElementById('cohort-list').innerHTML = html;
            } catch (err) {
                document.getElementById('cohort-list').innerHTML = `<div class="empty-state" style="color:var(--red)">Error: ${err.message}</div>`;
            }
        }

        async function loadPolicies() {
            try {
                const res = await fetch('/api/lakehouse/policies');
                const data = await res.json();
                let policies = data.policies || [];

                // Filter based on current role
                if (currentRole !== 'komodo') {
                    policies = policies.filter(p => (p.id || p.policy_id) === currentRole);
                }

                if (policies.length === 0) {
                    document.getElementById('policy-list').innerHTML = '<div class="empty-state">No ABAC policies configured</div>';
                    return;
                }

                document.getElementById('policy-list').innerHTML = `
                    <table class="results-table">
                        <thead>
                            <tr>
                                <th></th>
                                <th>Policy ID</th>
                                <th>Name</th>
                                <th>Filter Type</th>
                                <th>Kohort Files</th>
                                <th>Universe Op</th>
                                <th>Max Rows</th>
                                <th>Patient Count</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${policies.map((p, idx) => {
                                const policyId = p.id || p.policy_id;
                                const kohortFiles = p.kohort_files || [];
                                const kohortDisplay = kohortFiles.length > 0
                                    ? kohortFiles.map(f => `<div><span class="policy-tag">${f}</span></div>`).join('')
                                    : (p.bitmap_cohort ? `<span style="color:var(--muted)">${p.bitmap_cohort} (legacy)</span>` : '-');
                                const maskedCols = p.masked_columns || [];
                                const userRoles = p.user_roles || [];
                                const universeIds = p.universe_ids || [];
                                return `
                                    <tr style="cursor:pointer" onclick="togglePolicyDetail('${policyId}')">
                                        <td style="width:24px"><span id="expand-${policyId}" style="color:var(--muted)">&#x25B6;</span></td>
                                        <td><strong>${policyId}</strong></td>
                                        <td>${p.name || '-'}</td>
                                        <td><span class="policy-tag">${p.filter_type || p.type || '-'}</span></td>
                                        <td>${kohortDisplay}</td>
                                        <td>${p.universe_operation || 'union'}</td>
                                        <td>${p.max_rows ? p.max_rows.toLocaleString() : 'Unlimited'}</td>
                                        <td>${p.patient_count ? p.patient_count.toLocaleString() : '-'}</td>
                                        <td>${p.enabled ? '<span style="color:var(--green)">Enabled</span>' : '<span style="color:var(--red)">Disabled</span>'}</td>
                                    </tr>
                                    <tr id="detail-${policyId}" style="display:none">
                                        <td colspan="9" style="background:var(--bg);padding:16px">
                                            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;font-size:12px">
                                                <div>
                                                    <div style="color:var(--muted);font-size:10px;text-transform:uppercase;margin-bottom:4px">Description</div>
                                                    <div>${p.description || '-'}</div>
                                                </div>
                                                <div>
                                                    <div style="color:var(--muted);font-size:10px;text-transform:uppercase;margin-bottom:4px">Masked Columns</div>
                                                    <div>${maskedCols.length > 0 ? maskedCols.map(c => `<span class="policy-tag" style="margin:2px">${c}</span>`).join('') : 'None'}</div>
                                                </div>
                                                <div>
                                                    <div style="color:var(--muted);font-size:10px;text-transform:uppercase;margin-bottom:4px">User Roles</div>
                                                    <div>${userRoles.length > 0 ? userRoles.map(r => `<span class="policy-tag" style="margin:2px">${r}</span>`).join('') : '-'}</div>
                                                </div>
                                                <div>
                                                    <div style="color:var(--muted);font-size:10px;text-transform:uppercase;margin-bottom:4px">Universe IDs</div>
                                                    <div>${universeIds.length > 0 ? universeIds.join(', ') : '-'}</div>
                                                </div>
                                                <div>
                                                    <div style="color:var(--muted);font-size:10px;text-transform:uppercase;margin-bottom:4px">Priority</div>
                                                    <div>${p.priority || '-'}</div>
                                                </div>
                                                <div>
                                                    <div style="color:var(--muted);font-size:10px;text-transform:uppercase;margin-bottom:4px">Customer Tier</div>
                                                    <div>${p.customer_tier || '-'}</div>
                                                </div>
                                                <div>
                                                    <div style="color:var(--muted);font-size:10px;text-transform:uppercase;margin-bottom:4px">Created At</div>
                                                    <div>${p.created_at ? new Date(p.created_at).toLocaleString() : '-'}</div>
                                                </div>
                                                <div>
                                                    <div style="color:var(--muted);font-size:10px;text-transform:uppercase;margin-bottom:4px">Created By</div>
                                                    <div>${p.created_by || '-'}</div>
                                                </div>
                                                <div>
                                                    <div style="color:var(--muted);font-size:10px;text-transform:uppercase;margin-bottom:4px">Predicate (SQL)</div>
                                                    <div style="font-family:monospace">${p.predicate || '-'}</div>
                                                </div>
                                                <div>
                                                    <div style="color:var(--muted);font-size:10px;text-transform:uppercase;margin-bottom:4px">Allowed Sources</div>
                                                    <div>${(p.allowed_sources || []).length > 0 ? p.allowed_sources.map(s => '<span class="policy-tag" style="margin:2px">' + s + '</span>').join('') : '-'}</div>
                                                </div>
                                                <div>
                                                    <div style="color:var(--muted);font-size:10px;text-transform:uppercase;margin-bottom:4px">Source Bitset</div>
                                                    <div style="font-family:monospace">${p.source_bitset != null ? p.source_bitset + ' (0x' + p.source_bitset.toString(16).toUpperCase() + ')' : '-'}</div>
                                                </div>
                                            </div>
                                        </td>
                                    </tr>
                                `;
                            }).join('')}
                        </tbody>
                    </table>
                `;
            } catch (err) {
                document.getElementById('policy-list').innerHTML = `<div class="empty-state" style="color:var(--red)">Error: ${err.message}</div>`;
            }
        }

        function togglePolicyDetail(policyId) {
            const detailRow = document.getElementById('detail-' + policyId);
            const expandIcon = document.getElementById('expand-' + policyId);
            if (detailRow.style.display === 'none') {
                detailRow.style.display = 'table-row';
                expandIcon.innerHTML = '&#x25BC;';
            } else {
                detailRow.style.display = 'none';
                expandIcon.innerHTML = '&#x25B6;';
            }
        }

        const kohortDataLoaded = {};
        async function toggleKohortData(kohortName, kohortId) {
            const detailRow = document.getElementById('kohort-detail-' + kohortId);
            const expandIcon = document.getElementById('expand-kohort-' + kohortId);
            const dataDiv = document.getElementById('kohort-data-' + kohortId);

            if (detailRow.style.display === 'none') {
                detailRow.style.display = 'table-row';
                expandIcon.innerHTML = '&#x25BC;';

                // Load data if not already loaded
                if (!kohortDataLoaded[kohortId]) {
                    try {
                        const res = await fetch('/api/lakehouse/kohort/' + encodeURIComponent(kohortName) + '/data?limit=50&policy_id=' + encodeURIComponent(currentRole));
                        const data = await res.json();

                        if (data.error) {
                            dataDiv.innerHTML = '<div style="color:var(--red)">Error: ' + data.error + '</div>';
                            return;
                        }

                        let html = '<div style="margin-bottom:12px">';
                        html += '<strong>SELECT * FROM ' + kohortName + '</strong> ';
                        html += '<span style="color:var(--muted);font-size:12px">(' + data.modality + ') - ' + data.returned_rows + ' of ' + data.total_rows.toLocaleString() + ' rows in ' + data.elapsed_ms + 'ms</span>';

                        // Modality selector if multiple modalities
                        if (data.modalities && data.modalities.length > 1) {
                            html += '<select style="margin-left:16px;padding:4px 8px;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:4px" onchange="loadKohortModality(\\'' + kohortName + '\\', \\'' + kohortId + '\\', this.value)">';
                            data.modalities.forEach(m => {
                                html += '<option value="' + m + '"' + (m === data.modality ? ' selected' : '') + '>' + m + '</option>';
                            });
                            html += '</select>';
                        }

                        // Open in SQL button
                        html += '<button onclick="openInSql(\\'' + data.modality + '\\')" style="margin-left:16px;padding:4px 12px;background:var(--purple);color:white;border:none;border-radius:4px;cursor:pointer;font-size:11px">Open in SQL</button>';
                        html += '</div>';

                        // Data table
                        if (data.rows && data.rows.length > 0) {
                            html += '<div style="max-height:400px;overflow:auto;border:1px solid var(--border);border-radius:4px">';
                            html += '<table class="results-table" style="margin:0;font-size:11px">';
                            html += '<thead><tr>';
                            data.columns.forEach(col => {
                                html += '<th style="position:sticky;top:0;background:var(--card)">' + col + '</th>';
                            });
                            html += '</tr></thead><tbody>';
                            data.rows.forEach(row => {
                                html += '<tr>';
                                data.columns.forEach(col => {
                                    let val = row[col];
                                    if (val === null || val === undefined) val = '<span style="color:var(--muted)">null</span>';
                                    else if (typeof val === 'object') val = JSON.stringify(val);
                                    html += '<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + val + '</td>';
                                });
                                html += '</tr>';
                            });
                            html += '</tbody></table></div>';
                        } else {
                            html += '<div style="color:var(--muted)">No data rows</div>';
                        }

                        dataDiv.innerHTML = html;
                        kohortDataLoaded[kohortId] = true;
                    } catch (err) {
                        dataDiv.innerHTML = '<div style="color:var(--red)">Error loading data: ' + err.message + '</div>';
                    }
                }
            } else {
                detailRow.style.display = 'none';
                expandIcon.innerHTML = '&#x25B6;';
            }
        }

        async function loadKohortModality(kohortName, kohortId, modality) {
            const dataDiv = document.getElementById('kohort-data-' + kohortId);
            dataDiv.innerHTML = '<div class="loading"></div> Loading ' + modality + '...';
            kohortDataLoaded[kohortId] = false;

            try {
                const res = await fetch('/api/lakehouse/kohort/' + encodeURIComponent(kohortName) + '/data?modality=' + encodeURIComponent(modality) + '&limit=50&policy_id=' + encodeURIComponent(currentRole));
                const data = await res.json();

                if (data.error) {
                    dataDiv.innerHTML = '<div style="color:var(--red)">Error: ' + data.error + '</div>';
                    return;
                }

                let html = '<div style="margin-bottom:12px">';
                html += '<strong>SELECT * FROM ' + kohortName + '</strong> ';
                html += '<span style="color:var(--muted);font-size:12px">(' + data.modality + ') - ' + data.returned_rows + ' of ' + data.total_rows.toLocaleString() + ' rows in ' + data.elapsed_ms + 'ms</span>';

                if (data.modalities && data.modalities.length > 1) {
                    html += '<select style="margin-left:16px;padding:4px 8px;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:4px" onchange="loadKohortModality(\\'' + kohortName + '\\', \\'' + kohortId + '\\', this.value)">';
                    data.modalities.forEach(m => {
                        html += '<option value="' + m + '"' + (m === data.modality ? ' selected' : '') + '>' + m + '</option>';
                    });
                    html += '</select>';
                }

                // Open in SQL button
                html += '<button onclick="openInSql(\\'' + data.modality + '\\')" style="margin-left:16px;padding:4px 12px;background:var(--purple);color:white;border:none;border-radius:4px;cursor:pointer;font-size:11px">Open in SQL</button>';
                html += '</div>';

                if (data.rows && data.rows.length > 0) {
                    html += '<div style="max-height:400px;overflow:auto;border:1px solid var(--border);border-radius:4px">';
                    html += '<table class="results-table" style="margin:0;font-size:11px">';
                    html += '<thead><tr>';
                    data.columns.forEach(col => {
                        html += '<th style="position:sticky;top:0;background:var(--card)">' + col + '</th>';
                    });
                    html += '</tr></thead><tbody>';
                    data.rows.forEach(row => {
                        html += '<tr>';
                        data.columns.forEach(col => {
                            let val = row[col];
                            if (val === null || val === undefined) val = '<span style="color:var(--muted)">null</span>';
                            else if (typeof val === 'object') val = JSON.stringify(val);
                            html += '<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + val + '</td>';
                        });
                        html += '</tr>';
                    });
                    html += '</tbody></table></div>';
                } else {
                    html += '<div style="color:var(--muted)">No data rows</div>';
                }

                dataDiv.innerHTML = html;
                kohortDataLoaded[kohortId] = true;
            } catch (err) {
                dataDiv.innerHTML = '<div style="color:var(--red)">Error loading data: ' + err.message + '</div>';
            }
        }

        // Open modality in SQL console
        function openInSql(modality) {
            // Convert modality to SQL table name (e.g. "Patients-v1" -> "patients_v1")
            const tableName = modality.toLowerCase().replace(/[^a-z0-9_]/g, '_');

            // Navigate to SQL section
            document.querySelectorAll('.nav-item').forEach(item => {
                if (item.dataset.section === 'sql') {
                    item.click();
                }
            });

            // Wait for section to load, then set editor and run query
            setTimeout(() => {
                const editor = document.getElementById('sql-editor');
                if (editor) {
                    editor.value = `SELECT * FROM ${tableName} LIMIT 100`;
                    runSqlQuery();
                }
            }, 100);
        }

        // SQL Query functionality
        let sqlTables = [];
        let sqlBenchmarks = [];

        async function loadSqlTables() {
            try {
                const res = await fetch('/api/tables');
                const data = await res.json();
                sqlTables = data.tables || [];

                const tablesDiv = document.getElementById('sql-tables');
                if (tablesDiv && sqlTables.length > 0) {
                    tablesDiv.innerHTML = sqlTables.map(t =>
                        `<span style="cursor:pointer;background:var(--card);padding:4px 8px;border-radius:4px;font-size:11px" onclick="insertSqlTable('${t}')">${t}</span>`
                    ).join('');

                    // Set up benchmarks
                    sqlBenchmarks = [
                        { label: 'Full scan', sql: `SELECT * FROM ${sqlTables[0]} LIMIT 100` },
                        { label: 'COUNT(*)', sql: `SELECT count(*) AS total FROM ${sqlTables[0]}` },
                        { label: 'GROUP BY', sql: `SELECT patient_sex, count(*) AS n\\nFROM ${sqlTables[0]}\\nGROUP BY patient_sex` },
                        { label: 'Top states', sql: `SELECT patient_state, count(*) AS claims\\nFROM ${sqlTables[0]}\\nGROUP BY patient_state\\nORDER BY claims DESC\\nLIMIT 20` },
                        { label: 'Point lookup', sql: `-- Point query: bitmap index enables file pruning\\nSELECT * FROM ${sqlTables[0]}\\nWHERE patient_number = 20491624` }
                    ];
                }

                // Wire up keyboard shortcut
                const editor = document.getElementById('sql-editor');
                if (editor) {
                    editor.addEventListener('keydown', e => {
                        if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                            e.preventDefault();
                            runSqlQuery();
                        }
                    });
                }
            } catch (err) {
                console.error('Failed to load SQL tables:', err);
            }
        }

        function insertSqlTable(name) {
            const editor = document.getElementById('sql-editor');
            if (!editor) return;
            const start = editor.selectionStart;
            const end = editor.selectionEnd;
            const val = editor.value;
            editor.value = val.slice(0, start) + name + val.slice(end);
            editor.selectionStart = editor.selectionEnd = start + name.length;
            editor.focus();
        }

        function loadSqlBenchmark(idx) {
            const editor = document.getElementById('sql-editor');
            if (!editor || !sqlBenchmarks[idx]) return;
            editor.value = sqlBenchmarks[idx].sql.replace(/\\\\n/g, '\\n');
            editor.focus();
        }

        async function runSqlQuery() {
            const editor = document.getElementById('sql-editor');
            const resultsDiv = document.getElementById('sql-results');
            if (!editor || !resultsDiv) return;

            const sql = editor.value.trim();
            if (!sql) return;

            // Get selected engine
            const engineRadio = document.querySelector('input[name="sql-engine"]:checked');
            const engine = engineRadio ? engineRadio.value : 'datafusion';
            const usePartitions = document.getElementById('sql-use-partitions')?.checked || false;

            resultsDiv.innerHTML = '<div class="loading active"><div class="spinner"></div><div>Executing with ' + engine.toUpperCase() + '...</div></div>';

            const t0 = performance.now();
            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sql, limit: 500, engine, use_partitions: usePartitions })
                });
                const data = await res.json();
                const clientMs = Math.round(performance.now() - t0);

                if (!res.ok) {
                    resultsDiv.innerHTML = `<div style="color:var(--red);padding:12px;background:rgba(239,68,68,0.1);border-radius:4px">${data.detail || 'Query failed'}</div>`;
                    return;
                }

                // Render results
                const engineBadge = data.engine === 'duckdb'
                    ? '<span style="background:#10b981;color:#fff;padding:2px 6px;border-radius:4px;font-size:10px;margin-right:8px">🦆 DuckDB</span>'
                    : '<span style="background:#2563eb;color:#fff;padding:2px 6px;border-radius:4px;font-size:10px;margin-right:8px">🦅 DataFusion</span>';
                const partitionBadge = data.partitioned
                    ? '<span style="background:#a855f7;color:#fff;padding:2px 6px;border-radius:4px;font-size:10px;margin-right:8px">📦 Partitioned</span>'
                    : '';
                let html = `<div style="margin-bottom:12px;color:var(--muted);font-size:12px">`;
                html += engineBadge + partitionBadge;
                html += `<strong>${(data.total_rows || data.row_count || 0).toLocaleString()}</strong> rows`;
                if (data.elapsed_ms) html += ` &middot; Server: ${data.elapsed_ms.toFixed(1)}ms`;
                html += ` &middot; Client: ${clientMs}ms`;
                if (data.catalog_rows) html += ` &middot; Scanned: ${data.catalog_rows.toLocaleString()} rows`;
                html += `</div>`;

                if (data.columns && data.columns.length > 0 && data.rows && data.rows.length > 0) {
                    html += '<div style="max-height:400px;overflow:auto;border:1px solid var(--border);border-radius:4px">';
                    html += '<table class="results-table" style="margin:0;font-size:11px">';
                    html += '<thead><tr>';
                    data.columns.forEach(col => {
                        html += `<th style="position:sticky;top:0;background:var(--card)">${col}</th>`;
                    });
                    html += '</tr></thead><tbody>';
                    data.rows.forEach(row => {
                        html += '<tr>';
                        data.columns.forEach(col => {
                            let val = row[col];
                            if (val === null || val === undefined) val = '<span style="color:var(--muted)">null</span>';
                            else if (typeof val === 'object') val = JSON.stringify(val);
                            html += `<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${val}</td>`;
                        });
                        html += '</tr>';
                    });
                    html += '</tbody></table></div>';
                } else {
                    html += '<div style="color:var(--muted)">No results</div>';
                }

                resultsDiv.innerHTML = html;
            } catch (err) {
                resultsDiv.innerHTML = `<div style="color:var(--red);padding:12px;background:rgba(239,68,68,0.1);border-radius:4px">${err.message}</div>`;
            }
        }

        async function loadUniverses() {
            try {
                const res = await fetch('/api/lakehouse/universes');
                const data = await res.json();
                const universes = data.universes || [];

                if (universes.length === 0) {
                    document.getElementById('universe-list').innerHTML = '<div class="empty-state">No customer universes computed (no policies with kohort_files)</div>';
                    return;
                }

                // Calculate total patients across all universes
                const totalPatients = universes.reduce((sum, u) => sum + (u.patient_count || 0), 0);

                let html = `
                    <div class="stats-grid" style="margin-bottom:16px">
                        <div class="stat-card">
                            <div class="value">${universes.length}</div>
                            <div class="label">Customer Universes</div>
                        </div>
                        <div class="stat-card">
                            <div class="value">${totalPatients.toLocaleString()}</div>
                            <div class="label">Total Patients</div>
                        </div>
                    </div>
                    <table class="results-table">
                        <thead>
                            <tr>
                                <th>Customer</th>
                                <th>Kohort Files</th>
                                <th>Operation</th>
                                <th style="text-align:right">Universe Size</th>
                                <th style="text-align:right">Compute Time</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${universes.map(u => {
                                const kohortTags = (u.kohort_files || []).map(f =>
                                    `<div><span class="policy-tag">${f.replace('atlas_', '').replace('.kohort', '')}</span></div>`
                                ).join('');
                                const opBadge = u.operation === 'union'
                                    ? '<span style="color:var(--green)">&#x222A; union</span>'
                                    : '<span style="color:var(--accent)">&#x2229; intersect</span>';
                                return `
                                    <tr>
                                        <td>
                                            <strong>${u.policy_name || u.policy_id}</strong>
                                            ${u.description ? `<div style="font-size:11px;color:var(--muted)">${u.description}</div>` : ''}
                                        </td>
                                        <td>${kohortTags || '-'}</td>
                                        <td>${opBadge}</td>
                                        <td style="text-align:right"><strong>${(u.patient_count || 0).toLocaleString()}</strong></td>
                                        <td style="text-align:right;color:var(--muted)">${(u.compute_ms || 0).toFixed(1)}ms</td>
                                    </tr>
                                `;
                            }).join('')}
                        </tbody>
                    </table>
                `;

                document.getElementById('universe-list').innerHTML = html;
            } catch (err) {
                document.getElementById('universe-list').innerHTML = `<div class="empty-state" style="color:var(--red)">Error: ${err.message}</div>`;
            }
        }

        async function loadAuditLog() {
            try {
                const res = await fetch('/api/lakehouse/audit?limit=50');
                const data = await res.json();
                const entries = data.entries || [];

                if (entries.length === 0) {
                    document.getElementById('audit-log').innerHTML = '<div class="empty-state">No audit entries yet</div>';
                    return;
                }

                document.getElementById('audit-log').innerHTML = `
                    <table class="results-table">
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Query Type</th>
                                <th>User</th>
                                <th>Rows</th>
                                <th>Duration</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${entries.map(e => `
                                <tr>
                                    <td>${new Date(e.timestamp).toLocaleString()}</td>
                                    <td><span class="policy-tag">${e.query_type}</span></td>
                                    <td>${e.user_id || 'anonymous'}</td>
                                    <td>${e.rows_returned?.toLocaleString() || '-'}</td>
                                    <td>${e.elapsed_ms?.toFixed(0)}ms</td>
                                    <td>${e.success ? '<span style="color:#22c55e">✓</span>' : '<span style="color:#ef4444">✗</span>'}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                `;
            } catch (err) {
                document.getElementById('audit-log').innerHTML = `<div class="empty-state" style="color:#ef4444">Error: ${err.message}</div>`;
            }
        }

        // Allow Enter key to trigger search
        document.getElementById('search-query').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') runSemanticSearch();
        });
        document.getElementById('similar-patient-id').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') findSimilarPatients();
        });

        // Role selector functions
        async function loadRoleSelector() {
            try {
                const res = await fetch('/api/lakehouse/policies');
                const data = await res.json();
                allPolicies = data.policies || [];

                const select = document.getElementById('role-select');
                let options = '<option value="komodo">🔓 Komodo (Full Access)</option>';

                allPolicies.forEach(p => {
                    const policyId = p.id || p.policy_id;
                    options += `<option value="${policyId}">🔒 ${p.name || policyId}</option>`;
                });

                select.innerHTML = options;
                updateRoleInfo('komodo');
            } catch (err) {
                console.error('Error loading roles:', err);
            }
        }

        function switchRole(roleId) {
            currentRole = roleId;
            updateRoleInfo(roleId);
            // Refresh ALL sections to reflect new role
            loadDashboard();
            loadCohorts();
            loadPolicies();
            // Clear any search results since they may not be valid for new role
            document.getElementById('search-results').innerHTML = '<div class="empty-state">Role changed - run a new search</div>';
            document.getElementById('similar-results').innerHTML = '<div class="empty-state">Role changed - find new similar patients</div>';
        }

        function updateRoleInfo(roleId) {
            const infoDiv = document.getElementById('role-info');
            if (roleId === 'komodo') {
                infoDiv.innerHTML = `
                    <span class="role-badge admin">Admin</span>
                    <span>All datasets • All patients accessible</span>
                `;
            } else {
                const policy = allPolicies.find(p => (p.id || p.policy_id) === roleId);
                if (policy) {
                    let filterInfo = '';
                    if (policy.filter_type === 'kohort' && policy.kohort_files) {
                        filterInfo = `Kohort: ${policy.kohort_files.join(', ')}`;
                    } else if (policy.filter_type === 'bitmap') {
                        filterInfo = `Bitmap: ${policy.bitmap_cohort || 'configured'}`;
                    } else {
                        filterInfo = `Predicate: ${policy.predicate || 'configured'}`;
                    }
                    const patientCount = policy.patient_count ? ` • ${policy.patient_count.toLocaleString()} patients` : '';
                    const sourceInfo = policy.allowed_sources && policy.allowed_sources.length > 0
                        ? ` • Sources: ${policy.allowed_sources.join(', ')} (0x${policy.source_bitset.toString(16).toUpperCase()})`
                        : '';
                    infoDiv.innerHTML = `
                        <span class="role-badge customer">Customer</span>
                        <span>${filterInfo}${patientCount}${sourceInfo}</span>
                    `;
                }
            }
        }

        // Observability functions
        async function loadObservability() {
            try {
                const [auditRes, statsRes] = await Promise.all([
                    fetch('/api/lakehouse/audit?limit=50'),
                    fetch('/api/lakehouse/audit/stats')
                ]);

                const audit = await auditRes.json();
                const stats = await statsRes.json();

                const entries = audit.entries || [];

                // Calculate metrics
                const totalQueries = entries.length;
                const latencies = entries.map(e => e.elapsed_ms || 0).filter(l => l > 0);
                const avgLatency = latencies.length ? (latencies.reduce((a, b) => a + b, 0) / latencies.length).toFixed(0) : 0;
                const sortedLatencies = [...latencies].sort((a, b) => a - b);
                const p95Index = Math.floor(sortedLatencies.length * 0.95);
                const p95Latency = sortedLatencies[p95Index] || 0;
                const successCount = entries.filter(e => e.success).length;
                const successRate = totalQueries ? ((successCount / totalQueries) * 100).toFixed(1) : 100;

                // Update metrics
                document.getElementById('obs-total-queries').textContent = totalQueries.toLocaleString();
                document.getElementById('obs-avg-latency').textContent = avgLatency + 'ms';
                document.getElementById('obs-p95-latency').textContent = p95Latency.toFixed(0) + 'ms';
                document.getElementById('obs-success-rate').textContent = successRate + '%';

                // Latency chart (last 20 queries)
                const chartEntries = entries.slice(0, 20).reverse();
                if (chartEntries.length > 0) {
                    const maxLatency = Math.max(...chartEntries.map(e => e.elapsed_ms || 1));
                    document.getElementById('latency-chart').innerHTML = chartEntries.map(e => {
                        const height = Math.max(4, (e.elapsed_ms / maxLatency) * 160);
                        return `<div class="latency-bar" style="height:${height}px" data-value="${e.elapsed_ms?.toFixed(0)}ms - ${e.query_type}"></div>`;
                    }).join('');
                }

                // Query breakdown by type
                const byType = {};
                entries.forEach(e => {
                    byType[e.query_type] = (byType[e.query_type] || 0) + 1;
                });

                document.getElementById('query-breakdown').innerHTML = Object.entries(byType).map(([type, count]) => `
                    <div class="cohort-item">
                        <div class="name">${type}</div>
                        <div class="count">${count} queries</div>
                    </div>
                `).join('') || '<div class="empty-state">No query data</div>';

            } catch (err) {
                console.error('Error loading observability:', err);
                document.getElementById('query-breakdown').innerHTML = `<div class="empty-state" style="color:#ef4444">Error: ${err.message}</div>`;
            }
        }

        // ── Benchmark Functions ─────────────────────────────────────

        async function compareEngines() {
            const editor = document.getElementById('sql-editor');
            const resultsDiv = document.getElementById('sql-results');
            if (!editor || !resultsDiv) return;

            const sql = editor.value.trim() || `SELECT count(*) FROM ${sqlTables[0] || 'cohort'}`;

            resultsDiv.innerHTML = '<div class="loading active"><div class="spinner"></div><div>Comparing DataFusion vs DuckDB...</div></div>';

            try {
                const [dfRes, duckRes] = await Promise.all([
                    fetch('/api/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ sql, limit: 100, engine: 'datafusion' })
                    }),
                    fetch('/api/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ sql, limit: 100, engine: 'duckdb' })
                    })
                ]);

                const dfData = await dfRes.json();
                const duckData = await duckRes.json();

                const dfMs = dfData.elapsed_ms || 0;
                const duckMs = duckData.elapsed_ms || 0;
                const winner = dfMs < duckMs ? 'DataFusion' : 'DuckDB';
                const speedup = dfMs < duckMs ? (duckMs / dfMs).toFixed(1) : (dfMs / duckMs).toFixed(1);

                let html = `
                    <div style="margin-bottom:20px">
                        <h3 style="margin-bottom:12px">Engine Comparison Results</h3>
                        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:16px">
                            <div style="background:var(--bg);padding:16px;border-radius:8px;text-align:center;${dfMs < duckMs ? 'border:2px solid var(--green)' : ''}">
                                <div style="font-size:20px;margin-bottom:4px">🦅</div>
                                <div style="font-size:14px;font-weight:600">DataFusion</div>
                                <div style="font-size:24px;font-weight:700;color:var(--accent);margin:8px 0">${dfMs.toFixed(1)}ms</div>
                                ${dfMs < duckMs ? '<div style="color:var(--green);font-size:12px">✓ Winner</div>' : ''}
                            </div>
                            <div style="background:var(--bg);padding:16px;border-radius:8px;text-align:center;${duckMs < dfMs ? 'border:2px solid var(--green)' : ''}">
                                <div style="font-size:20px;margin-bottom:4px">🦆</div>
                                <div style="font-size:14px;font-weight:600">DuckDB</div>
                                <div style="font-size:24px;font-weight:700;color:var(--green);margin:8px 0">${duckMs.toFixed(1)}ms</div>
                                ${duckMs < dfMs ? '<div style="color:var(--green);font-size:12px">✓ Winner</div>' : ''}
                            </div>
                            <div style="background:var(--bg);padding:16px;border-radius:8px;text-align:center">
                                <div style="font-size:20px;margin-bottom:4px">⚡</div>
                                <div style="font-size:14px;font-weight:600">Speedup</div>
                                <div style="font-size:24px;font-weight:700;color:var(--yellow);margin:8px 0">${speedup}x</div>
                                <div style="font-size:12px;color:var(--muted)">${winner} faster</div>
                            </div>
                        </div>
                        <div style="font-size:12px;color:var(--muted);font-family:monospace;background:var(--bg);padding:8px;border-radius:4px">
                            ${sql.replace(/\\n/g, ' ')}
                        </div>
                    </div>
                `;

                resultsDiv.innerHTML = html;
            } catch (err) {
                resultsDiv.innerHTML = `<div style="color:var(--red)">${err.message}</div>`;
            }
        }

        async function runEngineComparison() {
            const btn = document.getElementById('engine-compare-btn');
            const resultsDiv = document.getElementById('engine-comparison-results');
            if (!resultsDiv) return;

            btn.disabled = true;
            btn.textContent = 'Running...';

            // Get first available table for the benchmark query
            const tablesRes = await fetch('/api/tables');
            const tablesData = await tablesRes.json();
            const firstTable = tablesData.tables?.[0] || 'cohort';
            const sql = `SELECT count(*) as cnt FROM "${firstTable}"`;

            resultsDiv.innerHTML = '<div class="loading active"><div class="spinner"></div><div>Comparing DataFusion vs DuckDB vs Polars...</div></div>';

            try {
                const [dfRes, duckRes, polarsRes] = await Promise.all([
                    fetch('/api/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ sql, limit: 100, engine: 'datafusion' })
                    }),
                    fetch('/api/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ sql, limit: 100, engine: 'duckdb' })
                    }),
                    fetch('/api/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ sql, limit: 100, engine: 'polars' })
                    })
                ]);

                const dfData = await dfRes.json();
                const duckData = await duckRes.json();
                const polarsData = await polarsRes.json();

                const dfMs = dfData.elapsed_ms || 0;
                const duckMs = duckData.elapsed_ms || 0;
                const polarsMs = polarsData.elapsed_ms || 0;

                const times = [
                    { name: 'DataFusion', ms: dfMs, emoji: '🦅', color: 'var(--accent)' },
                    { name: 'DuckDB', ms: duckMs, emoji: '🦆', color: 'var(--green)' },
                    { name: 'Polars', ms: polarsMs, emoji: '🐻‍❄️', color: 'var(--purple,#a371f7)' }
                ].sort((a, b) => a.ms - b.ms);

                const winner = times[0];
                const fastest = times[0].ms;

                let html = `
                    <h4 style="margin-bottom:12px">Engine Comparison Results</h4>
                    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px">
                `;

                times.forEach((eng, i) => {
                    const isWinner = i === 0;
                    const speedup = eng.ms > 0 ? (eng.ms / fastest).toFixed(1) : '∞';
                    html += `
                        <div style="background:var(--bg);padding:16px;border-radius:8px;text-align:center;${isWinner ? 'border:2px solid var(--green)' : ''}">
                            <div style="font-size:20px;margin-bottom:4px">${eng.emoji}</div>
                            <div style="font-size:14px;font-weight:600">${eng.name}</div>
                            <div style="font-size:24px;font-weight:700;color:${eng.color};margin:8px 0">${eng.ms.toFixed(1)}ms</div>
                            ${isWinner ? '<div style="color:var(--green);font-size:12px">✓ Fastest</div>' : `<div style="color:var(--muted);font-size:12px">${speedup}x slower</div>`}
                        </div>
                    `;
                });

                html += `</div>`;
                html += `<div style="font-size:12px;color:var(--muted);font-family:monospace;background:var(--bg);padding:8px;border-radius:4px">${sql}</div>`;

                resultsDiv.innerHTML = html;
            } catch (err) {
                resultsDiv.innerHTML = `<div style="color:var(--red)">${err.message}</div>`;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Run Engine Comparison';
            }
        }

        async function runCodecBenchmark() {
            const btn = document.getElementById('bench-codec-btn');
            const resultsDiv = document.getElementById('codec-benchmark-results');
            if (!resultsDiv) return;

            btn.disabled = true;
            btn.textContent = 'Running...';
            resultsDiv.innerHTML = '<div class="loading active"><div class="spinner"></div><div>Running codec benchmark (this may take 10-30 seconds)...</div></div>';

            try {
                const res = await fetch('/api/benchmark', { method: 'POST' });
                const data = await res.json();

                if (!res.ok) {
                    resultsDiv.innerHTML = `<div style="color:var(--red)">${data.detail || 'Benchmark failed'}</div>`;
                    return;
                }

                // Extract SDK benchmark data from API response
                const sdk = data.sdk || {};
                const grid = sdk.grid || {};
                const codecs = sdk.codecs || ['parquet', 'arrow_ipc', 'feather'];
                const summary = data.summary || {};

                let html = `<h4 style="margin-bottom:12px">Benchmark Results</h4>`;
                html += `<p style="color:var(--muted);font-size:12px;margin-bottom:12px">Source: ${summary.source_file || 'N/A'} | Rows tested: ${(summary.test_rows || 0).toLocaleString()}</p>`;
                html += '<div style="overflow-x:auto"><table class="results-table" style="font-size:12px">';
                html += '<thead><tr><th>Metric</th>';

                const codecLabels = {parquet: 'Parquet', arrow_ipc: 'Arrow IPC', feather: 'Feather'};
                codecs.forEach(c => { html += `<th style="text-align:center">${codecLabels[c] || c}</th>`; });
                html += '</tr></thead><tbody>';

                // Map API metrics to display labels
                const metrics = ['sdk_write', 'sdk_read', 'file_size', 'sql_count', 'sql_filter'];
                const labels = ['Write Time (ms)', 'Read Time (ms)', 'File Size (MB)', 'SQL COUNT(*) (ms)', 'SQL Filter (ms)'];

                metrics.forEach((m, i) => {
                    html += `<tr><td>${labels[i]}</td>`;
                    codecs.forEach(c => {
                        const val = grid[c]?.[m];
                        const display = val !== undefined ? (typeof val === 'number' ? val.toLocaleString() : val) : '-';
                        // Highlight winners
                        const isWinner = sdk.winners?.[m] === c;
                        const style = isWinner ? 'font-weight:600;color:var(--green)' : '';
                        html += `<td style="text-align:center;${style}">${display}</td>`;
                    });
                    html += '</tr>';
                });

                html += '</tbody></table></div>';

                // Add speedup summary
                const pqRead = grid.parquet?.sdk_read;
                const ipcRead = grid.arrow_ipc?.sdk_read;
                if (pqRead && ipcRead) {
                    const speedup = (pqRead / ipcRead).toFixed(1);
                    html += `<p style="margin-top:12px;color:var(--muted);font-size:12px">Arrow IPC is <strong style="color:var(--green)">${speedup}x faster</strong> than Parquet for reads.</p>`;
                }

                resultsDiv.innerHTML = html;
            } catch (err) {
                resultsDiv.innerHTML = `<div style="color:var(--red)">${err.message}</div>`;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Run Codec Benchmark';
            }
        }

        function loadBenchmarkFiles() {
            // Pre-populate any needed data for benchmark section
            console.log('Benchmark section loaded');
        }

        // ── Bitmap Demo Functions ───────────────────────────────────

        async function runBitmapDemo() {
            const patientIdInput = document.getElementById('bitmap-patient-id');
            const resultsDiv = document.getElementById('bitmap-demo-results');
            if (!resultsDiv) return;

            const patientId = patientIdInput?.value?.trim() || '20491624';
            const tableName = sqlTables[0] || 'cohort';

            resultsDiv.innerHTML = '<div class="loading active"><div class="spinner"></div><div>Running bitmap pruning benchmark...</div></div>';

            try {
                // Run same query with bitmap-enabled DataFusion
                const sql = `SELECT * FROM ${tableName} WHERE patient_number = ${patientId}`;

                const t0 = performance.now();
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sql, limit: 10, engine: 'datafusion' })
                });
                const dfMs = performance.now() - t0;
                const data = await res.json();

                const serverMs = data.elapsed_ms || 0;
                const foundRows = data.total_rows || 0;

                let html = `
                    <div style="background:var(--bg);padding:20px;border-radius:8px">
                        <h4 style="margin-bottom:12px">Point Query: WHERE patient_number = ${patientId}</h4>
                        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:16px">
                            <div style="text-align:center">
                                <div style="font-size:11px;color:var(--muted)">Server Time</div>
                                <div style="font-size:24px;font-weight:700;color:var(--green)">${serverMs.toFixed(1)}ms</div>
                            </div>
                            <div style="text-align:center">
                                <div style="font-size:11px;color:var(--muted)">Round Trip</div>
                                <div style="font-size:24px;font-weight:700;color:var(--accent)">${dfMs.toFixed(0)}ms</div>
                            </div>
                            <div style="text-align:center">
                                <div style="font-size:11px;color:var(--muted)">Rows Found</div>
                                <div style="font-size:24px;font-weight:700;color:var(--yellow)">${foundRows}</div>
                            </div>
                        </div>
                        <p style="color:var(--muted);font-size:12px;margin:0">
                            With bitmap index: DataFusion checks the Roaring bitmap to skip files that don't contain patient ${patientId}.
                            Only files where the bitmap contains this ID are scanned.
                        </p>
                    </div>
                `;

                resultsDiv.innerHTML = html;
            } catch (err) {
                resultsDiv.innerHTML = `<div style="color:var(--red)">${err.message}</div>`;
            }
        }

        // ── Partitioning Functions ──────────────────────────────────

        async function loadPartitionStatus() {
            const statusDiv = document.getElementById('partition-status');
            const sourceSelect = document.getElementById('partition-source');
            if (!statusDiv) return;

            try {
                const [statusRes, filesRes] = await Promise.all([
                    fetch('/api/partition/status'),
                    fetch('/api/files')
                ]);

                const status = await statusRes.json();
                const files = await filesRes.json();

                // Populate source file dropdown
                if (sourceSelect && files.files) {
                    sourceSelect.innerHTML = '<option value="">Select a .kohort file</option>' +
                        files.files.map(f => `<option value="${f.name}">${f.name} (${(f.num_rows || 0).toLocaleString()} rows)</option>`).join('');
                }

                if (status.has_partitions) {
                    statusDiv.innerHTML = `
                        <div style="display:flex;align-items:center;gap:16px">
                            <div style="background:var(--green);color:#fff;padding:8px 16px;border-radius:8px;font-weight:600">
                                ✓ Partitions Available
                            </div>
                            <div style="color:var(--muted)">
                                ${status.num_partitions} partition files in <code>${status.partition_dir}</code>
                            </div>
                        </div>
                        <div style="margin-top:16px">
                            <div style="font-size:12px;color:var(--muted);margin-bottom:8px">Partition Files:</div>
                            <div style="display:flex;flex-wrap:wrap;gap:8px">
                                ${status.files.map(f => `<span style="background:var(--bg);padding:4px 8px;border-radius:4px;font-size:11px">${f}</span>`).join('')}
                            </div>
                        </div>
                    `;
                } else {
                    statusDiv.innerHTML = `
                        <div style="display:flex;align-items:center;gap:16px">
                            <div style="background:var(--bg);color:var(--muted);padding:8px 16px;border-radius:8px">
                                No partitions created yet
                            </div>
                            <div style="color:var(--muted);font-size:13px">
                                Use the form below to partition a large .kohort file for parallel query execution.
                            </div>
                        </div>
                    `;
                }
            } catch (err) {
                statusDiv.innerHTML = `<div style="color:var(--red)">${err.message}</div>`;
            }
        }

        async function createPartitions() {
            const sourceSelect = document.getElementById('partition-source');
            const targetRowsInput = document.getElementById('partition-target-rows');
            const resultsDiv = document.getElementById('partition-create-results');
            if (!resultsDiv) return;

            const file = sourceSelect?.value;
            const targetRows = parseInt(targetRowsInput?.value) || 50000;

            if (!file) {
                resultsDiv.innerHTML = '<div style="color:var(--red)">Please select a source file</div>';
                return;
            }

            resultsDiv.innerHTML = '<div class="loading active"><div class="spinner"></div><div>Creating partitions...</div></div>';

            try {
                const res = await fetch('/api/partition', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ file, target_rows: targetRows })
                });

                const data = await res.json();

                if (!res.ok) {
                    resultsDiv.innerHTML = `<div style="color:var(--red)">${data.detail || 'Partitioning failed'}</div>`;
                    return;
                }

                resultsDiv.innerHTML = `
                    <div style="background:var(--green);color:#fff;padding:16px;border-radius:8px;margin-bottom:16px">
                        ✓ Created ${data.num_partitions} partitions in ${(data.elapsed_ms / 1000).toFixed(1)}s
                    </div>
                    <div style="display:flex;flex-wrap:wrap;gap:8px">
                        ${data.files.map(f => `<span style="background:var(--bg);padding:4px 8px;border-radius:4px;font-size:11px">${f}</span>`).join('')}
                    </div>
                `;

                // Refresh status
                loadPartitionStatus();
            } catch (err) {
                resultsDiv.innerHTML = `<div style="color:var(--red)">${err.message}</div>`;
            }
        }
    </script>
</body>
</html>
"""


# ── Entry point ──────────────────────────────────────────────


def main():
    import uvicorn

    global COHORT_DIR
    if len(sys.argv) > 1:
        COHORT_DIR = Path(sys.argv[1])
    if not COHORT_DIR.exists():
        print(f"Error: directory not found: {COHORT_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"Cohort Visualizer — scanning {COHORT_DIR.resolve()}")

    # Preload similarity model + cached indexes at startup
    cache_dir = COHORT_DIR / ".vector_cache"
    if cache_dir.exists() and any(cache_dir.glob("*_vectors.npz")):
        print("Preloading similarity engine...")
        try:
            from cohort_viz.similarity_engine import _get_model
            _get_model()  # Load BioLORD model once
            for npz in sorted(cache_dir.glob("*_vectors.npz")):
                cohort_file = npz.stem.replace("_vectors", "") + ".cohort"
                if (COHORT_DIR / cohort_file).exists():
                    _get_similarity_engine(cohort_file)
            print(f"Similarity engine ready ({len(_similarity_engines)} cohorts)")
        except Exception as e:
            print(f"Similarity preload skipped: {e}")

    uvicorn.run(app, host="127.0.0.1", port=4141)


if __name__ == "__main__":
    main()

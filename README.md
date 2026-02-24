# Atlas

A high-performance `.kohort` file format viewer with multi-engine SQL support.

## The .kohort Stack

```
┌─────────────────────────────────────────────────────────────┐
│  QUERY ENGINES (SQL execution)                              │
│  DataFusion — DuckDB — Polars                               │
├─────────────────────────────────────────────────────────────┤
│  CODECS (data storage format)                               │
│  Parquet — Arrow IPC — Feather — Vortex (coming soon)       │
├─────────────────────────────────────────────────────────────┤
│  INDEXES (fast filtering) — Tag 200                         │
│  Roaring Bitmaps — Source bitsets                           │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Multi-engine SQL**: Query `.kohort` files via DataFusion (native), DuckDB, or Polars
- **Roaring Bitmap indexes**: 10-100x faster point queries via bitmap pushdown
- **Patient similarity search**: BioLORD-2023 embeddings for semantic patient matching
- **Policy-based access control**: Universe definitions and row-level security
- **Codec benchmarks**: Compare Parquet, Arrow IPC, and Feather performance

## Quick Start

```bash
# Install dependencies
uv sync

# Run the Atlas server
uv run python -m cohort_viz.server /path/to/kohort/files

# Open in browser
open http://localhost:4141/atlas
```

## Query Engines

| Engine | Best For | SQL Support |
|--------|----------|-------------|
| **DataFusion** | Native .kohort, bitmap pushdown | Good |
| **DuckDB** | Complex analytics, window functions | Excellent |
| **Polars** | DataFrame ops, streaming | Basic |

### Example Queries

```sql
-- DataFusion (default)
SELECT COUNT(*) FROM patients_v1

-- DuckDB (full SQL)
SELECT patient_id,
       ROW_NUMBER() OVER (PARTITION BY state ORDER BY age DESC) as rank
FROM patients_v1

-- Polars
SELECT state, COUNT(*) as cnt FROM patients_v1 GROUP BY state
```

## Codecs

| Codec | Compression | Read Speed | Best For |
|-------|-------------|------------|----------|
| **Parquet** | Best | Good | Storage, archives |
| **Arrow IPC** | Low | Fastest | Hot data, caching |
| **Feather** | Low | Fast | Zero-copy reads |
| **Vortex** | TBD | TBD | Coming soon |

## .kohort File Format

A TIFF-style binary container with tagged sections:

```
┌──────────────────────────────────────┐
│ Header (magic, version, tag count)   │
├──────────────────────────────────────┤
│ Tag 100: Parquet data (patients)     │
│ Tag 101: Parquet data (claims)       │
│ Tag 200: Roaring bitmap indexes      │
│ Tag 300: Lance vector embeddings     │
└──────────────────────────────────────┘
```

## Project Structure

```
atlas/
├── cohort_viz/          # Atlas UI & FastAPI server
│   ├── server.py        # Main server with /atlas endpoint
│   ├── benchmarks.py    # Codec benchmark runner
│   └── similarity_engine.py  # BioLORD patient similarity
├── cohort-sdk/          # Rust SDK with Python bindings
│   ├── src/lib.rs       # PyO3 Rust implementation
│   └── python/          # Python wrapper classes
├── policies/            # Access control policies
│   ├── universes.json   # Universe definitions
│   └── policies/        # Per-customer policies
├── benchmarks/          # Benchmark scripts & results
└── cohort.py            # Core Python module
```

## Benchmarks

Run codec benchmarks:
```bash
uv run python -m cohort_viz.benchmarks
```

Or use the Atlas UI → Benchmarks section.

### Recent Results (3M rows)

| Metric | Parquet | Arrow IPC | Feather |
|--------|---------|-----------|---------|
| Write | 190ms | 85ms | 90ms |
| Read | 80ms | 25ms | 30ms |
| Size | 15MB | 45MB | 45MB |

## API Endpoints

```
GET  /atlas              # Main UI
GET  /api/files          # List .kohort files
POST /api/query          # Run SQL query
     {"sql": "...", "engine": "datafusion|duckdb|polars"}
POST /api/benchmark      # Run codec benchmark
GET  /api/similarity/{patient_id}  # Find similar patients
```

## License

MIT

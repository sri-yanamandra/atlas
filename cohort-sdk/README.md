# cohort-sdk

Python SDK for reading and writing `.cohort` files — a binary container format for patient cohort data. Backed by Rust for performance.

## Install

```bash
cd kohort
uv sync
```

## Quick Start

### Write a cohort

```python
from cohort_sdk import CohortDefinition, Selector, Lineage
import pyarrow as pa

claims = pa.table({"patient_id": [1, 2, 3], "diagnosis": ["E11.0", "E11.1", "E11.9"]})

definition = CohortDefinition(
    name="T2D Patients",
    selectors=[
        Selector(selector_id="sel-dx", field="diagnosis_code", operator="in", values=["E11.0", "E11.1", "E11.9"]),
        Selector(selector_id="sel-state", field="patient_state", operator="in", values=["CA", "TX"]),
    ],
    lineage=Lineage(source_system="Snowflake", source_tables=["claims"], extraction_date="2026-02-10"),
)

definition.write("my_cohort.cohort", [("Claims-v1", claims)])
```

### Read a cohort

```python
from cohort_sdk import CohortDefinition

cohort = CohortDefinition.read("my_cohort.cohort")

cohort.definition              # CohortDefinition
cohort.definition.name         # "T2D Patients"
cohort.definition.selectors    # [Selector(...), ...]
cohort.datasets              # ["Claims-v1"]
cohort["Claims-v1"]            # PyArrow Table
```

### Selectors

Selectors define the filter criteria applied to source data during extraction.

| Operator | Meaning | `values` example |
|----------|---------|------------------|
| `in` | Field value is in the list | `["E11.0", "E11.1"]` |
| `not_in` | Field value is NOT in the list | `["E11.0"]` |
| `eq` | Equals a single value | `["CA"]` |
| `neq` | Not equal | `["CA"]` |
| `gt` | Greater than | `[65]` |
| `gte` | Greater than or equal | `[65]` |
| `lt` | Less than | `[18]` |
| `lte` | Less than or equal | `[18]` |
| `between` | Inclusive range | `[40, 75]` |

```python
from cohort_sdk import Selector

Selector(selector_id="sel-dx", field="diagnosis_code", code_system="ICD-10-CM", operator="in", values=["E11.0", "E11.1"])
Selector(selector_id="sel-age", field="patient_age", operator="between", values=[40, 75])
Selector(selector_id="sel-state", field="patient_state", operator="not_in", values=["PR", "GU"])
Selector(selector_id="sel-gender", field="patient_gender", operator="eq", values=["F"])
```

### From Snowflake

```python
from cohort_sdk import SnowflakeConnection

with SnowflakeConnection() as sf:
    sf.extract_to_cohort(
        "my_cohort.cohort",
        cohort_name="T2D Patients",
        queries={
            "Claims-v1": "SELECT * FROM claims WHERE diagnosis_code LIKE 'E11%'",
            "Demographics-v1": "SELECT * FROM demographics WHERE patient_id IN (...)",
        },
        created_by="user:jsmith",
    )
```

Or with an existing Komodo connection:

```python
from komodo.snowflake import get_snowflake_connection
sf = SnowflakeConnection(conn=get_snowflake_connection())
```

### Revisions

```python
from cohort_sdk import CohortDefinition, Delta, RevisionEntry, Selector
from datetime import datetime, timezone

cohort = CohortDefinition.read("my_cohort.cohort")
defn = cohort.definition

# Add a selector
defn.apply_revision(RevisionEntry(
    revision=2,
    timestamp=datetime.now(timezone.utc),
    author="user:jdoe",
    message="Add age filter",
    deltas=[Delta(
        action="add",
        selector_id="sel-age",
        selector=Selector(selector_id="sel-age", field="patient_age", operator="between", values=[40, 75]),
    )],
))

# Modify a selector
defn.apply_revision(RevisionEntry(
    revision=3,
    timestamp=datetime.now(timezone.utc),
    author="user:jdoe",
    message="Expand to national scope",
    deltas=[Delta(action="modify", selector_id="sel-state", field="values", before=["CA", "TX"], after=["CA", "TX", "NY"])],
))

# Remove a selector
defn.apply_revision(RevisionEntry(
    revision=4,
    timestamp=datetime.now(timezone.utc),
    author="user:jdoe",
    message="Remove state filter",
    deltas=[Delta(action="remove", selector_id="sel-state")],
))

# Write updated cohort back
defn.write("my_cohort_v4.cohort", [("Claims-v1", cohort["Claims-v1"])])
```

## Architecture

```
Python                          Rust
──────                          ────
cohort_sdk/                     cohort-lib/src/
  __init__.py  (read, write)
  reader.py   ──►  _cohort_rs  ──►  reader.rs     (header parsing, validation)
  writer.py   ──►    (PyO3)    ──►  writer.rs     (binary layout, offset patching)
  models.py                         models.rs     (TagDirectory, CohortDefinition, ...)
  snowflake.py                      offset_reader.rs
```

Data crosses the Rust/Python boundary as **bytes** (Parquet blocks) and **JSON** (metadata).

## File Format

See `prompts/COHORT_FORMAT.md` for the full spec. Summary:

```
Offset 0:   COHORT         (6 bytes, magic)
Offset 6:   0x03           (1 byte, version)
Offset 7:   <dir_offset>   (8 bytes, u64 LE)
Offset 15:  <data blocks>  (Parquet)
            <tag directory> (JSON)
```

## Tests

```bash
# All tests (from kohort/)
uv run pytest

# Rust tests
cd cohort-lib && cargo test
```

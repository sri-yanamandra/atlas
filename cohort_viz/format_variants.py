"""Multi-format cohort writers for benchmarking.

Creates cohort files with different internal storage formats:
- .cohort  → Parquet (ZSTD compressed)
- .cohorta → Arrow IPC (uncompressed, fast reads)
- .cohortf → Feather (Arrow IPC with compression)

All use the same TIFF-like wrapper with JSON metadata at the end.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as pq


MAGIC = b"COHORT"
VERSION = 3


def _write_header(f, dir_offset_placeholder: int = 0) -> int:
    """Write the 15-byte cohort header. Returns position of dir_offset field."""
    f.write(MAGIC)  # 6 bytes
    f.write(struct.pack("B", VERSION))  # 1 byte
    dir_offset_pos = f.tell()
    f.write(struct.pack("<Q", dir_offset_placeholder))  # 8 bytes, little-endian u64
    return dir_offset_pos


def _write_directory(f, directory: dict[str, Any], dir_offset_pos: int):
    """Write JSON directory at end and patch header."""
    dir_offset = f.tell()
    f.write(json.dumps(directory).encode("utf-8"))

    # Patch the header with actual directory offset
    f.seek(dir_offset_pos)
    f.write(struct.pack("<Q", dir_offset))


def write_cohort_parquet(path: str, table: pa.Table, metadata: dict[str, Any] | None = None):
    """Write .cohort file with Parquet data block (current format)."""
    with open(path, "wb") as f:
        dir_offset_pos = _write_header(f)
        data_offset = f.tell()

        # Write Parquet data
        pq.write_table(table, f, compression="zstd")
        data_length = f.tell() - data_offset

        # Build directory
        directory = {
            "tags": [{
                "tag_id": 101,
                "label": "data",
                "codec": "parquet",
                "offset": data_offset,
                "length": data_length,
                "num_rows": table.num_rows,
                "num_columns": len(table.schema),
            }],
            "cohort_definition": metadata or {"name": "benchmark", "revision": 1},
        }

        _write_directory(f, directory, dir_offset_pos)


def write_cohort_arrow(path: str, table: pa.Table, metadata: dict[str, Any] | None = None):
    """Write .cohorta file with Arrow IPC data block (fast reads, larger files)."""
    with open(path, "wb") as f:
        dir_offset_pos = _write_header(f)
        data_offset = f.tell()

        # Write Arrow IPC data
        writer = ipc.new_file(f, table.schema)
        writer.write_table(table)
        writer.close()
        data_length = f.tell() - data_offset

        # Build directory
        directory = {
            "tags": [{
                "tag_id": 101,
                "label": "data",
                "codec": "arrow_ipc",
                "offset": data_offset,
                "length": data_length,
                "num_rows": table.num_rows,
                "num_columns": len(table.schema),
            }],
            "cohort_definition": metadata or {"name": "benchmark", "revision": 1},
        }

        _write_directory(f, directory, dir_offset_pos)


def write_cohort_feather(path: str, table: pa.Table, metadata: dict[str, Any] | None = None):
    """Write .cohortf file with Feather data block (compressed Arrow)."""
    import tempfile
    import os

    with open(path, "wb") as f:
        dir_offset_pos = _write_header(f)
        data_offset = f.tell()

        # Feather needs to write to a file, so we write to temp then copy
        with tempfile.NamedTemporaryFile(delete=False, suffix=".feather") as tmp:
            tmp_path = tmp.name

        try:
            feather.write_feather(table, tmp_path, compression="zstd")
            with open(tmp_path, "rb") as tmp_f:
                data = tmp_f.read()
                f.write(data)
            data_length = len(data)
        finally:
            os.unlink(tmp_path)

        # Build directory
        directory = {
            "tags": [{
                "tag_id": 101,
                "label": "data",
                "codec": "feather",
                "offset": data_offset,
                "length": data_length,
                "num_rows": table.num_rows,
                "num_columns": len(table.schema),
            }],
            "cohort_definition": metadata or {"name": "benchmark", "revision": 1},
        }

        _write_directory(f, directory, dir_offset_pos)


def read_cohort_any(path: str) -> tuple[pa.Table, dict[str, Any]]:
    """Read any cohort variant file, returns (table, directory)."""
    with open(path, "rb") as f:
        # Read header
        magic = f.read(6)
        if magic != MAGIC:
            raise ValueError(f"Invalid magic: {magic}")

        version = struct.unpack("B", f.read(1))[0]
        dir_offset = struct.unpack("<Q", f.read(8))[0]

        # Read directory
        f.seek(dir_offset)
        directory = json.loads(f.read().decode("utf-8"))

        # Read data based on codec
        tag = directory["tags"][0]
        codec = tag["codec"]
        offset = tag["offset"]
        length = tag["length"]

        f.seek(offset)
        data_bytes = f.read(length)

        if codec == "parquet":
            import io
            table = pq.read_table(io.BytesIO(data_bytes))
        elif codec == "arrow_ipc":
            import io
            reader = ipc.open_file(io.BytesIO(data_bytes))
            table = reader.read_all()
        elif codec == "feather":
            import io
            table = feather.read_table(io.BytesIO(data_bytes))
        else:
            raise ValueError(f"Unknown codec: {codec}")

        return table, directory


def inspect_cohort_any(path: str) -> dict[str, Any]:
    """Read just the metadata from any cohort variant (O(1) operation)."""
    with open(path, "rb") as f:
        magic = f.read(6)
        if magic != MAGIC:
            raise ValueError(f"Invalid magic: {magic}")

        version = struct.unpack("B", f.read(1))[0]
        dir_offset = struct.unpack("<Q", f.read(8))[0]

        f.seek(dir_offset)
        directory = json.loads(f.read().decode("utf-8"))

        # Add file size
        import os
        directory["file_size"] = os.path.getsize(path)
        directory["codec"] = directory["tags"][0]["codec"] if directory.get("tags") else "unknown"

        return directory


def write_cohort_roaring(path: str, table: pa.Table, id_column: str = "PATIENT_NUMBER", metadata: dict[str, Any] | None = None):
    """Write .cohortr file with Roaring bitmap for patient IDs (fast set ops)."""
    from pyroaring import BitMap

    # Extract integer IDs for roaring bitmap
    if id_column in table.column_names:
        ids = table.column(id_column).to_pylist()
    else:
        # Fallback to first int column
        for i in range(len(table.schema)):
            if "int" in str(table.schema.field(i).type).lower():
                ids = table.column(i).to_pylist()
                break
        else:
            ids = list(range(table.num_rows))  # fallback to row indices

    # Create roaring bitmap
    bitmap = BitMap(ids)
    bitmap_bytes = bitmap.serialize()

    with open(path, "wb") as f:
        dir_offset_pos = _write_header(f)
        data_offset = f.tell()

        # Write roaring bitmap bytes
        f.write(bitmap_bytes)
        data_length = len(bitmap_bytes)

        # Build directory
        directory = {
            "tags": [{
                "tag_id": 101,
                "label": "patient_ids",
                "codec": "roaring",
                "offset": data_offset,
                "length": data_length,
                "num_rows": len(ids),
                "num_columns": 1,
                "id_column": id_column,
            }],
            "cohort_definition": metadata or {"name": "benchmark", "revision": 1},
        }

        _write_directory(f, directory, dir_offset_pos)


def read_roaring_ids(path: str) -> set[int]:
    """Read patient IDs from a roaring bitmap cohort file."""
    from pyroaring import BitMap

    with open(path, "rb") as f:
        magic = f.read(6)
        if magic != MAGIC:
            raise ValueError(f"Invalid magic: {magic}")

        version = struct.unpack("B", f.read(1))[0]
        dir_offset = struct.unpack("<Q", f.read(8))[0]

        f.seek(dir_offset)
        directory = json.loads(f.read().decode("utf-8"))

        tag = directory["tags"][0]
        if tag["codec"] != "roaring":
            raise ValueError(f"Expected roaring codec, got {tag['codec']}")

        f.seek(tag["offset"])
        bitmap_bytes = f.read(tag["length"])

        bitmap = BitMap.deserialize(bitmap_bytes)
        return set(bitmap)


# Format registry
FORMATS = {
    "parquet": {
        "extension": ".cohort",
        "writer": write_cohort_parquet,
        "description": "Parquet (ZSTD) — best compression",
    },
    "arrow": {
        "extension": ".cohorta",
        "writer": write_cohort_arrow,
        "description": "Arrow IPC — fastest reads, larger files",
    },
    "feather": {
        "extension": ".cohortf",
        "writer": write_cohort_feather,
        "description": "Feather (ZSTD) — balanced speed/size",
    },
    "roaring": {
        "extension": ".cohortr",
        "writer": write_cohort_roaring,
        "description": "Roaring bitmap — blazing fast AND/OR/NOT",
    },
}

"""Atlas Kohort Format - Full .kohort files with all components.

Creates TIFF-style .kohort files that include:
- Header (15 bytes): Magic "COHORT" + Version + Dir Offset
- Tag 101+: Data modalities (Parquet)
- Tag 200: PatientMembership (Roaring bitmap)
- Tag 300: VectorIndex (Lance pointer)
- Tag Directory (JSON) at end

This is the COMPLETE format as specified in docs/ARCHITECTURE.md
"""

from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pyroaring import BitMap


MAGIC = b"COHORT"
VERSION = 3

# Tag ID ranges (from ARCHITECTURE.md)
TAG_DATA_START = 101  # 100-199: Data modalities
TAG_PATIENT_MEMBERSHIP = 200  # Roaring bitmap
TAG_VECTOR_INDEX = 300  # Lance pointer


@dataclass
class TagInfo:
    """Information about a tag in the .kohort file."""
    tag_id: int
    label: str
    codec: str
    offset: int
    length: int
    # Optional fields based on codec
    num_rows: int | None = None
    num_columns: int | None = None
    columns: list[dict] | None = None
    path: str | None = None  # For lance_ptr
    id_column: str | None = None  # For roaring


@dataclass
class Selector:
    """A selector in the cohort definition."""
    selector_id: str
    field: str
    operator: str
    values: list[Any]
    is_exclusion: bool = False
    code_system: str | None = None


@dataclass
class Lineage:
    """Lineage information for a cohort."""
    source_system: str
    source_tables: list[str]
    query_hash: str | None = None
    extraction_date: str | None = None


@dataclass
class CohortDefinition:
    """Full cohort definition."""
    revision: int
    name: str
    description: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = "atlas"
    logic_operator: str = "AND"
    selectors: list[dict] = field(default_factory=list)
    lineage: dict | None = None
    revision_history: list[dict] = field(default_factory=list)


class AtlasKohortWriter:
    """Write full .kohort files with all components.

    Usage:
        writer = AtlasKohortWriter("my_cohort.kohort")
        writer.add_parquet_block("Claims", claims_table, tag_id=101)
        writer.add_parquet_block("Rx", rx_table, tag_id=102)
        writer.add_roaring_bitmap(patient_ids, id_column="PATIENT_NUMBER")
        writer.add_lance_pointer("./vectors.lance")
        writer.set_cohort_definition(name="My Cohort", ...)
        writer.finalize()
    """

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.tags: list[TagInfo] = []
        self.cohort_definition: CohortDefinition | None = None
        self.patient_index: dict | None = None
        self._file = None
        self._dir_offset_pos: int | None = None

    def __enter__(self):
        self._file = open(self.output_path, "wb")
        self._write_header()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._write_directory()
        if self._file:
            self._file.close()
        return False

    def _write_header(self):
        """Write the 15-byte header."""
        self._file.write(MAGIC)  # 6 bytes
        self._file.write(struct.pack("B", VERSION))  # 1 byte
        self._dir_offset_pos = self._file.tell()
        self._file.write(struct.pack("<Q", 0))  # 8 bytes placeholder

    def add_parquet_block(
        self,
        label: str,
        table: pa.Table,
        tag_id: int | None = None,
    ) -> TagInfo:
        """Add a Parquet data block.

        Args:
            label: Label for this modality (e.g., "Claims", "Rx")
            table: PyArrow table with the data
            tag_id: Optional tag ID (auto-assigned from 101+ if not provided)
        """
        if tag_id is None:
            # Find next available tag ID
            existing_data_tags = [t.tag_id for t in self.tags if 100 <= t.tag_id < 200]
            tag_id = max(existing_data_tags, default=100) + 1

        offset = self._file.tell()

        # Write Parquet data
        pq.write_table(table, self._file, compression="zstd")
        length = self._file.tell() - offset

        # Build column info
        columns = []
        for i in range(len(table.schema)):
            f = table.schema.field(i)
            columns.append({
                "name": f.name,
                "data_type": str(f.type),
                "nullable": f.nullable,
            })

        tag = TagInfo(
            tag_id=tag_id,
            label=label,
            codec="parquet",
            offset=offset,
            length=length,
            num_rows=table.num_rows,
            num_columns=len(table.schema),
            columns=columns,
        )
        self.tags.append(tag)
        return tag

    def add_roaring_bitmap(
        self,
        patient_ids: list[int] | set[int] | BitMap,
        id_column: str = "PATIENT_NUMBER",
    ) -> TagInfo:
        """Add the patient membership roaring bitmap (Tag 200).

        Args:
            patient_ids: Collection of patient IDs
            id_column: Name of the ID column used
        """
        # Convert to BitMap if needed
        if isinstance(patient_ids, BitMap):
            bitmap = patient_ids
        else:
            bitmap = BitMap(patient_ids)

        offset = self._file.tell()
        bitmap_bytes = bitmap.serialize()
        self._file.write(bitmap_bytes)
        length = len(bitmap_bytes)

        tag = TagInfo(
            tag_id=TAG_PATIENT_MEMBERSHIP,
            label="PatientMembership",
            codec="roaring",
            offset=offset,
            length=length,
            num_rows=len(bitmap),
            id_column=id_column,
        )
        self.tags.append(tag)

        # Also store patient_index for directory
        self.patient_index = {
            "offset": offset,
            "length": length,
            "cardinality": len(bitmap),
        }

        return tag

    def add_lance_pointer(self, lance_path: str) -> TagInfo:
        """Add a pointer to a Lance vector index (Tag 300).

        Note: This doesn't embed the Lance data, just stores a pointer.

        Args:
            lance_path: Relative or absolute path to the Lance directory
        """
        # Write the path as bytes (we need some content for the tag)
        offset = self._file.tell()
        path_bytes = lance_path.encode("utf-8")
        self._file.write(path_bytes)
        length = len(path_bytes)

        tag = TagInfo(
            tag_id=TAG_VECTOR_INDEX,
            label="VectorIndex",
            codec="lance_ptr",
            offset=offset,
            length=length,
            path=lance_path,
        )
        self.tags.append(tag)
        return tag

    def set_cohort_definition(
        self,
        name: str,
        description: str | None = None,
        selectors: list[dict] | None = None,
        lineage: dict | None = None,
        revision: int = 1,
        created_by: str = "atlas",
        revision_history: list[dict] | None = None,
    ):
        """Set the cohort definition metadata."""
        self.cohort_definition = CohortDefinition(
            revision=revision,
            name=name,
            description=description,
            created_by=created_by,
            selectors=selectors or [],
            lineage=lineage,
            revision_history=revision_history or [],
        )

    def _write_directory(self):
        """Write the JSON tag directory at end and patch header."""
        dir_offset = self._file.tell()

        # Build directory
        directory = {
            "tags": [],
            "cohort_definition": asdict(self.cohort_definition) if self.cohort_definition else {},
        }

        if self.patient_index:
            directory["patient_index"] = self.patient_index

        # Add tags
        for tag in self.tags:
            tag_dict = {
                "tag_id": tag.tag_id,
                "label": tag.label,
                "codec": tag.codec,
                "offset": tag.offset,
                "length": tag.length,
            }
            if tag.num_rows is not None:
                tag_dict["num_rows"] = tag.num_rows
            if tag.num_columns is not None:
                tag_dict["num_columns"] = tag.num_columns
            if tag.columns is not None:
                tag_dict["columns"] = tag.columns
            if tag.path is not None:
                tag_dict["path"] = tag.path
            if tag.id_column is not None:
                tag_dict["id_column"] = tag.id_column
            directory["tags"].append(tag_dict)

        # Write directory
        self._file.write(json.dumps(directory, indent=2).encode("utf-8"))

        # Patch header with directory offset
        self._file.seek(self._dir_offset_pos)
        self._file.write(struct.pack("<Q", dir_offset))

    def finalize(self):
        """Finalize the file (called automatically when using context manager)."""
        if self._file:
            self._write_directory()
            self._file.close()
            self._file = None


class AtlasKohortReader:
    """Read full .kohort files with all components.

    Usage:
        reader = AtlasKohortReader("my_cohort.kohort")

        # Get metadata
        print(reader.tags)
        print(reader.cohort_definition)

        # Get data
        claims_df = reader.get_modality("Claims")
        bitmap = reader.get_membership_bitmap()
        lance_path = reader.get_vector_index_path()
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = None
        self.version: int = 0
        self.dir_offset: int = 0
        self.directory: dict = {}
        self.tags: list[dict] = []
        self.cohort_definition: dict = {}
        self.patient_index: dict | None = None

        self._read_header_and_directory()

    def _read_header_and_directory(self):
        """Read the header and tag directory."""
        with open(self.path, "rb") as f:
            # Read header
            magic = f.read(6)
            if magic != MAGIC:
                raise ValueError(f"Invalid magic: {magic}, expected {MAGIC}")

            self.version = struct.unpack("B", f.read(1))[0]
            self.dir_offset = struct.unpack("<Q", f.read(8))[0]

            # Read directory
            f.seek(self.dir_offset)
            self.directory = json.loads(f.read().decode("utf-8"))

        self.tags = self.directory.get("tags", [])
        self.cohort_definition = self.directory.get("cohort_definition", {})
        self.patient_index = self.directory.get("patient_index")

    def get_tag(self, tag_id: int) -> dict | None:
        """Get a tag by ID."""
        for tag in self.tags:
            if tag["tag_id"] == tag_id:
                return tag
        return None

    def get_tag_by_label(self, label: str) -> dict | None:
        """Get a tag by label."""
        label_lower = label.lower()
        for tag in self.tags:
            if tag["label"].lower() == label_lower:
                return tag
        return None

    def get_modality(self, label: str) -> pa.Table | None:
        """Get a data modality by label."""
        tag = self.get_tag_by_label(label)
        if tag is None or tag["codec"] != "parquet":
            return None

        with open(self.path, "rb") as f:
            f.seek(tag["offset"])
            data = f.read(tag["length"])

        return pq.read_table(io.BytesIO(data))

    def get_modality_by_tag_id(self, tag_id: int) -> pa.Table | None:
        """Get a data modality by tag ID."""
        tag = self.get_tag(tag_id)
        if tag is None or tag["codec"] != "parquet":
            return None

        with open(self.path, "rb") as f:
            f.seek(tag["offset"])
            data = f.read(tag["length"])

        return pq.read_table(io.BytesIO(data))

    def get_membership_bitmap(self) -> BitMap | None:
        """Get the patient membership roaring bitmap (Tag 200)."""
        tag = self.get_tag(TAG_PATIENT_MEMBERSHIP)
        if tag is None:
            return None

        with open(self.path, "rb") as f:
            f.seek(tag["offset"])
            data = f.read(tag["length"])

        return BitMap.deserialize(data)

    def get_vector_index_path(self) -> str | None:
        """Get the path to the Lance vector index (Tag 300)."""
        tag = self.get_tag(TAG_VECTOR_INDEX)
        if tag is None:
            return None

        # Check if path is in tag metadata
        if "path" in tag:
            return tag["path"]

        # Otherwise read from file
        with open(self.path, "rb") as f:
            f.seek(tag["offset"])
            data = f.read(tag["length"])

        return data.decode("utf-8")

    def list_modalities(self) -> list[str]:
        """List all data modalities (Tag 101-199)."""
        return [
            tag["label"]
            for tag in self.tags
            if 100 <= tag["tag_id"] < 200
        ]

    def has_membership_bitmap(self) -> bool:
        """Check if the file has a membership bitmap."""
        return self.get_tag(TAG_PATIENT_MEMBERSHIP) is not None

    def has_vector_index(self) -> bool:
        """Check if the file has a vector index pointer."""
        return self.get_tag(TAG_VECTOR_INDEX) is not None

    def summary(self) -> dict:
        """Get a summary of the .kohort file contents."""
        data_tags = [t for t in self.tags if 100 <= t["tag_id"] < 200]
        membership_tag = self.get_tag(TAG_PATIENT_MEMBERSHIP)
        vector_tag = self.get_tag(TAG_VECTOR_INDEX)

        return {
            "path": str(self.path),
            "version": self.version,
            "name": self.cohort_definition.get("name", "Unknown"),
            "modalities": [t["label"] for t in data_tags],
            "total_rows": sum(t.get("num_rows", 0) for t in data_tags),
            "has_membership": membership_tag is not None,
            "membership_cardinality": membership_tag.get("num_rows") if membership_tag else None,
            "has_vector_index": vector_tag is not None,
            "vector_path": vector_tag.get("path") if vector_tag else None,
            "file_size": self.path.stat().st_size,
        }


def inspect_kohort(path: str | Path) -> dict:
    """Quick inspection of a .kohort file without loading data."""
    reader = AtlasKohortReader(path)
    return reader.summary()

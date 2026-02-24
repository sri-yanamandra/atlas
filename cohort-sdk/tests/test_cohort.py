"""Tests for cohort-sdk: round-trip IO and revision operations."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pytest

from cohort_sdk import (
    CohortDefinition,
    CohortInspection,
    CohortReader,
    CohortWriter,
    Delta,
    Lineage,
    RevisionEntry,
    Selector,
    TagDirectory,
    inspect,
    intersect,
    merge,
    read,
    union,
    write,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _sample_table() -> pa.Table:
    return pa.table({
        "patient_id": pa.array([1, 2, 3], type=pa.int32()),
        "diagnosis": pa.array(["E11.0", "E11.1", "E11.9"]),
    })


def _sample_directory() -> TagDirectory:
    now = datetime.now(timezone.utc)
    return TagDirectory(
        cohort_definition=CohortDefinition(
            name="Test Cohort",
            description="A test cohort",
            created_at=now,
            created_by="user:test",
            selectors=[
                Selector(
                    selector_id="sel-dx",
                    field="diagnosis_code",
                    code_system="ICD-10-CM",
                    operator="in",
                    values=["E11.0", "E11.1", "E11.9"],
                    label="Type 2 Diabetes",
                )
            ],
            lineage=Lineage(
                source_system=["Test Warehouse"],
                source_tables=["claims"],
                extraction_date="2026-02-10",
            ),
            revision_history=[
                RevisionEntry(
                    revision=1,
                    timestamp=now,
                    author="user:test",
                    message="Initial cohort",
                )
            ],
        )
    )


def test_round_trip():
    """Test simple read/write top-level API."""
    table = _sample_table()
    defn = _sample_directory().cohort_definition

    with tempfile.NamedTemporaryFile(suffix=".cohort") as f:
        path = f.name
        write(path, defn, [("Claims-v1", table)])

        cohort = read(path)

        assert cohort.definition.revision == 1
        assert cohort.definition.name == "Test Cohort"
        assert cohort.definition.logic_operator == "AND"
        assert len(cohort.definition.selectors) == 1
        assert cohort.definition.selectors[0].selector_id == "sel-dx"
        assert cohort.definition.lineage.source_system == ["Test Warehouse"]

        assert cohort.datasets == ["Claims-v1"]

        result = cohort["Claims-v1"]
        assert result.num_rows == 3
        assert result.num_columns == 2


def test_multiple_datasets():
    table1 = _sample_table()
    table2 = pa.table({
        "patient_id": pa.array([1, 2], type=pa.int32()),
        "ndc": pa.array(["12345-6789", "98765-4321"]),
    })
    defn = _sample_directory().cohort_definition

    with tempfile.NamedTemporaryFile(suffix=".cohort") as f:
        path = f.name
        write(path, defn, [("Claims-v1", table1), ("Rx-v1", table2)])

        cohort = read(path)
        assert cohort.datasets == ["Claims-v1", "Rx-v1"]

        assert cohort["Claims-v1"].num_rows == 3
        assert cohort["Rx-v1"].num_rows == 2


def test_cohort_definition_defaults():
    defn = CohortDefinition(name="My Cohort")
    assert defn.revision == 1
    assert defn.logic_operator == "AND"
    assert defn.selectors == []
    assert defn.revision_history == []
    assert defn.lineage.source_system == []


def test_apply_revision_add():
    defn = CohortDefinition(
        name="Test",
        selectors=[
            Selector(
                selector_id="sel-1",
                field="diagnosis_code",
                operator="in",
                values=["E11.0"],
            )
        ],
    )

    new_sel = Selector(
        selector_id="sel-2",
        field="patient_age",
        operator="between",
        values=[40, 75],
        label="Age range",
    )

    defn.apply_revision(RevisionEntry(
        revision=2,
        timestamp=datetime.now(timezone.utc),
        author="user:test",
        message="Add age filter",
        deltas=[Delta(action="add", selector_id="sel-2", selector=new_sel)],
    ))

    assert defn.revision == 2
    assert len(defn.selectors) == 2
    assert defn.selectors[1].selector_id == "sel-2"


def test_apply_revision_modify():
    defn = CohortDefinition(
        name="Test",
        selectors=[
            Selector(
                selector_id="sel-age",
                field="patient_age",
                operator="between",
                values=[45, 75],
            )
        ],
    )

    defn.apply_revision(RevisionEntry(
        revision=2,
        timestamp=datetime.now(timezone.utc),
        author="user:test",
        message="Broaden age range",
        deltas=[
            Delta(
                action="modify",
                selector_id="sel-age",
                field="values",
                before=[45, 75],
                after=[40, 75],
            )
        ],
    ))

    assert defn.revision == 2
    assert defn.selectors[0].values == [40, 75]


def test_apply_revision_remove():
    defn = CohortDefinition(
        name="Test",
        selectors=[
            Selector(
                selector_id="sel-state",
                field="patient_state",
                operator="in",
                values=["CA"],
            )
        ],
    )

    defn.apply_revision(RevisionEntry(
        revision=2,
        timestamp=datetime.now(timezone.utc),
        author="user:test",
        message="Remove state filter",
        deltas=[Delta(action="remove", selector_id="sel-state")],
    ))

    assert defn.revision == 2
    assert defn.selectors == []


def test_apply_revision_wrong_version():
    defn = CohortDefinition(name="Test")

    try:
        defn.apply_revision(RevisionEntry(
            revision=5,
            timestamp=datetime.now(timezone.utc),
            author="user:test",
            message="Bad revision",
        ))
        assert False, "Should have raised"
    except ValueError as e:
        assert "Expected revision 2, got 5" in str(e)


def test_invalid_magic():
    with tempfile.NamedTemporaryFile(suffix=".cohort") as f:
        f.write(b"NOTCOH")
        f.flush()
        try:
            CohortReader.open(f.name)
            assert False, "Should have raised"
        except ValueError as e:
            assert "magic" in str(e).lower()


def test_invalid_version():
    with tempfile.NamedTemporaryFile(suffix=".cohort") as f:
        import struct
        f.write(b"COHORT")
        f.write(struct.pack("B", 99))
        f.write(struct.pack("<Q", 0))
        f.flush()
        try:
            CohortReader.open(f.name)
            assert False, "Should have raised"
        except ValueError as e:
            assert "version" in str(e).lower()


def test_cross_compat_with_rust():
    """Verify the Python SDK writes files the Rust crate can understand (same binary layout)."""
    table = _sample_table()
    directory = _sample_directory()

    with tempfile.NamedTemporaryFile(suffix=".cohort") as f:
        path = f.name
        CohortWriter.write(path, directory, [("Claims-v1", table)])

        # Manually verify binary layout
        with open(path, "rb") as raw:
            import struct as st
            magic = raw.read(6)
            assert magic == b"COHORT"

            version = st.unpack("B", raw.read(1))[0]
            assert version == 3

            dir_offset = st.unpack("<Q", raw.read(8))[0]
            assert dir_offset > 15  # data blocks exist

            # Jump to directory, verify it's valid JSON
            raw.seek(dir_offset)
            import json
            dir_data = json.loads(raw.read())
            assert "tags" in dir_data
            assert "cohort_definition" in dir_data
            assert dir_data["cohort_definition"]["name"] == "Test Cohort"


# ── Inspect tests ──────────────────────────────────────────────


def test_inspect_real_cohort():
    """Inspect f.cohort fixture — verify name, dataset, and row count."""
    info = CohortDefinition.inspect(FIXTURES / "f.cohort")

    assert isinstance(info, CohortInspection)
    assert info.cohort_definition.name == "Female Patients"
    assert info.file_metadata.format_version == 3
    assert info.file_metadata.num_datasets == 1
    assert info.file_metadata.file_size_bytes > 0

    assert len(info.datasets) == 1
    mod = info.datasets[0]
    assert mod.label == "Patients-v1"
    assert mod.num_rows > 0
    assert mod.num_columns > 0
    assert len(mod.columns) == mod.num_columns


def test_inspect_both_cohorts():
    """Inspect both fixtures — verify different names."""
    f_info = CohortDefinition.inspect(FIXTURES / "f.cohort")
    m_info = CohortDefinition.inspect(FIXTURES / "m.cohort")

    assert f_info.cohort_definition.name == "Female Patients"
    assert m_info.cohort_definition.name == "Male Patients"
    assert f_info.datasets[0].label == m_info.datasets[0].label == "Patients-v1"


def test_inspect_top_level():
    """Top-level inspect() convenience function."""
    info = inspect(FIXTURES / "f.cohort")
    assert info.cohort_definition.name == "Female Patients"


# ── Union tests ────────────────────────────────────────────────


def test_union_real_cohorts():
    """Union m.cohort + f.cohort, verify combined row count."""
    f_info = CohortDefinition.inspect(FIXTURES / "f.cohort")
    m_info = CohortDefinition.inspect(FIXTURES / "m.cohort")
    expected_rows = f_info.datasets[0].num_rows + m_info.datasets[0].num_rows

    with tempfile.NamedTemporaryFile(suffix=".cohort") as f:
        union(
            [FIXTURES / "m.cohort", FIXTURES / "f.cohort"],
            f.name,
            name="All Patients",
        )

        cohort = read(f.name)
        assert cohort.definition.name == "All Patients"
        assert cohort.datasets == ["Patients-v1"]
        assert cohort["Patients-v1"].num_rows == expected_rows


def test_union_then_inspect():
    """Union then inspect the result — verify combined stats."""
    with tempfile.NamedTemporaryFile(suffix=".cohort") as f:
        union(
            [FIXTURES / "m.cohort", FIXTURES / "f.cohort"],
            f.name,
            name="Combined",
        )

        info = CohortDefinition.inspect(f.name)
        assert info.cohort_definition.name == "Combined"
        assert info.file_metadata.num_datasets == 1
        assert info.datasets[0].label == "Patients-v1"

        f_info = CohortDefinition.inspect(FIXTURES / "f.cohort")
        m_info = CohortDefinition.inspect(FIXTURES / "m.cohort")
        assert info.datasets[0].num_rows == (
            f_info.datasets[0].num_rows + m_info.datasets[0].num_rows
        )


def test_union_generates_name():
    """Union without name= generates a descriptive name from file basenames."""
    with tempfile.NamedTemporaryFile(suffix=".cohort") as f:
        union(
            [FIXTURES / "m.cohort", FIXTURES / "f.cohort"],
            f.name,
        )
        cohort = read(f.name)
        assert cohort.definition.name.startswith("Union of")
        assert "m.cohort" in cohort.definition.name
        assert "f.cohort" in cohort.definition.name
        assert cohort["Patients-v1"].num_rows > 0


def test_merge_is_union_alias():
    """merge() is an alias for union()."""
    assert merge is union


def test_union_merged_history():
    """Union builds a merged definition with history from all inputs."""
    with tempfile.NamedTemporaryFile(suffix=".cohort") as f:
        union(
            [FIXTURES / "m.cohort", FIXTURES / "f.cohort"],
            f.name,
            name="Combined",
        )
        cohort = read(f.name)
        assert cohort.definition.name == "Combined"
        # Should have history entries from inputs plus the new operation entry
        assert len(cohort.definition.revision_history) >= 1
        last = cohort.definition.revision_history[-1]
        assert "Union of" in last.message
        assert last.author == "cohort-sdk"


def test_union_schema_mismatch():
    """Union with different schemas for the same label raises ValueError."""
    table1 = pa.table({"patient_id": pa.array([1, 2], type=pa.int32())})
    table2 = pa.table({"patient_id": pa.array(["A", "B"])})

    with tempfile.NamedTemporaryFile(suffix=".cohort") as f1, \
         tempfile.NamedTemporaryFile(suffix=".cohort") as f2, \
         tempfile.NamedTemporaryFile(suffix=".cohort") as out:

        CohortDefinition(name="A").write(f1.name, [("Data", table1)])
        CohortDefinition(name="B").write(f2.name, [("Data", table2)])

        with pytest.raises(ValueError, match="Schema mismatch"):
            union([f1.name, f2.name], out.name)


# ── Intersect tests ───────────────────────────────────────────


def test_intersect_shared_patients():
    """Intersect keeps only rows with shared keys."""
    table1 = pa.table({
        "patient_id": pa.array([1, 2, 3], type=pa.int32()),
        "diagnosis": pa.array(["E11.0", "E11.1", "E11.9"]),
    })
    table2 = pa.table({
        "patient_id": pa.array([2, 3, 4], type=pa.int32()),
        "diagnosis": pa.array(["E11.1", "E11.9", "E11.0"]),
    })

    with tempfile.NamedTemporaryFile(suffix=".cohort") as f1, \
         tempfile.NamedTemporaryFile(suffix=".cohort") as f2, \
         tempfile.NamedTemporaryFile(suffix=".cohort") as out:

        CohortDefinition(name="A").write(f1.name, [("Claims", table1)])
        CohortDefinition(name="B").write(f2.name, [("Claims", table2)])

        intersect(
            [f1.name, f2.name], out.name,
            on="patient_id", name="Shared",
        )

        cohort = read(out.name)
        assert cohort.definition.name == "Shared"
        result = cohort["Claims"]
        # Keys 2, 3 are shared — rows from both files, so 2+2 = 4
        assert result.num_rows == 4


def test_intersect_no_overlap():
    """Intersect with disjoint keys produces empty output."""
    table1 = pa.table({"patient_id": pa.array([1, 2], type=pa.int32())})
    table2 = pa.table({"patient_id": pa.array([3, 4], type=pa.int32())})

    with tempfile.NamedTemporaryFile(suffix=".cohort") as f1, \
         tempfile.NamedTemporaryFile(suffix=".cohort") as f2, \
         tempfile.NamedTemporaryFile(suffix=".cohort") as out:

        CohortDefinition(name="A").write(f1.name, [("Data", table1)])
        CohortDefinition(name="B").write(f2.name, [("Data", table2)])

        intersect([f1.name, f2.name], out.name, on="patient_id")

        cohort = read(out.name)
        assert cohort.datasets == []


def test_intersect_merged_history():
    """Intersect builds a merged definition with history."""
    table1 = pa.table({
        "patient_id": pa.array([1, 2, 3], type=pa.int32()),
        "diagnosis": pa.array(["E11.0", "E11.1", "E11.9"]),
    })
    table2 = pa.table({
        "patient_id": pa.array([2, 3, 4], type=pa.int32()),
        "diagnosis": pa.array(["E11.1", "E11.9", "E11.0"]),
    })

    with tempfile.NamedTemporaryFile(suffix=".cohort") as f1, \
         tempfile.NamedTemporaryFile(suffix=".cohort") as f2, \
         tempfile.NamedTemporaryFile(suffix=".cohort") as out:

        CohortDefinition(name="A").write(f1.name, [("Claims", table1)])
        CohortDefinition(name="B").write(f2.name, [("Claims", table2)])

        intersect(
            [f1.name, f2.name], out.name,
            on="patient_id", name="Shared",
        )

        cohort = read(out.name)
        assert cohort.definition.name == "Shared"
        last = cohort.definition.revision_history[-1]
        assert "Intersect of" in last.message
        assert last.author == "cohort-sdk"


def test_intersect_real_cohorts():
    """Intersect real fixture files on PATIENT_NUMBER — M/F are disjoint by gender."""
    with tempfile.NamedTemporaryFile(suffix=".cohort") as out:
        intersect(
            [FIXTURES / "m.cohort", FIXTURES / "f.cohort"],
            out.name,
            on="PATIENT_NUMBER",
            name="Shared Patients",
        )

        info = CohortDefinition.inspect(out.name)
        assert info.cohort_definition.name == "Shared Patients"
        # M and F are split by gender — no shared patients
        assert len(info.datasets) == 0


# ── Primary keys tests ──────────────────────────────────────────


def test_primary_keys_field_default():
    """CohortDefinition defaults primary_keys to empty list."""
    defn = CohortDefinition(name="Test")
    assert defn.primary_keys == []


def test_primary_keys_round_trip():
    """primary_keys survive write/read round-trip through the Rust binary format."""
    table = _sample_table()
    now = datetime.now(timezone.utc)
    defn = CohortDefinition(
        name="Keyed Cohort",
        created_at=now,
        created_by="test",
        primary_keys=["patient_id"],
        lineage=Lineage(
            source_system=["Test"],
            source_tables=["patients"],
            extraction_date="2026-02-11",
        ),
        revision_history=[
            RevisionEntry(
                revision=1, timestamp=now, author="test", message="Initial",
            )
        ],
    )

    with tempfile.NamedTemporaryFile(suffix=".cohort") as f:
        path = f.name
        write(path, defn, [("Patients", table)])

        cohort = read(path)
        assert cohort.definition.primary_keys == ["patient_id"]
        assert cohort.definition.name == "Keyed Cohort"


def test_primary_keys_inspect():
    """primary_keys appear in inspect output."""
    table = _sample_table()
    now = datetime.now(timezone.utc)
    defn = CohortDefinition(
        name="Keyed Cohort",
        created_at=now,
        created_by="test",
        primary_keys=["patient_id", "diagnosis"],
        lineage=Lineage(
            source_system=["Test"],
            source_tables=["patients"],
            extraction_date="2026-02-11",
        ),
    )

    with tempfile.NamedTemporaryFile(suffix=".cohort") as f:
        path = f.name
        write(path, defn, [("Patients", table)])

        info = inspect(path)
        assert info.cohort_definition.primary_keys == ["patient_id", "diagnosis"]


def test_primary_keys_empty_round_trip():
    """Empty primary_keys survives round-trip (backward compat with old files)."""
    table = _sample_table()
    defn = CohortDefinition(name="No Keys")

    with tempfile.NamedTemporaryFile(suffix=".cohort") as f:
        path = f.name
        write(path, defn, [("Data", table)])

        cohort = read(path)
        assert cohort.definition.primary_keys == []


# ── Iceberg upsert/append tests (mocked) ───────────────────────


def test_iceberg_upsert_no_backend_raises():
    """iceberg_upsert raises ValueError when target has no IcebergBackend."""
    from cohort_sdk import iceberg_upsert

    table = _sample_table()
    defn = CohortDefinition(name="Local Only")

    with tempfile.NamedTemporaryFile(suffix=".cohort") as target, \
         tempfile.NamedTemporaryFile(suffix=".cohort") as source:

        write(target.name, defn, [("Data", table)])
        write(source.name, defn, [("Data", table)])

        with pytest.raises(ValueError, match="no IcebergBackend"):
            iceberg_upsert(target.name, source.name, on=["patient_id"])


def test_iceberg_append_no_backend_raises():
    """iceberg_append raises ValueError when target has no IcebergBackend."""
    from cohort_sdk import iceberg_append

    table = _sample_table()
    defn = CohortDefinition(name="Local Only")

    with tempfile.NamedTemporaryFile(suffix=".cohort") as target, \
         tempfile.NamedTemporaryFile(suffix=".cohort") as source:

        write(target.name, defn, [("Data", table)])
        write(source.name, defn, [("Data", table)])

        with pytest.raises(ValueError, match="no IcebergBackend"):
            iceberg_append(target.name, source.name)


def test_iceberg_upsert_no_keys_no_flag_raises():
    """iceberg_upsert raises ValueError when no primary_keys and no --on."""
    from cohort_sdk import iceberg_upsert, CohortWriter, IcebergBackend

    table = _sample_table()
    now = datetime.now(timezone.utc)
    defn = CohortDefinition(
        name="Iceberg Target",
        created_at=now,
        created_by="test",
        primary_keys=[],  # no primary keys
    )
    directory = TagDirectory(
        cohort_definition=defn,
        backend=IcebergBackend(
            database="test_db",
            table_name="test_table",
            table_location="s3://bucket/path",
        ),
    )

    with tempfile.NamedTemporaryFile(suffix=".cohort") as target, \
         tempfile.NamedTemporaryFile(suffix=".cohort") as source:

        CohortWriter.write(target.name, directory, [])
        write(source.name, CohortDefinition(name="Source"), [("Data", table)])

        with pytest.raises(ValueError, match="No primary keys"):
            iceberg_upsert(target.name, source.name)

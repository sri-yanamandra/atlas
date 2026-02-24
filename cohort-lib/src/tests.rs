use std::sync::Arc;

use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use chrono::Utc;
use tempfile::NamedTempFile;

use crate::bitmap_index::{bitmap_can_skip_file, deserialize_bitmap, hash_key};
use crate::diff::{diff, format_diff};
use crate::inspect::inspect;
use crate::merge::{intersect, merge, union};
use crate::models::*;
use crate::reader::CohortReader;
use crate::selector_pruner::extract_key_filter_values;
use crate::writer::CohortWriter;

use datafusion::logical_expr::{col, lit, Expr};

fn sample_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("patient_id", DataType::Int32, false),
        Field::new("diagnosis", DataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["E11.0", "E11.1", "E11.9"])),
        ],
    )
    .unwrap()
}

fn sample_directory() -> TagDirectory {
    TagDirectory {
        tags: vec![], // populated by writer
        patient_index: ByteRange {
            offset: 0,
            length: 0,
        },
        backend: None,
        cohort_definition: CohortDefinition {
            revision: 1,
            name: "Test Cohort".to_string(),
            description: Some("A test cohort".to_string()),
            created_at: Utc::now(),
            created_by: "user:test".to_string(),
            logic_operator: "AND".to_string(),
            selectors: vec![Selector {
                selector_id: "sel-dx".to_string(),
                field: "diagnosis_code".to_string(),
                code_system: Some("ICD-10-CM".to_string()),
                operator: "in".to_string(),
                values: serde_json::json!(["E11.0", "E11.1", "E11.9"]),
                label: Some("Type 2 Diabetes".to_string()),
                is_exclusion: false,
            }],
            lineage: Lineage {
                source_system: vec!["Test Warehouse".to_string()],
                source_tables: vec!["claims".to_string()],
                query_hash: None,
                extraction_date: "2026-02-10".to_string(),
                generation_job_id: None,
            },
            revision_history: vec![RevisionEntry {
                revision: 1,
                timestamp: Utc::now(),
                author: "user:test".to_string(),
                message: "Initial cohort".to_string(),
                deltas: vec![],
                size: Some(3),
            }],
            primary_keys: vec![],
        },
    }
}

#[test]
fn round_trip_write_and_read() {
    let batch = sample_batch();
    let directory = sample_directory();
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    // Write
    CohortWriter::write(path, &directory, &[("Claims-v1", &batch)]).unwrap();

    // Read
    let reader = CohortReader::open(path).unwrap();

    // Verify cohort definition round-trips
    let def = reader.cohort_definition();
    assert_eq!(def.revision, 1);
    assert_eq!(def.name, "Test Cohort");
    assert_eq!(def.logic_operator, "AND");
    assert_eq!(def.selectors.len(), 1);
    assert_eq!(def.selectors[0].selector_id, "sel-dx");
    assert_eq!(def.lineage.source_system, vec!["Test Warehouse"]);
    assert_eq!(def.revision_history.len(), 1);

    // Verify tag directory
    assert_eq!(reader.directory.tags.len(), 1);
    assert_eq!(reader.directory.tags[0].label, "Claims-v1");
    assert_eq!(reader.directory.tags[0].codec, "parquet");

    // Verify data round-trips
    let batches: Vec<_> = reader
        .get_dataset_data("Claims-v1")
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 3);
    assert_eq!(batches[0].num_columns(), 2);
}

#[test]
fn round_trip_multiple_datasets() {
    let batch1 = sample_batch();

    let schema2 = Arc::new(Schema::new(vec![
        Field::new("patient_id", DataType::Int32, false),
        Field::new("ndc", DataType::Utf8, false),
    ]));
    let batch2 = RecordBatch::try_new(
        schema2,
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["12345-6789", "98765-4321"])),
        ],
    )
    .unwrap();

    let directory = sample_directory();
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    CohortWriter::write(path, &directory, &[("Claims-v1", &batch1), ("Rx-v1", &batch2)]).unwrap();

    let reader = CohortReader::open(path).unwrap();
    assert_eq!(reader.directory.tags.len(), 2);
    assert_eq!(reader.directory.tags[0].label, "Claims-v1");
    assert_eq!(reader.directory.tags[1].label, "Rx-v1");

    // Read each dataset independently
    let claims: Vec<_> = reader
        .get_dataset_data("Claims-v1")
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(claims[0].num_rows(), 3);

    let rx: Vec<_> = reader
        .get_dataset_data("Rx-v1")
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(rx[0].num_rows(), 2);
}

#[test]
fn cohort_definition_new_defaults() {
    let def = CohortDefinition::new("My Cohort");
    assert_eq!(def.revision, 1);
    assert_eq!(def.name, "My Cohort");
    assert_eq!(def.logic_operator, "AND");
    assert!(def.selectors.is_empty());
    assert!(def.revision_history.is_empty());
    assert!(def.lineage.source_system.is_empty());  // Vec is empty
}

#[test]
fn apply_revision_add_selector() {
    let mut def = CohortDefinition::new("Test");
    def.selectors.push(Selector {
        selector_id: "sel-1".into(),
        field: "diagnosis_code".into(),
        code_system: None,
        operator: "in".into(),
        values: serde_json::json!(["E11.0"]),
        label: None,
        is_exclusion: false,
    });

    let new_selector = Selector {
        selector_id: "sel-2".into(),
        field: "patient_age".into(),
        code_system: None,
        operator: "between".into(),
        values: serde_json::json!([40, 75]),
        label: Some("Age range".into()),
        is_exclusion: false,
    };

    let entry = RevisionEntry {
        revision: 2,
        timestamp: Utc::now(),
        author: "user:test".into(),
        message: "Add age filter".into(),
        deltas: vec![Delta {
            action: DeltaAction::Add,
            selector_id: "sel-2".into(),
            field: None,
            before: None,
            after: None,
            selector: Some(new_selector),
        }],
        size: None,
    };

    def.apply_revision(entry).unwrap();
    assert_eq!(def.revision, 2);
    assert_eq!(def.selectors.len(), 2);
    assert_eq!(def.selectors[1].selector_id, "sel-2");
    assert_eq!(def.revision_history.len(), 1);
}

#[test]
fn apply_revision_modify_selector() {
    let mut def = CohortDefinition::new("Test");
    def.selectors.push(Selector {
        selector_id: "sel-age".into(),
        field: "patient_age".into(),
        code_system: None,
        operator: "between".into(),
        values: serde_json::json!([45, 75]),
        label: None,
        is_exclusion: false,
    });

    let entry = RevisionEntry {
        revision: 2,
        timestamp: Utc::now(),
        author: "user:test".into(),
        message: "Broaden age range".into(),
        deltas: vec![Delta {
            action: DeltaAction::Modify,
            selector_id: "sel-age".into(),
            field: Some("values".into()),
            before: Some(serde_json::json!([45, 75])),
            after: Some(serde_json::json!([40, 75])),
            selector: None,
        }],
        size: None,
    };

    def.apply_revision(entry).unwrap();
    assert_eq!(def.revision, 2);
    assert_eq!(def.selectors[0].values, serde_json::json!([40, 75]));
}

#[test]
fn apply_revision_remove_selector() {
    let mut def = CohortDefinition::new("Test");
    def.selectors.push(Selector {
        selector_id: "sel-state".into(),
        field: "patient_state".into(),
        code_system: None,
        operator: "in".into(),
        values: serde_json::json!(["CA"]),
        label: None,
        is_exclusion: false,
    });

    let entry = RevisionEntry {
        revision: 2,
        timestamp: Utc::now(),
        author: "user:test".into(),
        message: "Remove state filter".into(),
        deltas: vec![Delta {
            action: DeltaAction::Remove,
            selector_id: "sel-state".into(),
            field: None,
            before: None,
            after: None,
            selector: None,
        }],
        size: None,
    };

    def.apply_revision(entry).unwrap();
    assert_eq!(def.revision, 2);
    assert!(def.selectors.is_empty());
}

#[test]
fn apply_revision_wrong_version() {
    let mut def = CohortDefinition::new("Test");

    let entry = RevisionEntry {
        revision: 5, // should be 2
        timestamp: Utc::now(),
        author: "user:test".into(),
        message: "Bad revision".into(),
        deltas: vec![],
        size: None,
    };

    let err = def.apply_revision(entry).unwrap_err();
    assert!(err.contains("Expected revision 2, got 5"));
}

#[test]
fn invalid_magic_rejected() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();
    std::fs::write(path, b"NOTCOH").unwrap();

    let err = CohortReader::open(path).unwrap_err();
    assert!(err.to_string().contains("magic"));
}

#[test]
fn invalid_version_rejected() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    let mut data = Vec::new();
    data.extend_from_slice(b"COHORT");
    data.push(99); // bad version
    data.extend_from_slice(&0u64.to_le_bytes());
    std::fs::write(path, &data).unwrap();

    let err = CohortReader::open(path).unwrap_err();
    assert!(err.to_string().contains("version"));
}

// ── Inspect tests ─────────────────────────────────────────────

#[test]
fn inspect_single_dataset() {
    let batch = sample_batch();
    let directory = sample_directory();
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    CohortWriter::write(path, &directory, &[("Claims-v1", &batch)]).unwrap();

    let result = inspect(path).unwrap();

    assert_eq!(result.file_metadata.format_version, 3);
    assert_eq!(result.file_metadata.num_datasets, 1);
    assert!(result.file_metadata.file_size_bytes > 0);

    assert_eq!(result.cohort_definition.name, "Test Cohort");
    assert_eq!(result.cohort_definition.revision, 1);
    assert_eq!(result.cohort_definition.selectors.len(), 1);
    assert_eq!(result.cohort_definition.selectors[0].selector_id, "sel-dx");

    assert_eq!(result.datasets.len(), 1);
    assert_eq!(result.datasets[0].label, "Claims-v1");
    assert_eq!(result.datasets[0].num_rows, 3);
    assert_eq!(result.datasets[0].num_columns, 2);
    assert_eq!(result.datasets[0].columns[0].name, "patient_id");
    assert_eq!(result.datasets[0].columns[1].name, "diagnosis");
}

#[test]
fn inspect_multiple_datasets() {
    let batch1 = sample_batch(); // 3 rows, 2 cols

    let schema2 = Arc::new(Schema::new(vec![
        Field::new("patient_id", DataType::Int32, false),
        Field::new("ndc", DataType::Utf8, false),
    ]));
    let batch2 = RecordBatch::try_new(
        schema2,
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["12345-6789", "98765-4321"])),
        ],
    )
    .unwrap();

    let directory = sample_directory();
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    CohortWriter::write(path, &directory, &[("Claims-v1", &batch1), ("Rx-v1", &batch2)]).unwrap();

    let result = inspect(path).unwrap();

    assert_eq!(result.file_metadata.num_datasets, 2);
    assert_eq!(result.datasets[0].label, "Claims-v1");
    assert_eq!(result.datasets[0].num_rows, 3);
    assert_eq!(result.datasets[0].num_columns, 2);
    assert_eq!(result.datasets[1].label, "Rx-v1");
    assert_eq!(result.datasets[1].num_rows, 2);
    assert_eq!(result.datasets[1].num_columns, 2);
}

// ── Merge tests ───────────────────────────────────────────────

#[test]
fn merge_two_files_same_dataset() {
    let batch1 = sample_batch(); // 3 rows
    let batch2 = sample_batch(); // 3 rows
    let directory = sample_directory();

    let tmp1 = NamedTempFile::new().unwrap();
    let tmp2 = NamedTempFile::new().unwrap();
    let out = NamedTempFile::new().unwrap();
    let p1 = tmp1.path().to_str().unwrap();
    let p2 = tmp2.path().to_str().unwrap();
    let po = out.path().to_str().unwrap();

    CohortWriter::write(p1, &directory, &[("Claims-v1", &batch1)]).unwrap();
    CohortWriter::write(p2, &directory, &[("Claims-v1", &batch2)]).unwrap();

    merge(&[p1, p2], po, None).unwrap();

    let reader = CohortReader::open(po).unwrap();
    assert_eq!(reader.directory.tags.len(), 1);
    assert_eq!(reader.directory.tags[0].label, "Claims-v1");

    let batches: Vec<_> = reader
        .get_dataset_data("Claims-v1")
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 6);

    // Verify merged definition
    let def = reader.cohort_definition();
    assert!(def.name.starts_with("Union of"));
    assert_eq!(def.lineage.source_system, vec!["Test Warehouse"]);
    assert!(!def.revision_history.is_empty());
    let last = def.revision_history.last().unwrap();
    assert!(last.message.starts_with("Union of"));
    assert_eq!(last.author, "cohort-sdk");
    assert_eq!(last.size, Some(6));
}

#[test]
fn merge_disjoint_datasets() {
    let claims_batch = sample_batch(); // Claims-v1: 3 rows

    let rx_schema = Arc::new(Schema::new(vec![
        Field::new("patient_id", DataType::Int32, false),
        Field::new("ndc", DataType::Utf8, false),
    ]));
    let rx_batch = RecordBatch::try_new(
        rx_schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["12345-6789", "98765-4321"])),
        ],
    )
    .unwrap();

    let directory = sample_directory();
    let tmp1 = NamedTempFile::new().unwrap();
    let tmp2 = NamedTempFile::new().unwrap();
    let out = NamedTempFile::new().unwrap();
    let p1 = tmp1.path().to_str().unwrap();
    let p2 = tmp2.path().to_str().unwrap();
    let po = out.path().to_str().unwrap();

    CohortWriter::write(p1, &directory, &[("Claims-v1", &claims_batch)]).unwrap();
    CohortWriter::write(p2, &directory, &[("Rx-v1", &rx_batch)]).unwrap();

    merge(&[p1, p2], po, None).unwrap();

    let reader = CohortReader::open(po).unwrap();
    assert_eq!(reader.directory.tags.len(), 2);

    let labels: Vec<&str> = reader.directory.tags.iter().map(|t| t.label.as_str()).collect();
    assert!(labels.contains(&"Claims-v1"));
    assert!(labels.contains(&"Rx-v1"));

    let claims: Vec<_> = reader
        .get_dataset_data("Claims-v1")
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(claims[0].num_rows(), 3);

    let rx: Vec<_> = reader
        .get_dataset_data("Rx-v1")
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(rx[0].num_rows(), 2);
}

#[test]
fn merge_schema_mismatch_rejected() {
    let batch1 = sample_batch(); // (patient_id: Int32, diagnosis: Utf8)

    let schema2 = Arc::new(Schema::new(vec![
        Field::new("patient_id", DataType::Utf8, false), // different type!
        Field::new("diagnosis", DataType::Utf8, false),
    ]));
    let batch2 = RecordBatch::try_new(
        schema2,
        vec![
            Arc::new(StringArray::from(vec!["A", "B"])),
            Arc::new(StringArray::from(vec!["E11.0", "E11.1"])),
        ],
    )
    .unwrap();

    let directory = sample_directory();
    let tmp1 = NamedTempFile::new().unwrap();
    let tmp2 = NamedTempFile::new().unwrap();
    let out = NamedTempFile::new().unwrap();
    let p1 = tmp1.path().to_str().unwrap();
    let p2 = tmp2.path().to_str().unwrap();
    let po = out.path().to_str().unwrap();

    CohortWriter::write(p1, &directory, &[("Claims-v1", &batch1)]).unwrap();
    CohortWriter::write(p2, &directory, &[("Claims-v1", &batch2)]).unwrap();

    let err = merge(&[p1, p2], po, None).unwrap_err();
    assert!(err.to_string().contains("Schema mismatch"));
}

#[test]
fn merge_empty_input_rejected() {
    let out = NamedTempFile::new().unwrap();
    let po = out.path().to_str().unwrap();

    let err = merge(&[], po, None).unwrap_err();
    assert!(err.to_string().contains("No input files"));
}

// ── Intersect tests ───────────────────────────────────────────

#[test]
fn intersect_shared_keys_only() {
    // File 1: patients 1, 2, 3
    let batch1 = sample_batch();
    // File 2: patients 2, 3, 4
    let batch2 = RecordBatch::try_new(
        batch1.schema(),
        vec![
            Arc::new(Int32Array::from(vec![2, 3, 4])),
            Arc::new(StringArray::from(vec!["E11.1", "E11.9", "E11.0"])),
        ],
    )
    .unwrap();

    let directory = sample_directory();
    let tmp1 = NamedTempFile::new().unwrap();
    let tmp2 = NamedTempFile::new().unwrap();
    let out = NamedTempFile::new().unwrap();
    let p1 = tmp1.path().to_str().unwrap();
    let p2 = tmp2.path().to_str().unwrap();
    let po = out.path().to_str().unwrap();

    CohortWriter::write(p1, &directory, &[("Claims-v1", &batch1)]).unwrap();
    CohortWriter::write(p2, &directory, &[("Claims-v1", &batch2)]).unwrap();

    intersect(&[p1, p2], po, "patient_id", None).unwrap();

    let reader = CohortReader::open(po).unwrap();
    let batches: Vec<_> = reader
        .get_dataset_data("Claims-v1")
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    // Shared keys are 2, 3 — but each appears in both files, so 2+2 = 4 rows
    assert_eq!(total_rows, 4);

    // Verify merged definition
    let def = reader.cohort_definition();
    assert!(def.name.starts_with("Intersect of"));
    let last = def.revision_history.last().unwrap();
    assert!(last.message.starts_with("Intersect of"));
}

#[test]
fn intersect_no_overlap() {
    // File 1: patients 1, 2
    let schema = Arc::new(Schema::new(vec![
        Field::new("patient_id", DataType::Int32, false),
        Field::new("diagnosis", DataType::Utf8, false),
    ]));
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["E11.0", "E11.1"])),
        ],
    )
    .unwrap();
    // File 2: patients 3, 4
    let batch2 = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![3, 4])),
            Arc::new(StringArray::from(vec!["E11.9", "E11.0"])),
        ],
    )
    .unwrap();

    let directory = sample_directory();
    let tmp1 = NamedTempFile::new().unwrap();
    let tmp2 = NamedTempFile::new().unwrap();
    let out = NamedTempFile::new().unwrap();
    let p1 = tmp1.path().to_str().unwrap();
    let p2 = tmp2.path().to_str().unwrap();
    let po = out.path().to_str().unwrap();

    CohortWriter::write(p1, &directory, &[("Claims-v1", &batch1)]).unwrap();
    CohortWriter::write(p2, &directory, &[("Claims-v1", &batch2)]).unwrap();

    intersect(&[p1, p2], po, "patient_id", None).unwrap();

    let reader = CohortReader::open(po).unwrap();
    assert_eq!(reader.directory.tags.len(), 0);
}

#[test]
fn intersect_needs_two_files() {
    let directory = sample_directory();
    let tmp = NamedTempFile::new().unwrap();
    let out = NamedTempFile::new().unwrap();
    let p = tmp.path().to_str().unwrap();
    let po = out.path().to_str().unwrap();

    let batch = sample_batch();
    CohortWriter::write(p, &directory, &[("Claims-v1", &batch)]).unwrap();

    let err = intersect(&[p], po, "patient_id", None).unwrap_err();
    assert!(err.to_string().contains("at least 2"));
}

// ── Multi-codec round-trip tests ──────────────────────────────

#[test]
fn round_trip_arrow_ipc() {
    let batch = sample_batch();
    let directory = sample_directory();
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    CohortWriter::write_with_codec(path, &directory, &[("Claims-v1", &batch)], "arrow_ipc")
        .unwrap();

    let reader = CohortReader::open(path).unwrap();
    assert_eq!(reader.directory.tags.len(), 1);
    assert_eq!(reader.directory.tags[0].codec, "arrow_ipc");

    let batches: Vec<_> = reader
        .get_dataset_data("Claims-v1")
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 3);
    assert_eq!(batches[0].num_columns(), 2);

    // Verify inspect works
    let result = inspect(path).unwrap();
    assert_eq!(result.datasets[0].codec, "arrow_ipc");
    assert_eq!(result.datasets[0].num_rows, 3);
}

#[test]
fn round_trip_feather() {
    let batch = sample_batch();
    let directory = sample_directory();
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    CohortWriter::write_with_codec(path, &directory, &[("Claims-v1", &batch)], "feather")
        .unwrap();

    let reader = CohortReader::open(path).unwrap();
    assert_eq!(reader.directory.tags.len(), 1);
    assert_eq!(reader.directory.tags[0].codec, "feather");

    let batches: Vec<_> = reader
        .get_dataset_data("Claims-v1")
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 3);
    assert_eq!(batches[0].num_columns(), 2);

    // Verify inspect works
    let result = inspect(path).unwrap();
    assert_eq!(result.datasets[0].codec, "feather");
    assert_eq!(result.datasets[0].num_rows, 3);
}

#[test]
fn round_trip_mixed_codec_merge() {
    // Write one file as Parquet, one as Arrow IPC
    let batch = sample_batch();
    let directory = sample_directory();

    let tmp1 = NamedTempFile::new().unwrap();
    let tmp2 = NamedTempFile::new().unwrap();
    let out = NamedTempFile::new().unwrap();
    let p1 = tmp1.path().to_str().unwrap();
    let p2 = tmp2.path().to_str().unwrap();
    let po = out.path().to_str().unwrap();

    CohortWriter::write(p1, &directory, &[("Claims-v1", &batch)]).unwrap();
    CohortWriter::write_with_codec(p2, &directory, &[("Claims-v1", &batch)], "arrow_ipc")
        .unwrap();

    // Merge reads from both codecs and writes (default parquet)
    merge(&[p1, p2], po, None).unwrap();

    let reader = CohortReader::open(po).unwrap();
    let batches: Vec<_> = reader
        .get_dataset_data("Claims-v1")
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 6);
}

// ── Union tests (alias for merge) ────────────────────────────

#[test]
fn union_same_as_merge() {
    let batch = sample_batch();
    let directory = sample_directory();

    let tmp1 = NamedTempFile::new().unwrap();
    let tmp2 = NamedTempFile::new().unwrap();
    let out = NamedTempFile::new().unwrap();
    let p1 = tmp1.path().to_str().unwrap();
    let p2 = tmp2.path().to_str().unwrap();
    let po = out.path().to_str().unwrap();

    CohortWriter::write(p1, &directory, &[("Claims-v1", &batch)]).unwrap();
    CohortWriter::write(p2, &directory, &[("Claims-v1", &batch)]).unwrap();

    union(&[p1, p2], po, Some("All Claims")).unwrap();

    let reader = CohortReader::open(po).unwrap();
    assert_eq!(reader.cohort_definition().name, "All Claims");

    let batches: Vec<_> = reader
        .get_dataset_data("Claims-v1")
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 6);
}

// ── Vortex round-trip tests ──────────────────────────────────

#[test]
fn round_trip_vortex() {
    let batch = sample_batch();
    let directory = sample_directory();
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    CohortWriter::write_with_codec(path, &directory, &[("Claims-v1", &batch)], "vortex")
        .unwrap();

    let reader = CohortReader::open(path).unwrap();
    assert_eq!(reader.directory.tags.len(), 1);
    assert_eq!(reader.directory.tags[0].codec, "vortex");

    let batches: Vec<_> = reader
        .get_dataset_data("Claims-v1")
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 3);
    assert_eq!(batches[0].num_columns(), 2);

    // Verify inspect works
    let result = inspect(path).unwrap();
    assert_eq!(result.datasets[0].codec, "vortex");
    assert_eq!(result.datasets[0].num_rows, 3);
}

// ── Bitmap index tests ───────────────────────────────────────

#[test]
fn bitmap_round_trip_integer_keys() {
    let batch = sample_batch(); // patient_id: [1, 2, 3]
    let directory = sample_directory();
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    CohortWriter::write_with_codec_and_index(
        path,
        &directory,
        &[("Claims-v1", &batch)],
        "parquet",
        Some("patient_id"),
    )
    .unwrap();

    let reader = CohortReader::open(path).unwrap();
    assert!(reader.directory.patient_index.length > 0);

    let bitmap_bytes = reader.read_bitmap_bytes().unwrap().unwrap();
    let bitmap = deserialize_bitmap(&bitmap_bytes).unwrap();
    assert_eq!(bitmap.len(), 3);
    assert!(bitmap.contains(1));
    assert!(bitmap.contains(2));
    assert!(bitmap.contains(3));
    assert!(!bitmap.contains(4));
}

#[test]
fn bitmap_round_trip_string_keys() {
    let batch = sample_batch(); // diagnosis: ["E11.0", "E11.1", "E11.9"]
    let directory = sample_directory();
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    CohortWriter::write_with_codec_and_index(
        path,
        &directory,
        &[("Claims-v1", &batch)],
        "parquet",
        Some("diagnosis"),
    )
    .unwrap();

    let reader = CohortReader::open(path).unwrap();
    let bitmap_bytes = reader.read_bitmap_bytes().unwrap().unwrap();
    let bitmap = deserialize_bitmap(&bitmap_bytes).unwrap();
    assert_eq!(bitmap.len(), 3);
    assert!(bitmap.contains(hash_key("E11.0")));
    assert!(bitmap.contains(hash_key("E11.1")));
    assert!(bitmap.contains(hash_key("E11.9")));
}

#[test]
fn bitmap_intersect_empty_fast_path() {
    // File 1: patients [1, 2], File 2: patients [3, 4] — no overlap
    let schema = Arc::new(Schema::new(vec![
        Field::new("patient_id", DataType::Int32, false),
        Field::new("diagnosis", DataType::Utf8, false),
    ]));
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["E11.0", "E11.1"])),
        ],
    )
    .unwrap();
    let batch2 = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![3, 4])),
            Arc::new(StringArray::from(vec!["E11.9", "E11.0"])),
        ],
    )
    .unwrap();

    let directory = sample_directory();
    let tmp1 = NamedTempFile::new().unwrap();
    let tmp2 = NamedTempFile::new().unwrap();
    let out = NamedTempFile::new().unwrap();
    let p1 = tmp1.path().to_str().unwrap();
    let p2 = tmp2.path().to_str().unwrap();
    let po = out.path().to_str().unwrap();

    // Write both with bitmap index on patient_id
    CohortWriter::write_with_codec_and_index(
        p1, &directory, &[("Claims-v1", &batch1)], "parquet", Some("patient_id"),
    ).unwrap();
    CohortWriter::write_with_codec_and_index(
        p2, &directory, &[("Claims-v1", &batch2)], "parquet", Some("patient_id"),
    ).unwrap();

    intersect(&[p1, p2], po, "patient_id", None).unwrap();

    let reader = CohortReader::open(po).unwrap();
    // No overlap → empty result
    assert_eq!(reader.directory.tags.len(), 0);
}

#[test]
fn bitmap_intersect_with_overlap() {
    // File 1: patients [1, 2, 3], File 2: patients [2, 3, 4]
    let batch1 = sample_batch();
    let batch2 = RecordBatch::try_new(
        batch1.schema(),
        vec![
            Arc::new(Int32Array::from(vec![2, 3, 4])),
            Arc::new(StringArray::from(vec!["E11.1", "E11.9", "E11.0"])),
        ],
    )
    .unwrap();

    let directory = sample_directory();
    let tmp1 = NamedTempFile::new().unwrap();
    let tmp2 = NamedTempFile::new().unwrap();
    let out = NamedTempFile::new().unwrap();
    let p1 = tmp1.path().to_str().unwrap();
    let p2 = tmp2.path().to_str().unwrap();
    let po = out.path().to_str().unwrap();

    CohortWriter::write_with_codec_and_index(
        p1, &directory, &[("Claims-v1", &batch1)], "parquet", Some("patient_id"),
    ).unwrap();
    CohortWriter::write_with_codec_and_index(
        p2, &directory, &[("Claims-v1", &batch2)], "parquet", Some("patient_id"),
    ).unwrap();

    intersect(&[p1, p2], po, "patient_id", None).unwrap();

    let reader = CohortReader::open(po).unwrap();
    let batches: Vec<_> = reader
        .get_dataset_data("Claims-v1")
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    // Shared keys 2, 3 — appear in both files → 2+2 = 4 rows
    assert_eq!(total_rows, 4);
}

// ── bitmap_can_skip_file tests ────────────────────────────────

#[test]
fn bitmap_can_skip_all_missing() {
    let mut bm = roaring::RoaringTreemap::new();
    bm.insert(1);
    bm.insert(2);
    bm.insert(3);
    // Query for values not in bitmap → can skip
    assert!(bitmap_can_skip_file(&bm, &[10, 20, 30]));
}

#[test]
fn bitmap_can_skip_some_present() {
    let mut bm = roaring::RoaringTreemap::new();
    bm.insert(1);
    bm.insert(2);
    bm.insert(3);
    // Query includes value 2 which IS in bitmap → cannot skip
    assert!(!bitmap_can_skip_file(&bm, &[2, 20]));
}

#[test]
fn bitmap_can_skip_empty_query() {
    let mut bm = roaring::RoaringTreemap::new();
    bm.insert(1);
    // Empty query → all(empty) is true → can skip
    assert!(bitmap_can_skip_file(&bm, &[]));
}

// ── extract_key_filter_values tests ───────────────────────────

#[test]
fn extract_eq_integer() {
    let expr = col("patient_id").eq(lit(42i64));
    let result = extract_key_filter_values(&expr, "patient_id", false);
    assert_eq!(result, Some(vec![42]));
}

#[test]
fn extract_eq_string() {
    let expr = col("patient_id").eq(lit("ABC"));
    let result = extract_key_filter_values(&expr, "patient_id", true);
    assert_eq!(result, Some(vec![hash_key("ABC")]));
}

#[test]
fn extract_in_list() {
    let expr = Expr::InList(datafusion::logical_expr::expr::InList {
        expr: Box::new(col("patient_id")),
        list: vec![lit(1i64), lit(2i64), lit(3i64)],
        negated: false,
    });
    let result = extract_key_filter_values(&expr, "patient_id", false);
    assert_eq!(result, Some(vec![1, 2, 3]));
}

#[test]
fn extract_and_combination() {
    let expr = col("patient_id")
        .eq(lit(42i64))
        .and(col("patient_id").eq(lit(99i64)));
    let result = extract_key_filter_values(&expr, "patient_id", false);
    assert_eq!(result, Some(vec![42, 99]));
}

#[test]
fn extract_non_matching_column() {
    let expr = col("other_col").eq(lit(42i64));
    let result = extract_key_filter_values(&expr, "patient_id", false);
    assert_eq!(result, None);
}

#[test]
fn extract_negated_in_list_returns_none() {
    let expr = Expr::InList(datafusion::logical_expr::expr::InList {
        expr: Box::new(col("patient_id")),
        list: vec![lit(1i64)],
        negated: true,
    });
    let result = extract_key_filter_values(&expr, "patient_id", false);
    assert_eq!(result, None);
}

// ── writer populates primary_keys ─────────────────────────────

#[test]
fn writer_stamps_primary_keys() {
    let batch = sample_batch();
    let directory = sample_directory();
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    CohortWriter::write_with_codec_and_index(
        path, &directory, &[("Claims-v1", &batch)], "parquet", Some("patient_id"),
    ).unwrap();

    let reader = CohortReader::open(path).unwrap();
    assert_eq!(reader.cohort_definition().primary_keys, vec!["patient_id"]);
}

// ── diff patient overlap tests ────────────────────────────────

#[test]
fn diff_full_overlap() {
    let batch = sample_batch(); // patients [1, 2, 3]
    let directory = sample_directory();
    let tmp1 = NamedTempFile::new().unwrap();
    let tmp2 = NamedTempFile::new().unwrap();
    let p1 = tmp1.path().to_str().unwrap();
    let p2 = tmp2.path().to_str().unwrap();

    CohortWriter::write_with_codec_and_index(
        p1, &directory, &[("Claims-v1", &batch)], "parquet", Some("patient_id"),
    ).unwrap();
    CohortWriter::write_with_codec_and_index(
        p2, &directory, &[("Claims-v1", &batch)], "parquet", Some("patient_id"),
    ).unwrap();

    let d = diff(p1, p2).unwrap();
    let po = d.patient_overlap.expect("should have overlap stats");
    assert_eq!(po.left_patient_count, 3);
    assert_eq!(po.right_patient_count, 3);
    assert_eq!(po.shared_patient_count, 3);
    assert_eq!(po.left_only_count, 0);
    assert_eq!(po.right_only_count, 0);
    assert!((po.jaccard_similarity - 1.0).abs() < 1e-10);
}

#[test]
fn diff_partial_overlap() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("patient_id", DataType::Int32, false),
        Field::new("diagnosis", DataType::Utf8, false),
    ]));
    // [1, 2, 3]
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["A", "B", "C"])),
        ],
    ).unwrap();
    // [2, 3, 4]
    let batch2 = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![2, 3, 4])),
            Arc::new(StringArray::from(vec!["B", "C", "D"])),
        ],
    ).unwrap();

    let directory = sample_directory();
    let tmp1 = NamedTempFile::new().unwrap();
    let tmp2 = NamedTempFile::new().unwrap();
    let p1 = tmp1.path().to_str().unwrap();
    let p2 = tmp2.path().to_str().unwrap();

    CohortWriter::write_with_codec_and_index(
        p1, &directory, &[("Claims-v1", &batch1)], "parquet", Some("patient_id"),
    ).unwrap();
    CohortWriter::write_with_codec_and_index(
        p2, &directory, &[("Claims-v1", &batch2)], "parquet", Some("patient_id"),
    ).unwrap();

    let d = diff(p1, p2).unwrap();
    let po = d.patient_overlap.expect("should have overlap stats");
    assert_eq!(po.shared_patient_count, 2);
    assert_eq!(po.left_only_count, 1);
    assert_eq!(po.right_only_count, 1);
    // Jaccard: 2 / (3+3-2) = 2/4 = 0.5
    assert!((po.jaccard_similarity - 0.5).abs() < 1e-10);
}

#[test]
fn diff_no_overlap() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("patient_id", DataType::Int32, false),
        Field::new("diagnosis", DataType::Utf8, false),
    ]));
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["A", "B"])),
        ],
    ).unwrap();
    let batch2 = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![3, 4])),
            Arc::new(StringArray::from(vec!["C", "D"])),
        ],
    ).unwrap();

    let directory = sample_directory();
    let tmp1 = NamedTempFile::new().unwrap();
    let tmp2 = NamedTempFile::new().unwrap();
    let p1 = tmp1.path().to_str().unwrap();
    let p2 = tmp2.path().to_str().unwrap();

    CohortWriter::write_with_codec_and_index(
        p1, &directory, &[("Claims-v1", &batch1)], "parquet", Some("patient_id"),
    ).unwrap();
    CohortWriter::write_with_codec_and_index(
        p2, &directory, &[("Claims-v1", &batch2)], "parquet", Some("patient_id"),
    ).unwrap();

    let d = diff(p1, p2).unwrap();
    let po = d.patient_overlap.expect("should have overlap stats");
    assert_eq!(po.shared_patient_count, 0);
    assert!((po.jaccard_similarity - 0.0).abs() < 1e-10);
}

#[test]
fn diff_no_bitmaps_returns_none() {
    let batch = sample_batch();
    let directory = sample_directory();
    let tmp1 = NamedTempFile::new().unwrap();
    let tmp2 = NamedTempFile::new().unwrap();
    let p1 = tmp1.path().to_str().unwrap();
    let p2 = tmp2.path().to_str().unwrap();

    // Write WITHOUT bitmap index
    CohortWriter::write(p1, &directory, &[("Claims-v1", &batch)]).unwrap();
    CohortWriter::write(p2, &directory, &[("Claims-v1", &batch)]).unwrap();

    let d = diff(p1, p2).unwrap();
    assert!(d.patient_overlap.is_none());
}

#[test]
fn diff_format_includes_overlap() {
    let batch = sample_batch();
    let directory = sample_directory();
    let tmp1 = NamedTempFile::new().unwrap();
    let tmp2 = NamedTempFile::new().unwrap();
    let p1 = tmp1.path().to_str().unwrap();
    let p2 = tmp2.path().to_str().unwrap();

    CohortWriter::write_with_codec_and_index(
        p1, &directory, &[("Claims-v1", &batch)], "parquet", Some("patient_id"),
    ).unwrap();
    CohortWriter::write_with_codec_and_index(
        p2, &directory, &[("Claims-v1", &batch)], "parquet", Some("patient_id"),
    ).unwrap();

    let d = diff(p1, p2).unwrap();
    let output = format_diff(&d);
    assert!(output.contains("Patient Overlap"));
    assert!(output.contains("Jaccard"));
    assert!(output.contains("Shared"));
}

// ── Row group statistics pruning tests ────────────────────────

use crate::table_provider::{can_skip_row_group};

#[test]
fn rg_pruning_eq_outside_range() {
    // Write file with patient_id [100, 200, 300] — row group stats: min=100, max=300
    let schema = Arc::new(Schema::new(vec![
        Field::new("patient_id", DataType::Int32, false),
        Field::new("diagnosis", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![100, 200, 300])),
            Arc::new(StringArray::from(vec!["A", "B", "C"])),
        ],
    ).unwrap();

    let directory = sample_directory();
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();
    CohortWriter::write(path, &directory, &[("Claims-v1", &batch)]).unwrap();

    // Get Parquet metadata
    let reader = CohortReader::open(path).unwrap();
    let tag = reader.directory.tags.iter().find(|t| t.label == "Claims-v1").unwrap();
    let offset_reader = crate::offset_reader::OffsetReader::new(path, tag.offset, tag.length);
    let meta = parquet::arrow::arrow_reader::ArrowReaderMetadata::load(
        &offset_reader, Default::default()
    ).unwrap();
    let rg_meta = meta.metadata().row_group(0);

    // patient_id = 50 → outside [100, 300] → should skip
    let expr_outside = col("patient_id").eq(lit(50i32));
    assert!(can_skip_row_group(&expr_outside, rg_meta));

    // patient_id = 400 → outside [100, 300] → should skip
    let expr_above = col("patient_id").eq(lit(400i32));
    assert!(can_skip_row_group(&expr_above, rg_meta));

    // patient_id = 200 → inside [100, 300] → should NOT skip
    let expr_inside = col("patient_id").eq(lit(200i32));
    assert!(!can_skip_row_group(&expr_inside, rg_meta));
}

#[test]
fn rg_pruning_range_comparisons() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("age", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![20, 30, 40]))],
    ).unwrap();

    let directory = sample_directory();
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();
    CohortWriter::write(path, &directory, &[("data", &batch)]).unwrap();

    let reader = CohortReader::open(path).unwrap();
    let tag = reader.directory.tags.iter().find(|t| t.label == "data").unwrap();
    let offset_reader = crate::offset_reader::OffsetReader::new(path, tag.offset, tag.length);
    let meta = parquet::arrow::arrow_reader::ArrowReaderMetadata::load(
        &offset_reader, Default::default()
    ).unwrap();
    let rg_meta = meta.metadata().row_group(0);

    // age > 50: max=40, so max <= 50 → can skip
    assert!(can_skip_row_group(&col("age").gt(lit(50i32)), rg_meta));
    // age > 30: max=40 > 30 → cannot skip
    assert!(!can_skip_row_group(&col("age").gt(lit(30i32)), rg_meta));
    // age < 10: min=20, so min >= 10 → can skip
    assert!(can_skip_row_group(&col("age").lt(lit(10i32)), rg_meta));
    // age < 25: min=20 < 25 → cannot skip
    assert!(!can_skip_row_group(&col("age").lt(lit(25i32)), rg_meta));
}

#[test]
fn rg_pruning_string_column() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("code", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(StringArray::from(vec!["B00", "D50", "F10"]))],
    ).unwrap();

    let directory = sample_directory();
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();
    CohortWriter::write(path, &directory, &[("data", &batch)]).unwrap();

    let reader = CohortReader::open(path).unwrap();
    let tag = reader.directory.tags.iter().find(|t| t.label == "data").unwrap();
    let offset_reader = crate::offset_reader::OffsetReader::new(path, tag.offset, tag.length);
    let meta = parquet::arrow::arrow_reader::ArrowReaderMetadata::load(
        &offset_reader, Default::default()
    ).unwrap();
    let rg_meta = meta.metadata().row_group(0);

    // code = "A00" → before min "B00" → skip
    assert!(can_skip_row_group(&col("code").eq(lit("A00")), rg_meta));
    // code = "Z99" → after max "F10" → skip
    assert!(can_skip_row_group(&col("code").eq(lit("Z99")), rg_meta));
    // code = "D50" → inside range → don't skip
    assert!(!can_skip_row_group(&col("code").eq(lit("D50")), rg_meta));
}

#[test]
fn rg_pruning_in_list() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("patient_id", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![100, 200, 300]))],
    ).unwrap();

    let directory = sample_directory();
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();
    CohortWriter::write(path, &directory, &[("data", &batch)]).unwrap();

    let reader = CohortReader::open(path).unwrap();
    let tag = reader.directory.tags.iter().find(|t| t.label == "data").unwrap();
    let offset_reader = crate::offset_reader::OffsetReader::new(path, tag.offset, tag.length);
    let meta = parquet::arrow::arrow_reader::ArrowReaderMetadata::load(
        &offset_reader, Default::default()
    ).unwrap();
    let rg_meta = meta.metadata().row_group(0);

    // IN (1, 2, 3) → all outside [100, 300] → skip
    let expr_outside = Expr::InList(datafusion::logical_expr::expr::InList {
        expr: Box::new(col("patient_id")),
        list: vec![lit(1i32), lit(2i32), lit(3i32)],
        negated: false,
    });
    assert!(can_skip_row_group(&expr_outside, rg_meta));

    // IN (1, 200, 999) → 200 inside [100, 300] → don't skip
    let expr_partial = Expr::InList(datafusion::logical_expr::expr::InList {
        expr: Box::new(col("patient_id")),
        list: vec![lit(1i32), lit(200i32), lit(999i32)],
        negated: false,
    });
    assert!(!can_skip_row_group(&expr_partial, rg_meta));
}

#[test]
fn rg_pruning_and_or_logic() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int32, false),
        Field::new("y", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![10, 20, 30])),
            Arc::new(Int32Array::from(vec![100, 200, 300])),
        ],
    ).unwrap();

    let directory = sample_directory();
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();
    CohortWriter::write(path, &directory, &[("data", &batch)]).unwrap();

    let reader = CohortReader::open(path).unwrap();
    let tag = reader.directory.tags.iter().find(|t| t.label == "data").unwrap();
    let offset_reader = crate::offset_reader::OffsetReader::new(path, tag.offset, tag.length);
    let meta = parquet::arrow::arrow_reader::ArrowReaderMetadata::load(
        &offset_reader, Default::default()
    ).unwrap();
    let rg_meta = meta.metadata().row_group(0);

    // AND: x = 5 (impossible) AND y = 200 (possible) → can skip (one side impossible)
    let and_expr = col("x").eq(lit(5i32)).and(col("y").eq(lit(200i32)));
    assert!(can_skip_row_group(&and_expr, rg_meta));

    // OR: x = 5 (impossible) OR y = 200 (possible) → cannot skip (one side possible)
    let or_expr = col("x").eq(lit(5i32)).or(col("y").eq(lit(200i32)));
    assert!(!can_skip_row_group(&or_expr, rg_meta));

    // OR: x = 5 (impossible) OR y = 999 (impossible) → can skip (both impossible)
    let or_both = col("x").eq(lit(5i32)).or(col("y").eq(lit(999i32)));
    assert!(can_skip_row_group(&or_both, rg_meta));
}

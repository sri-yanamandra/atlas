use std::collections::HashSet;

use arrow::array::{Array, StringArray};
use arrow::compute::{cast, concat_batches, filter_record_batch};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use chrono::Utc;

use crate::models::*;
use crate::reader::CohortReader;
use crate::writer::CohortWriter;

// ── Shared helpers ────────────────────────────────────────────

/// Read all datasets from a single .cohort file into memory.
pub fn read_all_datasets(
    reader: &CohortReader,
) -> Result<Vec<(String, RecordBatch)>, Box<dyn std::error::Error>> {
    let mut result = Vec::new();
    for tag in &reader.directory.tags {
        let batches: Vec<RecordBatch> = reader
            .get_dataset_data(&tag.label)?
            .collect::<Result<Vec<_>, _>>()?;
        if !batches.is_empty() {
            let schema = batches[0].schema();
            let merged = concat_batches(&schema, &batches)?;
            result.push((tag.label.clone(), merged));
        }
    }
    Ok(result)
}

/// Extract the values of a key column as a set of strings.
pub fn extract_key_set(
    batches: &[RecordBatch],
    key_col: &str,
) -> Result<HashSet<String>, Box<dyn std::error::Error>> {
    let mut keys = HashSet::new();
    for batch in batches {
        let col_idx = batch.schema().index_of(key_col)?;
        let array = batch.column(col_idx);
        let string_array = cast(array, &DataType::Utf8)?;
        let string_array = string_array
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| format!("Cannot cast column '{}' to string", key_col))?;
        for i in 0..string_array.len() {
            if !string_array.is_null(i) {
                keys.insert(string_array.value(i).to_string());
            }
        }
    }
    Ok(keys)
}

/// Filter a batch to only rows where the key column value is in `keep`.
pub fn filter_batch_by_keys(
    batch: &RecordBatch,
    key_col: &str,
    keep: &HashSet<String>,
) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let col_idx = batch.schema().index_of(key_col)?;
    let array = batch.column(col_idx);
    let string_array = cast(array, &DataType::Utf8)?;
    let string_array = string_array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| format!("Cannot cast column '{}' to string", key_col))?;

    let mask: Vec<bool> = (0..string_array.len())
        .map(|i| {
            if string_array.is_null(i) {
                false
            } else {
                keep.contains(string_array.value(i))
            }
        })
        .collect();

    let boolean_array = arrow::array::BooleanArray::from(mask);
    Ok(filter_record_batch(batch, &boolean_array)?)
}

/// Filter a batch to only rows where the key column value is NOT in `exclude`.
pub fn filter_batch_excluding_keys(
    batch: &RecordBatch,
    key_col: &str,
    exclude: &HashSet<String>,
) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let col_idx = batch.schema().index_of(key_col)?;
    let array = batch.column(col_idx);
    let string_array = cast(array, &DataType::Utf8)?;
    let string_array = string_array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| format!("Cannot cast column '{}' to string", key_col))?;

    let mask: Vec<bool> = (0..string_array.len())
        .map(|i| {
            if string_array.is_null(i) {
                true // keep nulls
            } else {
                !exclude.contains(string_array.value(i))
            }
        })
        .collect();

    let boolean_array = arrow::array::BooleanArray::from(mask);
    Ok(filter_record_batch(batch, &boolean_array)?)
}

/// Validate schemas match and concatenate batches.
pub fn validate_and_concat(
    label: &str,
    batches: &[RecordBatch],
) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let schema = batches[0].schema();
    for (i, batch) in batches.iter().enumerate().skip(1) {
        if batch.schema() != schema {
            return Err(format!(
                "Schema mismatch for dataset '{}': batch 0 has schema {:?}, batch {} has schema {:?}",
                label, schema, i, batch.schema()
            )
            .into());
        }
    }
    Ok(concat_batches(&schema, batches)?)
}

/// Write datasets to a .cohort file with the given directory.
pub fn write_datasets(
    output_path: &str,
    directory: &TagDirectory,
    datasets: &[(String, RecordBatch)],
) -> Result<(), Box<dyn std::error::Error>> {
    let datasets_ref: Vec<(&str, &RecordBatch)> = datasets
        .iter()
        .map(|(label, batch)| (label.as_str(), batch))
        .collect();
    CohortWriter::write(output_path, directory, &datasets_ref)?;
    Ok(())
}

/// Build a new revision entry and bump the definition.
fn bump_revision(
    def: &CohortDefinition,
    message: &str,
    author: &str,
    deltas: Vec<Delta>,
    total_rows: u64,
) -> (CohortDefinition, RevisionEntry) {
    let new_rev = def.revision + 1;
    let entry = RevisionEntry {
        revision: new_rev,
        timestamp: Utc::now(),
        author: author.to_string(),
        message: message.to_string(),
        deltas,
        size: Some(total_rows),
    };
    let mut new_def = def.clone();
    // apply_revision handles selectors + history
    new_def.apply_revision(entry.clone()).unwrap();
    (new_def, entry)
}

/// Resolve output path: use explicit output or default to in-place.
fn resolve_output<'a>(path: &'a str, output: Option<&'a str>) -> &'a str {
    output.unwrap_or(path)
}

// ── Append ────────────────────────────────────────────────────

/// Resolve the target dataset label for an append.
/// If `dataset_label` is provided, use it directly.
/// Otherwise, auto-detect by matching the new data's schema against existing datasets.
fn resolve_dataset_label(
    datasets: &[(String, RecordBatch)],
    dataset_label: Option<&str>,
    new_data: &RecordBatch,
) -> Result<String, Box<dyn std::error::Error>> {
    if let Some(label) = dataset_label {
        return Ok(label.to_string());
    }

    let matches: Vec<&str> = datasets
        .iter()
        .filter(|(_, batch)| batch.schema() == new_data.schema())
        .map(|(label, _)| label.as_str())
        .collect();

    match matches.len() {
        0 => Err("No existing dataset matches the schema of the incoming data. Specify --dataset to create a new one.".into()),
        1 => Ok(matches[0].to_string()),
        _ => Err(format!(
            "Ambiguous: {} datasets match the incoming schema: {}. Specify --dataset to disambiguate.",
            matches.len(),
            matches.join(", ")
        ).into()),
    }
}

/// Append rows to an existing dataset (or add a new dataset) in a .cohort file.
/// If `dataset_label` is `None`, auto-detects the target by schema matching.
pub fn append(
    path: &str,
    output: Option<&str>,
    dataset_label: Option<&str>,
    new_data: &RecordBatch,
    message: &str,
    author: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = CohortReader::open(path)?;
    let mut datasets = read_all_datasets(&reader)?;

    let label = resolve_dataset_label(&datasets, dataset_label, new_data)?;

    // Find existing dataset or add new
    if let Some(pos) = datasets.iter().position(|(l, _)| l == &label) {
        let existing = &datasets[pos].1;
        if existing.schema() != new_data.schema() {
            return Err(format!(
                "Schema mismatch for dataset '{}': existing has {:?}, new data has {:?}",
                label,
                existing.schema(),
                new_data.schema()
            )
            .into());
        }
        let merged = concat_batches(&existing.schema(), &[existing.clone(), new_data.clone()])?;
        datasets[pos] = (label.clone(), merged);
    } else {
        datasets.push((label.clone(), new_data.clone()));
    }

    let total_rows: u64 = datasets.iter().map(|(_, b)| b.num_rows() as u64).sum();
    let (new_def, _) = bump_revision(
        reader.cohort_definition(),
        message,
        author,
        vec![],
        total_rows,
    );

    let directory = TagDirectory {
        tags: vec![],
        patient_index: reader.directory.patient_index.clone(),
        cohort_definition: new_def,
        backend: reader.directory.backend.clone(),
    };

    write_datasets(resolve_output(path, output), &directory, &datasets)
}

// ── Delete rows ───────────────────────────────────────────────

/// Delete rows by key values across all datasets.
pub fn delete_rows(
    path: &str,
    output: Option<&str>,
    key_column: &str,
    values: &[String],
    message: &str,
    author: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = CohortReader::open(path)?;
    let datasets = read_all_datasets(&reader)?;

    let exclude: HashSet<String> = values.iter().cloned().collect();
    let mut result = Vec::new();
    for (label, batch) in &datasets {
        let filtered = filter_batch_excluding_keys(batch, key_column, &exclude)?;
        if filtered.num_rows() > 0 {
            result.push((label.clone(), filtered));
        }
    }

    let total_rows: u64 = result.iter().map(|(_, b)| b.num_rows() as u64).sum();
    let (new_def, _) = bump_revision(
        reader.cohort_definition(),
        message,
        author,
        vec![],
        total_rows,
    );

    let directory = TagDirectory {
        tags: vec![],
        patient_index: reader.directory.patient_index.clone(),
        cohort_definition: new_def,
        backend: reader.directory.backend.clone(),
    };

    write_datasets(resolve_output(path, output), &directory, &result)
}

// ── Delete dataset ────────────────────────────────────────────

/// Delete an entire dataset from a .cohort file.
pub fn delete_dataset(
    path: &str,
    output: Option<&str>,
    label: &str,
    message: &str,
    author: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = CohortReader::open(path)?;
    let datasets = read_all_datasets(&reader)?;

    if !datasets.iter().any(|(l, _)| l == label) {
        return Err(format!("Dataset '{}' not found", label).into());
    }

    let result: Vec<(String, RecordBatch)> = datasets
        .into_iter()
        .filter(|(l, _)| l != label)
        .collect();

    let total_rows: u64 = result.iter().map(|(_, b)| b.num_rows() as u64).sum();
    let (new_def, _) = bump_revision(
        reader.cohort_definition(),
        message,
        author,
        vec![],
        total_rows,
    );

    let directory = TagDirectory {
        tags: vec![],
        patient_index: reader.directory.patient_index.clone(),
        cohort_definition: new_def,
        backend: reader.directory.backend.clone(),
    };

    write_datasets(resolve_output(path, output), &directory, &result)
}

// ── Delete selectors ──────────────────────────────────────────

/// Delete selectors by ID from a .cohort file.
pub fn delete_selectors(
    path: &str,
    output: Option<&str>,
    selector_ids: &[String],
    message: &str,
    author: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = CohortReader::open(path)?;
    let datasets = read_all_datasets(&reader)?;

    let deltas: Vec<Delta> = selector_ids
        .iter()
        .map(|id| Delta {
            action: DeltaAction::Remove,
            selector_id: id.clone(),
            field: None,
            before: None,
            after: None,
            selector: None,
        })
        .collect();

    let total_rows: u64 = datasets.iter().map(|(_, b)| b.num_rows() as u64).sum();
    let (new_def, _) = bump_revision(
        reader.cohort_definition(),
        message,
        author,
        deltas,
        total_rows,
    );

    let directory = TagDirectory {
        tags: vec![],
        patient_index: reader.directory.patient_index.clone(),
        cohort_definition: new_def,
        backend: reader.directory.backend.clone(),
    };

    write_datasets(resolve_output(path, output), &directory, &datasets)
}

// ── Upsert ────────────────────────────────────────────────────

/// Update-or-insert rows by key column in a dataset.
/// Apply a sequence of incremental (delete-keys, add-rows) steps to a base batch.
/// Returns a Vec of batches to avoid offset overflow on large string columns.
pub fn increment(
    base: RecordBatch,
    key_column: &str,
    increments: &[(HashSet<String>, Option<RecordBatch>)],
) -> Result<Vec<RecordBatch>, Box<dyn std::error::Error>> {
    let mut batches = vec![base];
    for (delete_keys, adds) in increments {
        if !delete_keys.is_empty() {
            batches = batches
                .into_iter()
                .map(|b| filter_batch_excluding_keys(&b, key_column, delete_keys))
                .collect::<Result<Vec<_>, _>>()?;
            batches.retain(|b| b.num_rows() > 0);
        }
        if let Some(adds) = adds {
            batches.push(adds.clone());
        }
    }
    Ok(batches)
}

pub fn upsert(
    path: &str,
    output: Option<&str>,
    dataset_label: &str,
    key_column: &str,
    new_data: &RecordBatch,
    message: &str,
    author: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = CohortReader::open(path)?;
    let mut datasets = read_all_datasets(&reader)?;

    // Extract keys from new data
    let new_keys = extract_key_set(&[new_data.clone()], key_column)?;

    if let Some(pos) = datasets.iter().position(|(l, _)| l == dataset_label) {
        let existing = &datasets[pos].1;
        if existing.schema() != new_data.schema() {
            return Err(format!(
                "Schema mismatch for dataset '{}': existing has {:?}, new data has {:?}",
                dataset_label,
                existing.schema(),
                new_data.schema()
            )
            .into());
        }
        // Remove rows with matching keys, then concat with new data
        let filtered = filter_batch_excluding_keys(existing, key_column, &new_keys)?;
        let merged = if filtered.num_rows() > 0 {
            concat_batches(&existing.schema(), &[filtered, new_data.clone()])?
        } else {
            new_data.clone()
        };
        datasets[pos] = (dataset_label.to_string(), merged);
    } else {
        // New dataset — just add it
        datasets.push((dataset_label.to_string(), new_data.clone()));
    }

    let total_rows: u64 = datasets.iter().map(|(_, b)| b.num_rows() as u64).sum();
    let (new_def, _) = bump_revision(
        reader.cohort_definition(),
        message,
        author,
        vec![],
        total_rows,
    );

    let directory = TagDirectory {
        tags: vec![],
        patient_index: reader.directory.patient_index.clone(),
        cohort_definition: new_def,
        backend: reader.directory.backend.clone(),
    };

    write_datasets(resolve_output(path, output), &directory, &datasets)
}

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use arrow::compute::concat_batches;
use arrow::record_batch::RecordBatch;
use chrono::Utc;

use crate::bitmap_index::{deserialize_bitmap, intersect_bitmaps};
use crate::models::*;
use crate::mutate::{extract_key_set, filter_batch_by_keys, write_datasets};
use crate::reader::CohortReader;

/// Read all datasets from all input files, grouped by label.
/// Returns a Vec (one per file) of (TagDirectory, BTreeMap<label, Vec<RecordBatch>>).
fn read_all_files(
    input_paths: &[&str],
) -> Result<Vec<(TagDirectory, BTreeMap<String, Vec<RecordBatch>>)>, Box<dyn std::error::Error>> {
    let mut all_files = Vec::with_capacity(input_paths.len());
    for path in input_paths {
        let reader = CohortReader::open(path)?;
        let mut datasets: BTreeMap<String, Vec<RecordBatch>> = BTreeMap::new();
        for tag in &reader.directory.tags {
            let batches: Vec<RecordBatch> = reader
                .get_dataset_data(&tag.label)?
                .collect::<Result<Vec<_>, _>>()?;
            datasets.entry(tag.label.clone()).or_default().extend(batches);
        }
        all_files.push((reader.directory.clone(), datasets));
    }
    Ok(all_files)
}

/// Build a merged CohortDefinition from multiple input directories.
fn build_merged_definition(
    directories: &[&TagDirectory],
    input_paths: &[&str],
    operation_label: &str,
    name: Option<&str>,
) -> TagDirectory {
    let defs: Vec<&CohortDefinition> = directories
        .iter()
        .map(|d| &d.cohort_definition)
        .collect();

    // Name: use override, or generate from file basenames
    let file_names: Vec<String> = input_paths
        .iter()
        .map(|p| {
            Path::new(p)
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_else(|| p.to_string())
        })
        .collect();

    let merged_name = match name {
        Some(n) => n.to_string(),
        None => {
            let joiner = if operation_label == "Intersect" { " ∩ " } else { ", " };
            format!("{} of {}", operation_label, file_names.join(joiner))
        }
    };

    // Selectors: dedup-merge by selector_id (first occurrence wins)
    let mut seen_ids = HashSet::new();
    let mut selectors = Vec::new();
    for def in &defs {
        for sel in &def.selectors {
            if seen_ids.insert(sel.selector_id.clone()) {
                selectors.push(sel.clone());
            }
        }
    }

    // Lineage: union source_systems, union source_tables, latest extraction_date
    let mut source_systems = Vec::new();
    let mut seen_systems = HashSet::new();
    let mut source_tables = Vec::new();
    let mut seen_tables = HashSet::new();
    let mut latest_date = String::new();
    for def in &defs {
        for sys in &def.lineage.source_system {
            if seen_systems.insert(sys.clone()) {
                source_systems.push(sys.clone());
            }
        }
        for tbl in &def.lineage.source_tables {
            if seen_tables.insert(tbl.clone()) {
                source_tables.push(tbl.clone());
            }
        }
        if def.lineage.extraction_date > latest_date {
            latest_date = def.lineage.extraction_date.clone();
        }
    }

    // Revision: max of all revisions + 1
    let max_rev = defs.iter().map(|d| d.revision).max().unwrap_or(0);
    let new_rev = max_rev + 1;

    // Revision history: concatenate all, dedup by revision number, then append new entry
    let mut seen_revs = HashSet::new();
    let mut history = Vec::new();
    for def in &defs {
        for entry in &def.revision_history {
            if seen_revs.insert(entry.revision) {
                history.push(entry.clone());
            }
        }
    }
    history.sort_by_key(|e| e.revision);

    let message = format!("{} of {}", operation_label, file_names.join(", "));
    history.push(RevisionEntry {
        revision: new_rev,
        timestamp: Utc::now(),
        author: "cohort-sdk".to_string(),
        message,
        deltas: vec![],
        size: None, // stamped after row count is known
    });

    let merged_def = CohortDefinition {
        revision: new_rev,
        name: merged_name,
        description: None,
        created_at: Utc::now(),
        created_by: "cohort-sdk".to_string(),
        logic_operator: "AND".to_string(),
        selectors,
        lineage: Lineage {
            source_system: source_systems,
            source_tables,
            query_hash: None,
            extraction_date: latest_date,
            generation_job_id: None,
        },
        revision_history: history,
        primary_keys: vec![],
    };

    TagDirectory {
        tags: vec![],
        patient_index: ByteRange {
            offset: 0,
            length: 0,
        },
        cohort_definition: merged_def,
        backend: None,
    }
}

/// Validate schemas and concat a list of batches.
fn validate_and_concat(
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

// ── Public API ────────────────────────────────────────────────

/// Union: concatenate all rows from all input files per dataset.
pub fn union(
    input_paths: &[&str],
    output_path: &str,
    name: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    if input_paths.is_empty() {
        return Err("No input files provided".into());
    }

    let all_files = read_all_files(input_paths)?;

    // Build merged definition from all input directories
    let directories: Vec<&TagDirectory> = all_files.iter().map(|(d, _)| d).collect();
    let directory = build_merged_definition(&directories, input_paths, "Union", name);

    // Flatten into label → all batches
    let mut dataset_batches: BTreeMap<String, Vec<RecordBatch>> = BTreeMap::new();
    for (_dir, file) in &all_files {
        for (label, batches) in file {
            dataset_batches.entry(label.clone()).or_default().extend(batches.clone());
        }
    }

    let mut merged: Vec<(String, RecordBatch)> = Vec::new();
    for (label, batches) in &dataset_batches {
        if batches.is_empty() {
            continue;
        }
        merged.push((label.clone(), validate_and_concat(label, batches)?));
    }

    // Stamp total row count on the last history entry
    let total_rows: u64 = merged.iter().map(|(_, b)| b.num_rows() as u64).sum();
    let mut directory = directory;
    if let Some(last) = directory.cohort_definition.revision_history.last_mut() {
        last.size = Some(total_rows);
    }

    write_datasets(output_path, &directory, &merged)
}

/// Intersect: keep only rows where `on` key appears in ALL input files.
pub fn intersect(
    input_paths: &[&str],
    output_path: &str,
    on: &str,
    name: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    if input_paths.len() < 2 {
        return Err("Intersect requires at least 2 input files".into());
    }

    // Bitmap fast-path: if all files have a bitmap patient index,
    // compute the intersection of bitmaps. If the result is empty,
    // we can skip loading any data blocks entirely.
    let readers: Vec<CohortReader> = input_paths
        .iter()
        .map(|p| CohortReader::open(p))
        .collect::<Result<_, _>>()?;
    let all_have_bitmap = readers
        .iter()
        .all(|r| r.directory.patient_index.length > 0);
    if all_have_bitmap {
        let mut bitmaps = Vec::with_capacity(readers.len());
        for reader in &readers {
            if let Some(bytes) = reader.read_bitmap_bytes()? {
                bitmaps.push(deserialize_bitmap(&bytes)?);
            }
        }
        let shared = intersect_bitmaps(&bitmaps);
        if shared.is_empty() {
            // No overlap — write an empty cohort and return early
            let directories: Vec<&TagDirectory> =
                readers.iter().map(|r| &r.directory).collect();
            let mut directory =
                build_merged_definition(&directories, input_paths, "Intersect", name);
            if let Some(last) = directory.cohort_definition.revision_history.last_mut() {
                last.size = Some(0);
            }
            return write_datasets(output_path, &directory, &[]);
        }
    }
    drop(readers);

    let all_files = read_all_files(input_paths)?;

    // Build merged definition from all input directories
    let directories: Vec<&TagDirectory> = all_files.iter().map(|(d, _)| d).collect();
    let directory = build_merged_definition(&directories, input_paths, "Intersect", name);

    // Collect all dataset labels that exist in EVERY file
    let common_labels: Vec<String> = {
        let mut label_sets: Vec<HashSet<String>> = all_files
            .iter()
            .map(|(_, f)| f.keys().cloned().collect())
            .collect();
        let first = label_sets.remove(0);
        label_sets
            .into_iter()
            .fold(first, |acc, s| acc.intersection(&s).cloned().collect())
            .into_iter()
            .collect()
    };

    let mut result: Vec<(String, RecordBatch)> = Vec::new();

    for label in &common_labels {
        // Get key sets per file for this dataset
        let per_file_batches: Vec<&Vec<RecordBatch>> = all_files
            .iter()
            .filter_map(|(_, f)| f.get(label))
            .collect();

        let mut key_sets: Vec<HashSet<String>> = Vec::new();
        for batches in &per_file_batches {
            key_sets.push(extract_key_set(batches, on)?);
        }

        // Intersection of all key sets
        let shared_keys = key_sets
            .into_iter()
            .reduce(|a, b| a.intersection(&b).cloned().collect())
            .unwrap_or_default();

        if shared_keys.is_empty() {
            continue;
        }

        // Filter all batches to shared keys, then concat
        let mut filtered: Vec<RecordBatch> = Vec::new();
        for batches in &per_file_batches {
            for batch in *batches {
                let fb = filter_batch_by_keys(batch, on, &shared_keys)?;
                if fb.num_rows() > 0 {
                    filtered.push(fb);
                }
            }
        }

        if !filtered.is_empty() {
            result.push((label.clone(), validate_and_concat(label, &filtered)?));
        }
    }

    // Stamp total row count on the last history entry
    let total_rows: u64 = result.iter().map(|(_, b)| b.num_rows() as u64).sum();
    let mut directory = directory;
    if let Some(last) = directory.cohort_definition.revision_history.last_mut() {
        last.size = Some(total_rows);
    }

    write_datasets(output_path, &directory, &result)
}

// Keep `merge` as an alias for backward compatibility
pub use union as merge;

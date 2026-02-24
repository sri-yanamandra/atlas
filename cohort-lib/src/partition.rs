use std::path::Path;

use arrow::compute::concat_batches;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ArrowReaderMetadata;

use crate::models::{ByteRange, TagDirectory};
use crate::offset_reader::OffsetReader;
use crate::reader::CohortReader;
use crate::writer::CohortWriter;

/// Split a `.cohort` file into N smaller partition files for parallel queries.
///
/// Returns the list of output file paths created.
///
/// # Arguments
/// - `input_path`: Path to the source `.cohort` file
/// - `output_dir`: Directory to write partition files into (created if needed)
/// - `num_partitions`: Explicit partition count (overrides auto-detect)
/// - `target_rows`: Target rows per partition (alternative to `num_partitions`)
pub fn partition(
    input_path: &str,
    output_dir: &str,
    num_partitions: Option<usize>,
    target_rows: Option<usize>,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let reader = CohortReader::open(input_path)?;
    let tags = &reader.directory.tags;

    if tags.is_empty() {
        return Err("No datasets found in input file".into());
    }

    // Determine codec from the first tag (all tags in a file use the same codec)
    let codec = tags[0].codec.clone();

    // For parquet, load metadata to get row group info
    let parquet_meta = if codec == "parquet" {
        let tag = &tags[0];
        let offset_reader = OffsetReader::new(input_path, tag.offset, tag.length);
        Some(ArrowReaderMetadata::load(&offset_reader, Default::default())?)
    } else {
        None
    };

    // Determine total rows and row group count
    let total_rows: usize = tags
        .iter()
        .filter_map(|t| t.num_rows.map(|n| n as usize))
        .max()
        .unwrap_or(0);
    let rg_count = parquet_meta
        .as_ref()
        .map(|m| m.metadata().num_row_groups())
        .unwrap_or(1);

    // Compute N
    let n = compute_partition_count(num_partitions, target_rows, total_rows, rg_count);

    if n <= 1 {
        return Err("File is too small to partition (would produce only 1 file)".into());
    }

    // Create output directory
    std::fs::create_dir_all(output_dir)?;

    // Derive output file stem from input
    let stem = Path::new(input_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("cohort");

    let mut output_paths = Vec::with_capacity(n);

    if codec == "parquet" {
        partition_parquet(
            &reader,
            n,
            rg_count,
            output_dir,
            stem,
            &codec,
            &mut output_paths,
        )?;
    } else {
        partition_non_parquet(&reader, n, total_rows, output_dir, stem, &codec, &mut output_paths)?;
    }

    Ok(output_paths)
}

/// Compute how many partitions to create.
fn compute_partition_count(
    num_partitions: Option<usize>,
    target_rows: Option<usize>,
    total_rows: usize,
    rg_count: usize,
) -> usize {
    if let Some(n) = num_partitions {
        // Cap at row group count for parquet
        return n.min(rg_count).max(1);
    }
    if let Some(target) = target_rows {
        if target == 0 {
            return 1;
        }
        let n = (total_rows + target - 1) / target;
        return n.min(rg_count).max(1);
    }
    // Auto-detect: max(cpus, ceil(rows / 3M)), capped at row group count
    let cpus = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    let rows_based = (total_rows + 3_000_000 - 1) / 3_000_000;
    let n = cpus.max(rows_based);
    n.min(rg_count).max(1)
}

/// Partition a parquet-codec .cohort by distributing row groups.
fn partition_parquet(
    reader: &CohortReader,
    n: usize,
    rg_count: usize,
    output_dir: &str,
    stem: &str,
    codec: &str,
    output_paths: &mut Vec<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Distribute row groups evenly across N partitions
    let rg_per_part = (rg_count + n - 1) / n;

    let mut part_idx = 0;
    let mut rg_start = 0;

    while rg_start < rg_count {
        let rg_end = (rg_start + rg_per_part).min(rg_count);
        let rg_indices: Vec<usize> = (rg_start..rg_end).collect();

        let out_path = format!("{}/{}_part_{:02}.cohort", output_dir, stem, part_idx);

        // Read each dataset's row groups for this partition
        let mut datasets: Vec<(String, RecordBatch)> = Vec::new();
        for tag in &reader.directory.tags {
            let batches: Vec<RecordBatch> = reader
                .get_dataset_data_filtered(&tag.label, None, Some(rg_indices.clone()))?
                .collect::<Result<Vec<_>, _>>()?;

            if batches.is_empty() {
                continue;
            }

            let schema = batches[0].schema();
            let merged = concat_batches(&schema, &batches)?;
            datasets.push((tag.label.clone(), merged));
        }

        if datasets.is_empty() {
            rg_start = rg_end;
            part_idx += 1;
            continue;
        }

        write_partition(&out_path, &reader.directory, &datasets, codec)?;
        output_paths.push(out_path);

        rg_start = rg_end;
        part_idx += 1;
    }

    Ok(())
}

/// Partition non-parquet codecs by slicing rows.
fn partition_non_parquet(
    reader: &CohortReader,
    n: usize,
    total_rows: usize,
    output_dir: &str,
    stem: &str,
    codec: &str,
    output_paths: &mut Vec<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    if total_rows > 100_000_000 {
        eprintln!(
            "Warning: loading {}M rows into memory for non-parquet partitioning",
            total_rows / 1_000_000
        );
    }

    // Read all datasets into memory
    let mut all_data: Vec<(String, RecordBatch)> = Vec::new();
    for tag in &reader.directory.tags {
        let batches: Vec<RecordBatch> = reader
            .get_dataset_data(&tag.label)?
            .collect::<Result<Vec<_>, _>>()?;
        if batches.is_empty() {
            continue;
        }
        let schema = batches[0].schema();
        let merged = concat_batches(&schema, &batches)?;
        all_data.push((tag.label.clone(), merged));
    }

    let rows_per_part = (total_rows + n - 1) / n;

    for part_idx in 0..n {
        let start = part_idx * rows_per_part;
        if start >= total_rows {
            break;
        }
        let len = rows_per_part.min(total_rows - start);

        let out_path = format!("{}/{}_part_{:02}.cohort", output_dir, stem, part_idx);

        let datasets: Vec<(String, RecordBatch)> = all_data
            .iter()
            .map(|(label, batch)| {
                let batch_len = batch.num_rows();
                let s = start.min(batch_len);
                let l = len.min(batch_len.saturating_sub(s));
                (label.clone(), batch.slice(s, l))
            })
            .filter(|(_, b)| b.num_rows() > 0)
            .collect();

        if datasets.is_empty() {
            continue;
        }

        write_partition(&out_path, &reader.directory, &datasets, codec)?;
        output_paths.push(out_path);
    }

    Ok(())
}

/// Write a single partition file with the same cohort definition as the source.
fn write_partition(
    path: &str,
    source_directory: &TagDirectory,
    datasets: &[(String, RecordBatch)],
    codec: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let directory = TagDirectory {
        tags: vec![], // Writer computes tags from data blocks
        patient_index: ByteRange {
            offset: 0,
            length: 0,
        },
        cohort_definition: source_directory.cohort_definition.clone(),
        backend: source_directory.backend.clone(),
    };

    let datasets_ref: Vec<(&str, &RecordBatch)> = datasets
        .iter()
        .map(|(label, batch)| (label.as_str(), batch))
        .collect();

    CohortWriter::write_with_codec(path, &directory, &datasets_ref, codec)?;
    Ok(())
}

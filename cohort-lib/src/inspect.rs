use std::fs;
use std::io::{Read, Seek, SeekFrom};

use arrow::ipc::reader::FileReader as IpcFileReader;
use bytes::Bytes;
use parquet::file::metadata::ParquetMetaDataReader;
use serde::Serialize;

use crate::models::IcebergBackend;
use crate::offset_reader::OffsetReadSeek;
use crate::reader::CohortReader;

#[derive(Serialize, Debug, Clone)]
pub struct CohortInspection {
    pub file_metadata: FileMetadata,
    pub cohort_definition: CohortDefinitionSummary,
    pub datasets: Vec<DatasetInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<IcebergBackend>,
}

#[derive(Serialize, Debug, Clone)]
pub struct FileMetadata {
    pub path: String,
    pub file_size_bytes: u64,
    pub format_version: u8,
    pub num_datasets: usize,
}

#[derive(Serialize, Debug, Clone)]
pub struct CohortDefinitionSummary {
    pub name: String,
    pub description: Option<String>,
    pub created_at: String,
    pub created_by: String,
    pub revision: u32,
    pub selectors: Vec<SelectorSummary>,
    pub lineage: LineageSummary,
    pub revision_history: Vec<RevisionSummary>,
    pub primary_keys: Vec<String>,
}

#[derive(Serialize, Debug, Clone)]
pub struct SelectorSummary {
    pub selector_id: String,
    pub field: String,
    pub operator: String,
    pub values: serde_json::Value,
    pub label: Option<String>,
    pub is_exclusion: bool,
}

#[derive(Serialize, Debug, Clone)]
pub struct LineageSummary {
    pub source_system: Vec<String>,
    pub source_tables: Vec<String>,
    pub extraction_date: String,
}

#[derive(Serialize, Debug, Clone)]
pub struct RevisionSummary {
    pub revision: u32,
    pub author: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
}

#[derive(Serialize, Debug, Clone)]
pub struct DatasetInfo {
    pub label: String,
    pub codec: String,
    pub compressed_size: u64,
    pub num_rows: i64,
    pub num_columns: usize,
    pub columns: Vec<ColumnInfo>,
}

#[derive(Serialize, Debug, Clone)]
pub struct ColumnInfo {
    pub name: String,
    pub data_type: String,
    pub nullable: bool,
}

pub fn inspect(path: &str) -> Result<CohortInspection, Box<dyn std::error::Error>> {
    let reader = CohortReader::open(path)?;
    let file_size = fs::metadata(path)?.len();
    let def = reader.cohort_definition();

    let selectors: Vec<SelectorSummary> = def
        .selectors
        .iter()
        .map(|s| SelectorSummary {
            selector_id: s.selector_id.clone(),
            field: s.field.clone(),
            operator: s.operator.clone(),
            values: s.values.clone(),
            label: s.label.clone(),
            is_exclusion: s.is_exclusion,
        })
        .collect();

    let revision_history: Vec<RevisionSummary> = def
        .revision_history
        .iter()
        .map(|r| RevisionSummary {
            revision: r.revision,
            author: r.author.clone(),
            message: r.message.clone(),
            size: r.size,
        })
        .collect();

    let cohort_definition = CohortDefinitionSummary {
        name: def.name.clone(),
        description: def.description.clone(),
        created_at: def.created_at.to_rfc3339(),
        created_by: def.created_by.clone(),
        revision: def.revision,
        selectors,
        lineage: LineageSummary {
            source_system: def.lineage.source_system.clone(),
            source_tables: def.lineage.source_tables.clone(),
            extraction_date: def.lineage.extraction_date.clone(),
        },
        revision_history,
        primary_keys: def.primary_keys.clone(),
    };

    // Build dataset info from the tag directory.  Files written by the
    // current writer have num_rows/columns stamped directly in each Tag,
    // so we never need to touch Parquet at all.  For older files that lack
    // these fields we fall back to reading the Parquet footer.
    let mut datasets = Vec::new();
    for tag in &reader.directory.tags {
        let (num_rows, num_columns, columns) =
            if let (Some(nr), Some(nc), Some(cols)) =
                (tag.num_rows, tag.num_columns, tag.columns.as_ref())
            {
                // Fast path: everything is in the tag directory already
                let cols = cols
                    .iter()
                    .map(|c| ColumnInfo {
                        name: c.name.clone(),
                        data_type: c.data_type.clone(),
                        nullable: c.nullable,
                    })
                    .collect();
                (nr, nc, cols)
            } else {
                // Fallback for files written before schema was stamped in tags
                match tag.codec.as_str() {
                    "arrow_ipc" | "feather" => {
                        read_ipc_footer_info(path, tag.offset, tag.length)?
                    }
                    _ => {
                        read_parquet_footer_info(path, tag.offset, tag.length)?
                    }
                }
            };

        datasets.push(DatasetInfo {
            label: tag.label.clone(),
            codec: tag.codec.clone(),
            compressed_size: tag.length,
            num_rows,
            num_columns,
            columns,
        });
    }

    let file_metadata = FileMetadata {
        path: path.to_string(),
        file_size_bytes: file_size,
        format_version: crate::FORMAT_VERSION,
        num_datasets: datasets.len(),
    };

    Ok(CohortInspection {
        file_metadata,
        cohort_definition,
        datasets,
        backend: reader.directory.backend.clone(),
    })
}

/// Only used for files written before schema info was stamped in the tag directory.
fn read_parquet_footer_info(
    path: &str,
    block_offset: u64,
    block_length: u64,
) -> Result<(i64, usize, Vec<ColumnInfo>), Box<dyn std::error::Error>> {
    let mut file = fs::File::open(path)?;
    let block_end = block_offset + block_length;

    // Read the last 8 bytes: 4-byte footer length + 4-byte magic
    file.seek(SeekFrom::Start(block_end - 8))?;
    let mut tail = [0u8; 8];
    file.read_exact(&mut tail)?;
    let footer_len = i32::from_le_bytes(tail[0..4].try_into().unwrap()) as usize;

    // Read just the footer + the 8-byte suffix
    let total = footer_len + 8;
    file.seek(SeekFrom::Start(block_end - total as u64))?;
    let mut footer_buf = vec![0u8; total];
    file.read_exact(&mut footer_buf)?;

    let parquet_meta = ParquetMetaDataReader::new()
        .parse_and_finish(&Bytes::from(footer_buf))?;
    let file_meta = parquet_meta.file_metadata();

    let num_rows: i64 = (0..parquet_meta.num_row_groups())
        .map(|i| parquet_meta.row_group(i).num_rows())
        .sum();

    let schema = file_meta.schema_descr();
    let num_columns = schema.num_columns();
    let columns: Vec<ColumnInfo> = (0..num_columns)
        .map(|i| {
            let col = schema.column(i);
            ColumnInfo {
                name: col.name().to_string(),
                data_type: format!("{:?}", col.physical_type()),
                nullable: col.self_type().is_optional(),
            }
        })
        .collect();

    Ok((num_rows, num_columns, columns))
}

/// Fallback for IPC/Feather files written before schema info was stamped in tags.
fn read_ipc_footer_info(
    path: &str,
    block_offset: u64,
    block_length: u64,
) -> Result<(i64, usize, Vec<ColumnInfo>), Box<dyn std::error::Error>> {
    let read_seek = OffsetReadSeek::new(path, block_offset, block_length)?;
    let reader = IpcFileReader::try_new(read_seek, None)?;

    let schema = reader.schema();
    let num_columns = schema.fields().len();
    let columns: Vec<ColumnInfo> = schema
        .fields()
        .iter()
        .map(|f| ColumnInfo {
            name: f.name().clone(),
            data_type: format!("{}", f.data_type()),
            nullable: f.is_nullable(),
        })
        .collect();

    let num_rows: i64 = reader.map(|b| b.map(|b| b.num_rows() as i64).unwrap_or(0)).sum();

    Ok((num_rows, num_columns, columns))
}

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

use arrow::record_batch::RecordBatch;
use duckdb::Connection;

use crate::catalog::{dataset_to_table_name, CohortCatalog};
use crate::reader::CohortReader;

pub struct DuckDbEngine {
    conn: Connection,
    tables: Vec<String>,
    /// Temp dir holding extracted parquet files — dropped when engine is dropped.
    _tmp_dir: tempfile::TempDir,
}

impl DuckDbEngine {
    pub fn new(catalog: CohortCatalog) -> Result<Self, Box<dyn std::error::Error>> {
        let conn = Connection::open_in_memory()?;
        let tmp_dir = tempfile::tempdir()?;

        let mut tables = Vec::new();
        // table_name -> list of parquet file paths
        let mut parquet_paths: std::collections::HashMap<String, Vec<PathBuf>> =
            std::collections::HashMap::new();

        for file in &catalog.files {
            let base_path = std::path::Path::new(&catalog.base_path);
            let file_path = base_path.join(&file.file_path);

            let reader = CohortReader::open(file_path.to_str().unwrap())?;

            for dataset in &file.datasets {
                let table_name = dataset_to_table_name(&dataset.label);

                // Find the tag to get offset/length
                let tag = reader
                    .directory
                    .tags
                    .iter()
                    .find(|t| t.label == dataset.label)
                    .ok_or_else(|| format!("No tag for dataset '{}'", dataset.label))?;

                if tag.codec != "parquet" {
                    // Non-parquet codecs: fall back to reading through Arrow
                    // and writing a temp parquet file
                    let iter = reader.get_dataset_data(&dataset.label)?;
                    let batches: Vec<RecordBatch> = iter.collect::<Result<Vec<_>, _>>()?;
                    if batches.is_empty() {
                        continue;
                    }
                    let tmp_path = tmp_dir.path().join(format!(
                        "{}_{}.parquet",
                        table_name,
                        parquet_paths.entry(table_name.clone()).or_default().len()
                    ));
                    let out_file = std::fs::File::create(&tmp_path)?;
                    let mut writer = parquet::arrow::ArrowWriter::try_new(
                        out_file,
                        batches[0].schema(),
                        None,
                    )?;
                    for batch in &batches {
                        writer.write(batch)?;
                    }
                    writer.close()?;
                    parquet_paths
                        .entry(table_name)
                        .or_default()
                        .push(tmp_path);
                    continue;
                }

                // Parquet codec: extract the raw parquet bytes to a temp file
                let tmp_path = tmp_dir.path().join(format!(
                    "{}_{}.parquet",
                    table_name,
                    parquet_paths.entry(table_name.clone()).or_default().len()
                ));
                let mut src = std::fs::File::open(&file_path)?;
                src.seek(SeekFrom::Start(tag.offset))?;
                let mut dst = std::fs::File::create(&tmp_path)?;
                let mut remaining = tag.length as usize;
                let mut buf = vec![0u8; 64 * 1024];
                while remaining > 0 {
                    let to_read = remaining.min(buf.len());
                    let n = src.read(&mut buf[..to_read])?;
                    if n == 0 {
                        break;
                    }
                    dst.write_all(&buf[..n])?;
                    remaining -= n;
                }
                parquet_paths
                    .entry(table_name)
                    .or_default()
                    .push(tmp_path);
            }
        }

        // Create DuckDB views over the extracted parquet files
        for (table_name, paths) in &parquet_paths {
            let path_list: Vec<String> = paths
                .iter()
                .map(|p| p.to_str().unwrap().replace('\'', "''"))
                .collect();

            let quoted: Vec<String> = path_list.iter().map(|p| format!("'{}'", p)).collect();
            let sql = format!(
                "CREATE VIEW \"{}\" AS SELECT * FROM read_parquet([{}], union_by_name=true)",
                table_name,
                quoted.join(", ")
            );
            conn.execute_batch(&sql)?;
            tables.push(table_name.clone());
        }

        tables.sort();
        Ok(Self {
            conn,
            tables,
            _tmp_dir: tmp_dir,
        })
    }

    /// Execute a SQL query and return Arrow RecordBatches.
    pub fn query(&self, sql: &str) -> Result<Vec<RecordBatch>, Box<dyn std::error::Error>> {
        let mut stmt = self.conn.prepare(sql)?;
        let arrow_result = stmt.query_arrow([])?;
        let batches: Vec<RecordBatch> = arrow_result.collect();
        Ok(batches)
    }

    /// List all registered table names.
    pub fn list_tables(&self) -> Vec<String> {
        self.tables.clone()
    }
}

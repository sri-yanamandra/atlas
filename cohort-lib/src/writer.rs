use std::fs::File;
use std::io::{Cursor, Seek, SeekFrom, Write};

use arrow::ipc::writer::{FileWriter as IpcFileWriter, IpcWriteOptions};
use arrow::ipc::CompressionType;
use arrow::record_batch::RecordBatch;
use byteorder::{LittleEndian, WriteBytesExt};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use vortex::array::arrow::FromArrowArray;
use vortex::file::WriteOptionsSessionExt;
use vortex::VortexSessionDefault;

use crate::bitmap_index::{build_bitmap, serialize_bitmap};
use crate::models::{ByteRange, Tag, TagColumnInfo, TagDirectory};

pub struct CohortWriter;

impl CohortWriter {
    /// Write a `.cohort` file to disk using the default Parquet codec.
    pub fn write(
        path: &str,
        directory: &TagDirectory,
        datasets: &[(&str, &RecordBatch)],
    ) -> Result<(), Box<dyn std::error::Error>> {
        Self::write_with_codec(path, directory, datasets, "parquet")
    }

    /// Write a `.cohort` file to disk with the specified codec and optional patient index.
    ///
    /// Supported codecs: `"parquet"`, `"arrow_ipc"`, `"feather"`, `"vortex"`.
    pub fn write_with_codec(
        path: &str,
        directory: &TagDirectory,
        datasets: &[(&str, &RecordBatch)],
        codec: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Self::write_with_codec_and_index(path, directory, datasets, codec, None)
    }

    /// Write a `.cohort` file with an optional bitmap patient index.
    ///
    /// When `index_column` is `Some`, a Roaring bitmap of the key column values
    /// is embedded after the data blocks and recorded in `patient_index`.
    pub fn write_with_codec_and_index(
        path: &str,
        directory: &TagDirectory,
        datasets: &[(&str, &RecordBatch)],
        codec: &str,
        index_column: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(path)?;

        // 1. Write magic, version, and placeholder for dir_offset
        file.write_all(b"COHORT")?;
        file.write_u8(crate::FORMAT_VERSION)?;
        let dir_offset_pos = file.stream_position()?;
        file.write_u64::<LittleEndian>(0)?; // placeholder

        // 2. Write each data block, recording offsets
        let mut tags: Vec<Tag> = Vec::new();
        for (label, batch) in datasets {
            let offset = file.stream_position()?;

            match codec {
                "parquet" => {
                    let props = WriterProperties::builder()
                        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
                        .set_max_row_group_size(1_000_000)
                        .set_statistics_enabled(
                            parquet::file::properties::EnabledStatistics::Page,
                        )
                        .build();
                    let mut writer =
                        ArrowWriter::try_new(&mut file, batch.schema(), Some(props))?;
                    writer.write(batch)?;
                    writer.close()?;
                }
                "arrow_ipc" => {
                    let mut writer =
                        IpcFileWriter::try_new(&mut file, &batch.schema())?;
                    writer.write(batch)?;
                    writer.finish()?;
                }
                "feather" => {
                    let opts = IpcWriteOptions::default()
                        .try_with_compression(Some(CompressionType::ZSTD))?;
                    let mut writer = IpcFileWriter::try_new_with_options(
                        &mut file,
                        &batch.schema(),
                        opts,
                    )?;
                    writer.write(batch)?;
                    writer.finish()?;
                }
                "vortex" => {
                    Self::write_vortex_block(&mut file, batch)?;
                }
                _ => {
                    return Err(format!("Unsupported codec: {codec}").into());
                }
            }

            let length = file.stream_position()? - offset;

            // Stamp schema info so inspect() never needs to parse the data block
            let schema = batch.schema();
            let columns: Vec<TagColumnInfo> = schema
                .fields()
                .iter()
                .map(|f| TagColumnInfo {
                    name: f.name().clone(),
                    data_type: format!("{}", f.data_type()),
                    nullable: f.is_nullable(),
                })
                .collect();

            tags.push(Tag {
                tag_id: 0,
                label: label.to_string(),
                codec: codec.to_string(),
                offset,
                length,
                schema_type: None,
                num_rows: Some(batch.num_rows() as i64),
                num_columns: Some(columns.len()),
                columns: Some(columns),
            });
        }

        // 2b. Optionally write the bitmap patient index block
        let patient_index = if let Some(key_col) = index_column {
            let batches: Vec<&RecordBatch> = datasets.iter().map(|(_, b)| *b).collect();
            let bitmap = build_bitmap(&batches.iter().map(|b| (*b).clone()).collect::<Vec<_>>(), key_col)?;
            let bitmap_bytes = serialize_bitmap(&bitmap);
            let offset = file.stream_position()?;
            file.write_all(&bitmap_bytes)?;
            let length = file.stream_position()? - offset;
            ByteRange { offset, length }
        } else {
            directory.patient_index.clone()
        };

        // 3. Build the final directory with computed tags.
        //    When datasets are provided the writer computes tags from block
        //    offsets.  For metadata-only files (e.g. Iceberg-backed cohorts)
        //    the caller pre-populates tags in the directory; preserve those
        //    when no data blocks were written.
        let final_tags = if tags.is_empty() {
            directory.tags.clone()
        } else {
            tags
        };
        let total_rows: u64 = if datasets.is_empty() {
            final_tags.iter().filter_map(|t| t.num_rows.map(|n| n as u64)).sum()
        } else {
            datasets.iter().map(|(_, b)| b.num_rows() as u64).sum()
        };
        let mut cohort_definition = directory.cohort_definition.clone();
        // Record the index column as a primary key so readers can discover it
        if let Some(col) = index_column {
            if !cohort_definition.primary_keys.contains(&col.to_string()) {
                cohort_definition.primary_keys = vec![col.to_string()];
            }
        }
        // Stamp size on the latest revision entry if not already set
        if let Some(last) = cohort_definition.revision_history.last_mut() {
            if last.size.is_none() {
                last.size = Some(total_rows);
            }
        }
        let final_directory = TagDirectory {
            tags: final_tags,
            patient_index,
            cohort_definition,
            backend: directory.backend.clone(),
        };

        // 4. Write the Tag Directory as compact JSON
        let dir_offset = file.stream_position()?;
        let dir_json = serde_json::to_vec(&final_directory)?;
        file.write_all(&dir_json)?;

        // 5. Back-patch the dir_offset in the header
        file.seek(SeekFrom::Start(dir_offset_pos))?;
        file.write_u64::<LittleEndian>(dir_offset)?;

        Ok(())
    }

    /// Write a RecordBatch as a Vortex data block into the file.
    fn write_vortex_block(
        file: &mut File,
        batch: &RecordBatch,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use vortex::io::session::RuntimeSessionExt;

        let rt = tokio::runtime::Runtime::new()?;

        // Convert RecordBatch → Vortex array
        let vortex_array = vortex::array::ArrayRef::from_arrow(batch, false);

        // Write to an in-memory buffer first, then copy to the file.
        let mut buf: Vec<u8> = Vec::new();

        let _guard = rt.enter();
        let session = vortex::session::VortexSession::default().with_tokio();
        rt.block_on(async {
            session
                .write_options()
                .write(Cursor::new(&mut buf), vortex_array.to_array_stream())
                .await
        })?;

        file.write_all(&buf)?;
        Ok(())
    }
}

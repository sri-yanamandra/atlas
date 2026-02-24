use std::fs::File;
use std::io::{Read, Seek};

use arrow::ipc::reader::FileReader as IpcFileReader;
use arrow::record_batch::RecordBatch;
use byteorder::{LittleEndian, ReadBytesExt};
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ParquetRecordBatchReaderBuilder};
use parquet::arrow::ProjectionMask;

use crate::models::{CohortDefinition, TagDirectory};
use crate::offset_reader::{OffsetReadSeek, OffsetReader};

#[derive(Debug)]
pub struct CohortReader {
    pub directory: TagDirectory,
    file_path: String,
}

impl CohortReader {
    pub fn open(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;

        // 1. Validate magic bytes
        let mut magic = [0u8; 6];
        file.read_exact(&mut magic)?;
        if &magic != b"COHORT" {
            return Err("Invalid file: missing COHORT magic bytes".into());
        }

        // 2. Read version
        let version = file.read_u8()?;
        if version != crate::FORMAT_VERSION {
            return Err(format!("Unsupported format version: {version}").into());
        }

        // 3. Read directory offset and seek to it
        let dir_offset = file.read_u64::<LittleEndian>()?;
        file.seek(std::io::SeekFrom::Start(dir_offset))?;

        // 4. Parse the Tag Directory (JSON to end of file)
        let mut dir_json = String::new();
        file.read_to_string(&mut dir_json)?;
        let directory: TagDirectory = serde_json::from_str(&dir_json)?;

        Ok(Self {
            directory,
            file_path: path.to_string(),
        })
    }

    /// Look up a dataset tag by label and return an Arrow RecordBatch iterator.
    pub fn get_dataset_data(
        &self,
        label: &str,
    ) -> Result<
        Box<dyn Iterator<Item = arrow::error::Result<arrow::record_batch::RecordBatch>>>,
        Box<dyn std::error::Error>,
    > {
        let tag = self
            .directory
            .tags
            .iter()
            .find(|t| t.label == label)
            .ok_or_else(|| format!("No tag with label '{label}'"))?;

        match tag.codec.as_str() {
            "arrow_ipc" | "feather" => {
                let read_seek =
                    OffsetReadSeek::new(&self.file_path, tag.offset, tag.length)?;
                let reader = IpcFileReader::try_new(read_seek, None)?;
                Ok(Box::new(reader))
            }
            "vortex" => {
                let batches = Self::read_vortex_block(&self.file_path, tag.offset, tag.length)?;
                Ok(Box::new(batches.into_iter().map(Ok)))
            }
            _ => {
                // Default to parquet (including "parquet" and any legacy values)
                let offset_reader =
                    OffsetReader::new(&self.file_path, tag.offset, tag.length);
                let builder = ParquetRecordBatchReaderBuilder::try_new(offset_reader)?;
                let reader = builder.build()?;
                Ok(Box::new(reader))
            }
        }
    }

    /// Read a dataset's data block with column projection pushed down.
    ///
    /// - `projection`: Arrow column indices (0-based). `None` = all columns.
    /// - `row_groups`: Parquet row group indices to read. `None` = all groups. Ignored for IPC/Vortex.
    pub fn get_dataset_data_filtered(
        &self,
        label: &str,
        projection: Option<&[usize]>,
        row_groups: Option<Vec<usize>>,
    ) -> Result<
        Box<dyn Iterator<Item = arrow::error::Result<arrow::record_batch::RecordBatch>>>,
        Box<dyn std::error::Error>,
    > {
        let tag = self
            .directory
            .tags
            .iter()
            .find(|t| t.label == label)
            .ok_or_else(|| format!("No tag with label '{label}'"))?;

        match tag.codec.as_str() {
            "arrow_ipc" | "feather" => {
                let read_seek =
                    OffsetReadSeek::new(&self.file_path, tag.offset, tag.length)?;
                let proj_vec = projection.map(|p| p.to_vec());
                let reader = IpcFileReader::try_new(read_seek, proj_vec)?;
                Ok(Box::new(reader))
            }
            "vortex" => {
                // Vortex: read all then project client-side
                let batches = Self::read_vortex_block(&self.file_path, tag.offset, tag.length)?;
                let projected = if let Some(proj) = projection {
                    batches
                        .into_iter()
                        .map(|b| b.project(proj).map_err(|e| e.into()))
                        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?
                } else {
                    batches
                };
                Ok(Box::new(projected.into_iter().map(Ok)))
            }
            _ => {
                let offset_reader =
                    OffsetReader::new(&self.file_path, tag.offset, tag.length);
                let mut builder =
                    ParquetRecordBatchReaderBuilder::try_new(offset_reader)?;
                if let Some(proj) = projection {
                    let mask = ProjectionMask::roots(
                        builder.parquet_schema(),
                        proj.iter().copied(),
                    );
                    builder = builder.with_projection(mask);
                }
                if let Some(rg) = row_groups {
                    builder = builder.with_row_groups(rg);
                }
                Ok(Box::new(builder.build()?))
            }
        }
    }

    /// Read a dataset's Parquet block using pre-parsed metadata (avoids re-reading the footer).
    /// For non-Parquet codecs, delegates to `get_dataset_data_filtered()`.
    pub fn get_dataset_data_with_metadata(
        &self,
        label: &str,
        metadata: ArrowReaderMetadata,
        projection: Option<&[usize]>,
        row_groups: Option<Vec<usize>>,
    ) -> Result<
        Box<dyn Iterator<Item = arrow::error::Result<arrow::record_batch::RecordBatch>>>,
        Box<dyn std::error::Error>,
    > {
        let tag = self
            .directory
            .tags
            .iter()
            .find(|t| t.label == label)
            .ok_or_else(|| format!("No tag with label '{label}'"))?;

        match tag.codec.as_str() {
            "arrow_ipc" | "feather" | "vortex" => {
                self.get_dataset_data_filtered(label, projection, row_groups)
            }
            _ => {
                let offset_reader =
                    OffsetReader::new(&self.file_path, tag.offset, tag.length);
                let mut builder =
                    ParquetRecordBatchReaderBuilder::new_with_metadata(
                        offset_reader, metadata,
                    );
                if let Some(proj) = projection {
                    let mask = ProjectionMask::roots(
                        builder.parquet_schema(),
                        proj.iter().copied(),
                    );
                    builder = builder.with_projection(mask);
                }
                if let Some(rg) = row_groups {
                    builder = builder.with_row_groups(rg);
                }
                Ok(Box::new(builder.build()?))
            }
        }
    }

    /// Read the raw bitmap index bytes from the file, if present.
    pub fn read_bitmap_bytes(&self) -> Result<Option<Vec<u8>>, Box<dyn std::error::Error>> {
        let pi = &self.directory.patient_index;
        if pi.length == 0 {
            return Ok(None);
        }
        let mut file = File::open(&self.file_path)?;
        file.seek(std::io::SeekFrom::Start(pi.offset))?;
        let mut buf = vec![0u8; pi.length as usize];
        file.read_exact(&mut buf)?;
        Ok(Some(buf))
    }

    /// Return the current cohort definition (selectors at latest revision).
    pub fn cohort_definition(&self) -> &CohortDefinition {
        &self.directory.cohort_definition
    }

    /// Read a Vortex data block from the file and return as RecordBatches.
    fn read_vortex_block(
        file_path: &str,
        offset: u64,
        length: u64,
    ) -> Result<Vec<RecordBatch>, Box<dyn std::error::Error>> {
        use vortex::array::arrow::IntoArrowArray;
        use vortex::file::OpenOptionsSessionExt;
        use vortex::VortexSessionDefault;

        // Read the vortex bytes into memory
        let mut file = File::open(file_path)?;
        file.seek(std::io::SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; length as usize];
        file.read_exact(&mut buf)?;

        let rt = tokio::runtime::Runtime::new()?;
        let _guard = rt.enter();
        let session = {
            use vortex::io::session::RuntimeSessionExt;
            vortex::session::VortexSession::default().with_tokio()
        };

        let batches = rt.block_on(async {
            let vortex_file = session
                .open_options()
                .open_buffer(buf)
                .map_err(|e| -> Box<dyn std::error::Error> { format!("Vortex open error: {e}").into() })?;

            let stream = vortex_file
                .scan()
                .map_err(|e| -> Box<dyn std::error::Error> { format!("Vortex scan error: {e}").into() })?
                .into_array_stream()
                .map_err(|e| -> Box<dyn std::error::Error> { format!("Vortex stream error: {e}").into() })?;

            use futures::TryStreamExt;
            let arrays: Vec<vortex::array::ArrayRef> = stream
                .try_collect::<Vec<_>>()
                .await
                .map_err(|e| -> Box<dyn std::error::Error> { format!("Vortex collect error: {e}").into() })?;

            let mut result = Vec::new();
            for array in arrays {
                let arrow_array = IntoArrowArray::into_arrow_preferred(array)
                    .map_err(|e| -> Box<dyn std::error::Error> { format!("Vortex to Arrow error: {e}").into() })?;
                let struct_array = arrow_array
                    .as_any()
                    .downcast_ref::<arrow::array::StructArray>()
                    .ok_or_else(|| -> Box<dyn std::error::Error> { "Vortex array is not a StructArray".into() })?;
                result.push(RecordBatch::from(struct_array));
            }
            Ok::<Vec<RecordBatch>, Box<dyn std::error::Error>>(result)
        })?;

        Ok(batches)
    }
}

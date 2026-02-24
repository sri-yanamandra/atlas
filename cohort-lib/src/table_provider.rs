use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::stats::Precision;
use datafusion::datasource::TableProvider;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::logical_expr::{BinaryExpr, Expr, Operator, TableProviderFilterPushDown, TableType};
use datafusion::scalar::ScalarValue;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream, Statistics,
};
use futures::stream;
use arrow::ipc::reader::FileReader as IpcFileReader;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ParquetRecordBatchReaderBuilder};
use parquet::file::metadata::RowGroupMetaData;
use parquet::file::statistics::Statistics as ParquetStats;

use roaring::RoaringTreemap;

use crate::bitmap_index::{bitmap_can_skip_file, deserialize_bitmap};
use crate::catalog::{CatalogColumn, CohortCatalog};
use crate::models::Selector;
use crate::offset_reader::{OffsetReadSeek, OffsetReader};
use crate::reader::CohortReader;
use crate::selector_pruner::{can_skip_file, extract_key_filter_values};

// ── Per-file metadata kept at scan time ────────────────────────

#[derive(Debug, Clone)]
struct FileInfo {
    path: String,
    selectors: Vec<Selector>,
    logic_operator: String,
    num_rows: i64,
    row_groups: Option<Vec<usize>>,
}

// ── TableProvider ──────────────────────────────────────────────

pub struct CohortTableProvider {
    schema: SchemaRef,
    dataset_label: String,
    files: Vec<FileInfo>,
    /// Parquet metadata loaded once at provider creation, indexed parallel to `files`.
    cached_metadata: Vec<Option<ArrowReaderMetadata>>,
    /// Deserialized bitmaps loaded once at provider creation, parallel to `files`.
    cached_bitmaps: Vec<Option<RoaringTreemap>>,
    /// Primary key column name discovered from file metadata (if any).
    primary_key_col: Option<String>,
    /// Whether the primary key column is a string type (determines hash vs direct lookup).
    primary_key_is_string: bool,
}

impl CohortTableProvider {
    /// Create a provider for one dataset across all catalog files that contain it.
    pub fn new(
        catalog: &CohortCatalog,
        dataset_label: &str,
    ) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        let entries = catalog.entries_for_dataset(dataset_label);
        if entries.is_empty() {
            return Err(format!("No files contain dataset '{dataset_label}'").into());
        }

        // Collect schemas from all files and unify to the widest types.
        let mut schemas: Vec<Schema> = Vec::new();
        for entry in &entries {
            let s = entry
                .datasets
                .iter()
                .find(|m| m.label == dataset_label)
                .and_then(|m| {
                    if m.schema.is_empty() {
                        None
                    } else {
                        schema_from_catalog_columns(&m.schema)
                    }
                })
                .or_else(|| {
                    let path = catalog.resolve_path(entry);
                    read_dataset_schema(path.to_str().unwrap(), dataset_label)
                        .ok()
                        .map(|s| s.as_ref().clone())
                });
            if let Some(s) = s {
                schemas.push(s);
            }
        }
        let schema: SchemaRef = if schemas.is_empty() {
            let first_path = catalog.resolve_path(entries[0]);
            read_dataset_schema(first_path.to_str().unwrap(), dataset_label)
                .expect("Failed to read schema from Parquet")
        } else {
            Arc::new(widen_schemas(&schemas))
        };

        let files: Vec<FileInfo> = entries
            .iter()
            .map(|e| {
                let nr = e
                    .datasets
                    .iter()
                    .find(|m| m.label == dataset_label)
                    .map(|m| m.num_rows)
                    .unwrap_or(0);
                FileInfo {
                    path: catalog.resolve_path(e).to_str().unwrap().to_string(),
                    selectors: e.selectors.clone(),
                    logic_operator: e.logic_operator.clone(),
                    num_rows: nr,
                    row_groups: None,
                }
            })
            .collect();

        // Pre-load Parquet footer metadata and bitmaps once so queries never re-read them.
        let mut cached_metadata: Vec<Option<ArrowReaderMetadata>> = Vec::with_capacity(files.len());
        let mut cached_bitmaps: Vec<Option<RoaringTreemap>> = Vec::with_capacity(files.len());
        let mut primary_key_col: Option<String> = None;
        let mut primary_key_is_string = false;

        for f in &files {
            let reader = CohortReader::open(&f.path).ok();

            // Parquet metadata
            let meta = reader.as_ref().and_then(|r| {
                let tag = r.directory.tags.iter().find(|t| t.label == dataset_label)?;
                if tag.codec != "parquet" {
                    return None;
                }
                let offset_reader = OffsetReader::new(&f.path, tag.offset, tag.length);
                ArrowReaderMetadata::load(&offset_reader, Default::default()).ok()
            });
            cached_metadata.push(meta);

            // Bitmap
            let bitmap = reader.as_ref().and_then(|r| {
                let bytes = r.read_bitmap_bytes().ok()??;
                deserialize_bitmap(&bytes).ok()
            });
            cached_bitmaps.push(bitmap);

            // Discover primary key column from the first file that has one
            if primary_key_col.is_none() {
                if let Some(r) = &reader {
                    if let Some(pk) = r.directory.cohort_definition.primary_keys.first() {
                        primary_key_col = Some(pk.clone());
                        // Determine if the key column is string-typed
                        primary_key_is_string = schema
                            .field_with_name(&pk.to_lowercase())
                            .map(|field| matches!(field.data_type(), DataType::Utf8 | DataType::LargeUtf8))
                            .unwrap_or(false);
                    }
                }
            }
        }

        Ok(Self {
            schema,
            dataset_label: dataset_label.to_string(),
            files,
            cached_metadata,
            cached_bitmaps,
            primary_key_col,
            primary_key_is_string,
        })
    }
}

#[async_trait]
impl TableProvider for CohortTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        // We use filters as hints for file pruning; DataFusion still applies them.
        Ok(filters
            .iter()
            .map(|_| TableProviderFilterPushDown::Inexact)
            .collect())
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Build a single AND predicate from all filters.
        let predicate = filters
            .iter()
            .cloned()
            .reduce(|a, b| a.and(b));

        // Extract bitmap filter values before pruning (if a primary key exists).
        let bitmap_filter_values = predicate.as_ref().and_then(|pred| {
            self.primary_key_col.as_ref().and_then(|pk| {
                extract_key_filter_values(pred, pk, self.primary_key_is_string)
            })
        });

        // Prune files whose selectors contradict the predicate,
        // then prune files whose bitmap proves none of the queried keys are present.
        let (surviving_files, surviving_meta): (Vec<FileInfo>, Vec<Option<ArrowReaderMetadata>>) =
            self.files
                .iter()
                .zip(self.cached_metadata.iter())
                .zip(self.cached_bitmaps.iter())
                .filter(|((f, _), _)| {
                    if let Some(ref pred) = predicate {
                        !can_skip_file(&f.selectors, &f.logic_operator, pred)
                    } else {
                        true
                    }
                })
                .filter(|((_, _), bitmap)| {
                    if let Some(ref values) = bitmap_filter_values {
                        match bitmap {
                            Some(bm) => !bitmap_can_skip_file(bm, values),
                            None => true, // no bitmap → conservative, keep the file
                        }
                    } else {
                        true
                    }
                })
                .map(|((f, m), _)| (f.clone(), m.clone()))
                .unzip();

        let projected_schema = if let Some(proj) = projection {
            Arc::new(self.schema.project(proj)?)
        } else {
            self.schema.clone()
        };

        Ok(Arc::new(CohortExec::new(
            projected_schema,
            self.dataset_label.clone(),
            surviving_files,
            surviving_meta,
            projection.cloned(),
            _limit,
            predicate,
        )))
    }
}

impl fmt::Debug for CohortTableProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CohortTableProvider")
            .field("dataset", &self.dataset_label)
            .field("files", &self.files.len())
            .finish()
    }
}

// ── ExecutionPlan ──────────────────────────────────────────────

struct CohortExec {
    schema: SchemaRef,
    dataset_label: String,
    files: Vec<FileInfo>,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    properties: PlanProperties,
    total_num_rows: usize,
    cached_metadata: Vec<Option<ArrowReaderMetadata>>,
}

impl CohortExec {
    fn new(
        schema: SchemaRef,
        dataset_label: String,
        files: Vec<FileInfo>,
        file_metadata: Vec<Option<ArrowReaderMetadata>>,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
        predicate: Option<Expr>,
    ) -> Self {
        // Expand each file into per-row-group sub-partitions for finer parallelism.
        // Row groups whose column statistics contradict the predicate are skipped.
        let mut partitions: Vec<FileInfo> = Vec::new();
        let mut cached_metadata: Vec<Option<ArrowReaderMetadata>> = Vec::new();
        for (file, meta) in files.iter().zip(file_metadata.iter()) {
            if let Some(ref m) = meta {
                let n_rg = m.metadata().num_row_groups();
                for rg in 0..n_rg {
                    if let Some(ref pred) = predicate {
                        if can_skip_row_group(pred, m.metadata().row_group(rg)) {
                            continue;
                        }
                    }
                    let rg_rows = m.metadata().row_group(rg).num_rows();
                    partitions.push(FileInfo {
                        path: file.path.clone(),
                        selectors: file.selectors.clone(),
                        logic_operator: file.logic_operator.clone(),
                        num_rows: rg_rows,
                        row_groups: Some(vec![rg]),
                    });
                    cached_metadata.push(Some(m.clone()));
                }
            } else {
                partitions.push(file.clone());
                cached_metadata.push(None);
            }
        }

        let total_num_rows: usize = partitions.iter().map(|f| f.num_rows as usize).sum();
        let n = partitions.len().max(1);
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(n),
            datafusion::physical_plan::execution_plan::EmissionType::Incremental,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        );
        Self {
            schema,
            dataset_label,
            files: partitions,
            projection,
            limit,
            properties,
            total_num_rows,
            cached_metadata,
        }
    }
}

impl fmt::Debug for CohortExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CohortExec")
            .field("dataset_label", &self.dataset_label)
            .field("partitions", &self.files.len())
            .field("total_num_rows", &self.total_num_rows)
            .field("projection", &self.projection)
            .field("limit", &self.limit)
            .finish()
    }
}

impl DisplayAs for CohortExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "CohortExec: dataset={}, partitions={}",
            self.dataset_label,
            self.files.len()
        )
    }
}

impl ExecutionPlan for CohortExec {
    fn name(&self) -> &str {
        "CohortExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition >= self.files.len() {
            // No files survived pruning — return empty stream.
            let empty = RecordBatch::new_empty(self.schema.clone());
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                self.schema.clone(),
                stream::iter(vec![Ok(empty)]),
            )));
        }

        let file = &self.files[partition];
        let label = &self.dataset_label;

        // Read batches with projection pushed down to Parquet.
        let reader = CohortReader::open(&file.path)
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;
        let iter: Box<dyn Iterator<Item = arrow::error::Result<RecordBatch>>> =
            if let Some(ref meta) = self.cached_metadata[partition] {
                Box::new(
                    reader
                        .get_dataset_data_with_metadata(
                            label,
                            meta.clone(),
                            self.projection.as_deref(),
                            file.row_groups.clone(),
                        )
                        .map_err(|e| DataFusionError::Execution(e.to_string()))?,
                )
            } else {
                Box::new(
                    reader
                        .get_dataset_data_filtered(
                            label,
                            self.projection.as_deref(),
                            file.row_groups.clone(),
                        )
                        .map_err(|e| DataFusionError::Execution(e.to_string()))?,
                )
            };

        // Collect batches, stopping early when we have enough rows for limit.
        // Rename columns to lowercase so they match the registered schema.
        let target_schema = self.schema.clone();
        let mut batches: Vec<RecordBatch> = Vec::new();
        let mut rows_so_far: usize = 0;
        for batch_result in iter {
            let batch = batch_result.map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
            let batch = if batch.schema().fields() != target_schema.fields() {
                cast_batch_to_schema(&batch, &target_schema)
                    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?
            } else {
                batch
            };
            rows_so_far += batch.num_rows();
            batches.push(batch);
            if let Some(limit) = self.limit {
                if rows_so_far >= limit {
                    break;
                }
            }
        }

        let schema = self.schema.clone();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(batches.into_iter().map(Ok)),
        )))
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn statistics(&self) -> Result<Statistics> {
        Ok(Statistics {
            num_rows: Precision::Exact(self.total_num_rows),
            total_byte_size: Precision::Absent,
            column_statistics: Statistics::unknown_column(&self.schema),
        })
    }
}

// ── Helpers ────────────────────────────────────────────────────

/// Reconstruct an Arrow schema from catalog column metadata.
/// Field names are lowercased so DataFusion's default identifier normalisation
/// (unquoted → lowercase) matches without requiring users to quote columns.
fn schema_from_catalog_columns(cols: &[CatalogColumn]) -> Option<Schema> {
    let fields: Option<Vec<Field>> = cols
        .iter()
        .map(|c| {
            parse_data_type(&c.data_type)
                .map(|dt| Field::new(c.name.to_lowercase(), dt, c.nullable))
        })
        .collect();
    fields.map(Schema::new)
}

/// Parse a data type string (as produced by `format!("{}", DataType)`) back to an Arrow DataType.
fn parse_data_type(s: &str) -> Option<DataType> {
    match s {
        "Boolean" => Some(DataType::Boolean),
        "Int8" => Some(DataType::Int8),
        "Int16" => Some(DataType::Int16),
        "Int32" => Some(DataType::Int32),
        "Int64" => Some(DataType::Int64),
        "UInt8" => Some(DataType::UInt8),
        "UInt16" => Some(DataType::UInt16),
        "UInt32" => Some(DataType::UInt32),
        "UInt64" => Some(DataType::UInt64),
        "Float16" => Some(DataType::Float16),
        "Float32" => Some(DataType::Float32),
        "Float64" => Some(DataType::Float64),
        "Utf8" => Some(DataType::Utf8),
        "LargeUtf8" => Some(DataType::LargeUtf8),
        "Binary" => Some(DataType::Binary),
        "LargeBinary" => Some(DataType::LargeBinary),
        "Date32" => Some(DataType::Date32),
        "Date64" => Some(DataType::Date64),
        "Null" => Some(DataType::Null),
        _ => None,
    }
}

/// Widen multiple schemas into one by picking the widest numeric type per column.
fn widen_schemas(schemas: &[Schema]) -> Schema {
    let first = &schemas[0];
    let fields: Vec<Field> = first
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let mut widest = f.data_type().clone();
            for s in &schemas[1..] {
                if let Some(other) = s.fields().get(i) {
                    widest = wider_type(&widest, other.data_type());
                }
            }
            Field::new(f.name(), widest, f.is_nullable())
        })
        .collect();
    Schema::new(fields)
}

/// Pick the wider of two numeric types (signed ints only for now).
fn wider_type(a: &DataType, b: &DataType) -> DataType {
    if a == b {
        return a.clone();
    }
    fn int_rank(dt: &DataType) -> Option<u8> {
        match dt {
            DataType::Int8 => Some(1),
            DataType::Int16 => Some(2),
            DataType::Int32 => Some(3),
            DataType::Int64 => Some(4),
            DataType::UInt8 => Some(1),
            DataType::UInt16 => Some(2),
            DataType::UInt32 => Some(3),
            DataType::UInt64 => Some(4),
            _ => None,
        }
    }
    match (int_rank(a), int_rank(b)) {
        (Some(ra), Some(rb)) => {
            if ra >= rb { a.clone() } else { b.clone() }
        }
        _ => a.clone(), // non-numeric mismatch: keep the first
    }
}

/// Cast a RecordBatch's columns to match a target schema.
fn cast_batch_to_schema(
    batch: &RecordBatch,
    target: &SchemaRef,
) -> arrow::error::Result<RecordBatch> {
    let columns: Vec<Arc<dyn arrow::array::Array>> = batch
        .columns()
        .iter()
        .zip(target.fields().iter())
        .map(|(col, field)| {
            if col.data_type() == field.data_type() {
                Ok(col.clone())
            } else {
                cast(col, field.data_type())
            }
        })
        .collect::<arrow::error::Result<Vec<_>>>()?;
    RecordBatch::try_new(target.clone(), columns)
}

// ── Row group statistics pruning ──────────────────────────────

/// Returns `true` if the row group can definitely be skipped — no row can match `predicate`
/// based on the column min/max statistics stored in the Parquet metadata.
pub fn can_skip_row_group(predicate: &Expr, rg_meta: &RowGroupMetaData) -> bool {
    match predicate {
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => match op {
            Operator::And => {
                can_skip_row_group(left, rg_meta) || can_skip_row_group(right, rg_meta)
            }
            Operator::Or => {
                can_skip_row_group(left, rg_meta) && can_skip_row_group(right, rg_meta)
            }
            Operator::Eq | Operator::NotEq | Operator::Lt | Operator::LtEq
            | Operator::Gt | Operator::GtEq => check_rg_comparison(left, *op, right, rg_meta),
            _ => false,
        },
        Expr::InList(il) => check_rg_in_list(&il.expr, &il.list, il.negated, rg_meta),
        _ => false,
    }
}

fn check_rg_comparison(
    left: &Expr,
    op: Operator,
    right: &Expr,
    rg_meta: &RowGroupMetaData,
) -> bool {
    let (col_name, scalar, op) = match (left, right) {
        (Expr::Column(c), Expr::Literal(v, _)) => (c.name(), v, op),
        (Expr::Literal(v, _), Expr::Column(c)) => (c.name(), v, flip_op(op)),
        _ => return false,
    };

    let col_idx = match find_rg_column_index(rg_meta, col_name) {
        Some(i) => i,
        None => return false,
    };
    let stats = match rg_meta.column(col_idx).statistics() {
        Some(s) if s.min_bytes_opt().is_some() && s.max_bytes_opt().is_some() => s,
        _ => return false,
    };

    // Numeric comparison
    if let (Some((min, max)), Some(qv)) = (stats_min_max_f64(stats), scalar_as_f64(scalar)) {
        return match op {
            Operator::Eq => qv < min || qv > max,
            Operator::NotEq => (min - max).abs() < f64::EPSILON && (min - qv).abs() < f64::EPSILON,
            Operator::Lt => min >= qv,
            Operator::LtEq => min > qv,
            Operator::Gt => max <= qv,
            Operator::GtEq => max < qv,
            _ => false,
        };
    }

    // String comparison
    if let (Some((ref min_s, ref max_s)), Some(ref qv_s)) =
        (stats_min_max_string(stats), scalar_as_string(scalar))
    {
        return match op {
            Operator::Eq => qv_s.as_str() < min_s.as_str() || qv_s.as_str() > max_s.as_str(),
            Operator::NotEq => min_s == max_s && min_s == qv_s,
            Operator::Lt => min_s.as_str() >= qv_s.as_str(),
            Operator::LtEq => min_s.as_str() > qv_s.as_str(),
            Operator::Gt => max_s.as_str() <= qv_s.as_str(),
            Operator::GtEq => max_s.as_str() < qv_s.as_str(),
            _ => false,
        };
    }

    false
}

fn check_rg_in_list(
    expr: &Expr,
    list: &[Expr],
    negated: bool,
    rg_meta: &RowGroupMetaData,
) -> bool {
    if negated || list.is_empty() {
        return false;
    }
    let col_name = match expr {
        Expr::Column(c) => c.name(),
        _ => return false,
    };
    let col_idx = match find_rg_column_index(rg_meta, col_name) {
        Some(i) => i,
        None => return false,
    };
    let stats = match rg_meta.column(col_idx).statistics() {
        Some(s) if s.min_bytes_opt().is_some() && s.max_bytes_opt().is_some() => s,
        _ => return false,
    };

    // Numeric: skip if every value is outside [min, max]
    if let Some((min, max)) = stats_min_max_f64(stats) {
        let all_outside = list.iter().all(|e| match e {
            Expr::Literal(v, _) => scalar_as_f64(v).map(|f| f < min || f > max).unwrap_or(false),
            _ => false,
        });
        if all_outside {
            return true;
        }
    }

    // String: skip if every value is outside [min, max]
    if let Some((ref min_s, ref max_s)) = stats_min_max_string(stats) {
        let all_outside = list.iter().all(|e| match e {
            Expr::Literal(v, _) => scalar_as_string(v)
                .map(|s| s.as_str() < min_s.as_str() || s.as_str() > max_s.as_str())
                .unwrap_or(false),
            _ => false,
        });
        if all_outside {
            return true;
        }
    }

    false
}

fn find_rg_column_index(rg_meta: &RowGroupMetaData, col_name: &str) -> Option<usize> {
    let col_lower = col_name.to_lowercase();
    (0..rg_meta.num_columns()).find(|&i| {
        rg_meta.column(i).column_descr().name().to_lowercase() == col_lower
    })
}

fn flip_op(op: Operator) -> Operator {
    match op {
        Operator::Lt => Operator::Gt,
        Operator::LtEq => Operator::GtEq,
        Operator::Gt => Operator::Lt,
        Operator::GtEq => Operator::LtEq,
        other => other,
    }
}

fn stats_min_max_f64(stats: &ParquetStats) -> Option<(f64, f64)> {
    match stats {
        ParquetStats::Int32(s) => Some((*s.min_opt()? as f64, *s.max_opt()? as f64)),
        ParquetStats::Int64(s) => Some((*s.min_opt()? as f64, *s.max_opt()? as f64)),
        ParquetStats::Float(s) => Some((*s.min_opt()? as f64, *s.max_opt()? as f64)),
        ParquetStats::Double(s) => Some((*s.min_opt()? as f64, *s.max_opt()? as f64)),
        _ => None,
    }
}

fn stats_min_max_string(stats: &ParquetStats) -> Option<(String, String)> {
    match stats {
        ParquetStats::ByteArray(s) => {
            let mn = std::str::from_utf8(s.min_bytes_opt()?).ok()?.to_string();
            let mx = std::str::from_utf8(s.max_bytes_opt()?).ok()?.to_string();
            Some((mn, mx))
        }
        _ => None,
    }
}

fn scalar_as_f64(val: &ScalarValue) -> Option<f64> {
    match val {
        ScalarValue::Int8(Some(v)) => Some(*v as f64),
        ScalarValue::Int16(Some(v)) => Some(*v as f64),
        ScalarValue::Int32(Some(v)) => Some(*v as f64),
        ScalarValue::Int64(Some(v)) => Some(*v as f64),
        ScalarValue::UInt8(Some(v)) => Some(*v as f64),
        ScalarValue::UInt16(Some(v)) => Some(*v as f64),
        ScalarValue::UInt32(Some(v)) => Some(*v as f64),
        ScalarValue::UInt64(Some(v)) => Some(*v as f64),
        ScalarValue::Float32(Some(v)) => Some(*v as f64),
        ScalarValue::Float64(Some(v)) => Some(*v as f64),
        _ => None,
    }
}

fn scalar_as_string(val: &ScalarValue) -> Option<String> {
    match val {
        ScalarValue::Utf8(Some(v))
        | ScalarValue::LargeUtf8(Some(v))
        | ScalarValue::Utf8View(Some(v)) => Some(v.clone()),
        _ => None,
    }
}

/// Read the Arrow schema for a dataset's data block inside a .cohort file.
/// Field names are lowercased to match DataFusion's identifier normalisation.
fn read_dataset_schema(
    file_path: &str,
    label: &str,
) -> std::result::Result<SchemaRef, Box<dyn std::error::Error>> {
    let reader = CohortReader::open(file_path)?;
    let tag = reader
        .directory
        .tags
        .iter()
        .find(|t| t.label == label)
        .ok_or_else(|| format!("No tag with label '{label}'"))?;

    let schema = match tag.codec.as_str() {
        "arrow_ipc" | "feather" => {
            let read_seek =
                OffsetReadSeek::new(file_path, tag.offset, tag.length)?;
            let ipc_reader = IpcFileReader::try_new(read_seek, None)?;
            ipc_reader.schema()
        }
        _ => {
            let offset_reader = OffsetReader::new(file_path, tag.offset, tag.length);
            let builder = ParquetRecordBatchReaderBuilder::try_new(offset_reader)?;
            builder.schema().clone()
        }
    };

    let lc_fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|f| Field::new(f.name().to_lowercase(), f.data_type().clone(), f.is_nullable()))
        .collect();
    Ok(Arc::new(Schema::new(lc_fields)))
}

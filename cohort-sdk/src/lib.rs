use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use arrow::array::RecordBatch;
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use cohort_lib::catalog::CohortCatalog;
use cohort_lib::models::{CohortDefinition, RevisionEntry, Selector, TagDirectory};
use cohort_lib::query_engine::QueryEngine;
use cohort_lib::reader::CohortReader;
use cohort_lib::writer::CohortWriter;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

/// Open a .cohort file and return the tag directory as a JSON string.
#[pyfunction]
fn open_cohort(path: &str) -> PyResult<String> {
    let reader = CohortReader::open(path)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let json = serde_json::to_string(&reader.directory)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(json)
}

/// Read the raw Parquet bytes for a dataset block identified by label.
#[pyfunction]
fn read_dataset_bytes<'py>(py: Python<'py>, path: &str, label: &str) -> PyResult<Bound<'py, PyBytes>> {
    let reader = CohortReader::open(path)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let tag = reader
        .directory
        .tags
        .iter()
        .find(|t| t.label == label)
        .ok_or_else(|| PyValueError::new_err(format!("No tag with label '{label}'")))?;

    let mut file = File::open(path)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    file.seek(SeekFrom::Start(tag.offset))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut buf = vec![0u8; tag.length as usize];
    file.read_exact(&mut buf)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyBytes::new(py, &buf))
}

/// Write a .cohort file from pyarrow RecordBatches (zero-copy Arrow FFI).
///
/// `directory_json`: JSON string of the TagDirectory (cohort_definition, patient_index).
/// `datasets`: list of (label, pyarrow.RecordBatch) tuples.
/// `codec`: encoding format — "parquet" (default), "arrow_ipc", "feather", or "vortex".
/// `index_column`: optional column name to build a Roaring bitmap patient index on.
#[pyfunction]
#[pyo3(signature = (path, directory_json, datasets, codec=None, index_column=None))]
fn write_cohort_arrow(
    path: &str,
    directory_json: &str,
    datasets: Vec<(String, Bound<'_, PyAny>)>,
    codec: Option<&str>,
    index_column: Option<&str>,
) -> PyResult<()> {
    let directory: TagDirectory = serde_json::from_str(directory_json)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let batches: Vec<(String, RecordBatch)> = datasets
        .into_iter()
        .map(|(label, py_batch)| {
            let batch = RecordBatch::from_pyarrow_bound(&py_batch)
                .map_err(|e| PyValueError::new_err(format!("Arrow FFI error for '{label}': {e}")))?;
            Ok((label, batch))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let batch_refs: Vec<(&str, &RecordBatch)> = batches
        .iter()
        .map(|(label, batch)| (label.as_str(), batch))
        .collect();

    let codec = codec.unwrap_or("parquet");
    CohortWriter::write_with_codec_and_index(path, &directory, &batch_refs, codec, index_column)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(())
}

/// Read a dataset as a PyArrow RecordBatch via the Rust codec-aware reader (zero-copy FFI).
#[pyfunction]
fn read_dataset_arrow(py: Python<'_>, path: &str, label: &str) -> PyResult<Py<PyAny>> {
    let reader = CohortReader::open(path)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let batches: Vec<RecordBatch> = reader
        .get_dataset_data(label)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    if batches.is_empty() {
        return Err(PyValueError::new_err(format!("No data in dataset '{label}'")));
    }

    let schema = batches[0].schema();
    let merged = arrow::compute::concat_batches(&schema, &batches)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(merged.to_pyarrow(py)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .unbind())
}

/// Inspect a .cohort file and return metadata as a JSON string.
#[pyfunction]
fn inspect_cohort(path: &str) -> PyResult<String> {
    let inspection = cohort_lib::inspect::inspect(path)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let json = serde_json::to_string(&inspection)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(json)
}

/// Union: concatenate all rows from multiple .cohort files into one.
#[pyfunction]
#[pyo3(signature = (input_paths, output_path, name=None))]
fn union_cohorts(input_paths: Vec<String>, output_path: &str, name: Option<&str>) -> PyResult<()> {
    let paths: Vec<&str> = input_paths.iter().map(|s| s.as_str()).collect();
    cohort_lib::merge::union(&paths, output_path, name)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(())
}

/// Intersect: keep only rows where `on` key appears in ALL input files.
#[pyfunction]
#[pyo3(signature = (input_paths, output_path, on, name=None))]
fn intersect_cohorts(input_paths: Vec<String>, output_path: &str, on: &str, name: Option<&str>) -> PyResult<()> {
    let paths: Vec<&str> = input_paths.iter().map(|s| s.as_str()).collect();
    cohort_lib::merge::intersect(&paths, output_path, on, name)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(())
}

/// Apply a revision to a cohort definition (both as JSON strings).
/// Returns the updated cohort definition as JSON.
#[pyfunction]
fn apply_revision(definition_json: &str, revision_json: &str) -> PyResult<String> {
    let mut definition: CohortDefinition = serde_json::from_str(definition_json)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let entry: RevisionEntry = serde_json::from_str(revision_json)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    definition
        .apply_revision(entry)
        .map_err(|e| PyValueError::new_err(e))?;

    let json = serde_json::to_string(&definition)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(json)
}

// ── Annotate ──────────────────────────────────────────────────

/// Modify name, description, and/or selectors of a .cohort file.
#[pyfunction]
#[pyo3(signature = (path, output=None, name=None, description=None, add_selectors_json=None, remove_selector_ids=None, message="Annotate", author="cohort-sdk"))]
fn annotate_cohort(
    path: &str,
    output: Option<&str>,
    name: Option<&str>,
    description: Option<&str>,
    add_selectors_json: Option<&str>,
    remove_selector_ids: Option<Vec<String>>,
    message: &str,
    author: &str,
) -> PyResult<()> {
    let add_selectors: Vec<Selector> = match add_selectors_json {
        Some(json) => serde_json::from_str(json)
            .map_err(|e| PyValueError::new_err(format!("Invalid selectors JSON: {e}")))?,
        None => vec![],
    };
    let remove_ids = remove_selector_ids.unwrap_or_default();

    cohort_lib::annotate::annotate(
        path, output, name, description, &add_selectors, &remove_ids, message, author,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

// ── Diff ──────────────────────────────────────────────────────

/// Compare two .cohort files and return a JSON diff.
#[pyfunction]
fn diff_cohorts(left_path: &str, right_path: &str) -> PyResult<String> {
    let d = cohort_lib::diff::diff(left_path, right_path)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    serde_json::to_string(&d).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Compare two revisions within the same .cohort file and return a JSON diff.
#[pyfunction]
fn diff_revisions_cohort(path: &str, rev_a: u32, rev_b: u32) -> PyResult<String> {
    let d = cohort_lib::diff::diff_revisions(path, rev_a, rev_b)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    serde_json::to_string(&d).map_err(|e| PyValueError::new_err(e.to_string()))
}

// ── Append ────────────────────────────────────────────────────

/// Append rows to a dataset from a PyArrow RecordBatch.
/// If dataset_label is None, auto-detects by schema matching.
#[pyfunction]
#[pyo3(signature = (path, data, dataset_label=None, output=None, message="Append data", author="cohort-sdk"))]
fn append_cohort(
    path: &str,
    data: Bound<'_, PyAny>,
    dataset_label: Option<&str>,
    output: Option<&str>,
    message: &str,
    author: &str,
) -> PyResult<()> {
    let batch = RecordBatch::from_pyarrow_bound(&data)
        .map_err(|e| PyValueError::new_err(format!("Arrow FFI error: {e}")))?;
    cohort_lib::mutate::append(path, output, dataset_label, &batch, message, author)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

// ── Delete ────────────────────────────────────────────────────

/// Delete rows by key values.
#[pyfunction]
#[pyo3(signature = (path, key_column, values, output=None, message="Delete rows", author="cohort-sdk"))]
fn delete_rows_cohort(
    path: &str,
    key_column: &str,
    values: Vec<String>,
    output: Option<&str>,
    message: &str,
    author: &str,
) -> PyResult<()> {
    cohort_lib::mutate::delete_rows(path, output, key_column, &values, message, author)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Delete an entire dataset.
#[pyfunction]
#[pyo3(signature = (path, label, output=None, message="Delete dataset", author="cohort-sdk"))]
fn delete_dataset_cohort(
    path: &str,
    label: &str,
    output: Option<&str>,
    message: &str,
    author: &str,
) -> PyResult<()> {
    cohort_lib::mutate::delete_dataset(path, output, label, message, author)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Delete selectors by ID.
#[pyfunction]
#[pyo3(signature = (path, selector_ids, output=None, message="Delete selectors", author="cohort-sdk"))]
fn delete_selectors_cohort(
    path: &str,
    selector_ids: Vec<String>,
    output: Option<&str>,
    message: &str,
    author: &str,
) -> PyResult<()> {
    cohort_lib::mutate::delete_selectors(path, output, &selector_ids, message, author)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

// ── Upsert ────────────────────────────────────────────────────

/// Update-or-insert rows by key column from a PyArrow RecordBatch.
#[pyfunction]
#[pyo3(signature = (path, dataset_label, key_column, data, output=None, message="Upsert data", author="cohort-sdk"))]
fn upsert_cohort(
    path: &str,
    dataset_label: &str,
    key_column: &str,
    data: Bound<'_, PyAny>,
    output: Option<&str>,
    message: &str,
    author: &str,
) -> PyResult<()> {
    let batch = RecordBatch::from_pyarrow_bound(&data)
        .map_err(|e| PyValueError::new_err(format!("Arrow FFI error: {e}")))?;
    cohort_lib::mutate::upsert(path, output, dataset_label, key_column, &batch, message, author)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Apply incremental (delete + add) steps to a base Arrow batch in Rust.
#[pyfunction]
fn increment_cohort(
    py: Python<'_>,
    base: Bound<'_, PyAny>,
    key_column: &str,
    delete_key_lists: Vec<Vec<String>>,
    add_batches: Vec<Option<Bound<'_, PyAny>>>,
) -> PyResult<Py<PyAny>> {
    let base_batch = RecordBatch::from_pyarrow_bound(&base)
        .map_err(|e| PyValueError::new_err(format!("Arrow FFI error for base: {e}")))?;

    let increments: Vec<(std::collections::HashSet<String>, Option<RecordBatch>)> = delete_key_lists
        .into_iter()
        .zip(add_batches.into_iter())
        .map(|(del_keys, add_py)| {
            let key_set = del_keys.into_iter().collect();
            let adds = add_py
                .map(|b| {
                    RecordBatch::from_pyarrow_bound(&b)
                        .map_err(|e| PyValueError::new_err(format!("Arrow FFI error for adds: {e}")))
                })
                .transpose()?;
            Ok((key_set, adds))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let batches = cohort_lib::increment(base_batch, key_column, &increments)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let py_batches: Vec<Py<PyAny>> = batches
        .iter()
        .map(|b| Ok(b.to_pyarrow(py)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .unbind()))
        .collect::<PyResult<Vec<_>>>()?;

    Ok(PyList::new(py, &py_batches)?.into())
}

// ── Partition ──────────────────────────────────────────────────

/// Split a .cohort file into N smaller partition files for parallel queries.
/// Returns a list of output file paths.
#[pyfunction]
#[pyo3(signature = (input_path, output_dir, num_partitions=None, target_rows=None))]
fn partition_cohort(
    input_path: &str,
    output_dir: &str,
    num_partitions: Option<usize>,
    target_rows: Option<usize>,
) -> PyResult<Vec<String>> {
    cohort_lib::partition::partition(input_path, output_dir, num_partitions, target_rows)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

// ── CohortCatalog PyO3 class (SQL query engine) ───────────────

#[pyclass(name = "CohortCatalog")]
struct PyCohortCatalog {
    engine: QueryEngine,
    runtime: tokio::runtime::Runtime,
}

#[pymethods]
impl PyCohortCatalog {
    /// Build a catalog from .cohort files: CohortCatalog.init("./data/")
    #[staticmethod]
    fn init(path: &str) -> PyResult<()> {
        CohortCatalog::init(path)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Load an existing catalog: catalog = CohortCatalog("./data/")
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let catalog = CohortCatalog::load(path)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let engine = QueryEngine::new(catalog)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { engine, runtime })
    }

    /// Run a SQL query → PyArrow Table.
    fn query(&self, py: Python<'_>, sql: &str) -> PyResult<Py<PyAny>> {
        let (schema, batches) = self
            .runtime
            .block_on(self.engine.query(sql))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        // Convert Vec<RecordBatch> → PyArrow Table
        let pyarrow = py.import("pyarrow")?;
        let py_batches: Vec<Py<PyAny>> = batches
            .iter()
            .map(|b| Ok(b.to_pyarrow(py)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .unbind()))
            .collect::<PyResult<Vec<_>>>()?;

        if py_batches.is_empty() {
            // Return an empty table that preserves the query schema
            let empty_batch = RecordBatch::new_empty(schema);
            let py_batch = empty_batch.to_pyarrow(py)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let table = pyarrow
                .getattr("Table")?
                .call_method1("from_batches", (vec![py_batch],))?;
            return Ok(table.unbind());
        }

        let table = pyarrow
            .getattr("Table")?
            .call_method1("from_batches", (py_batches,))?;

        Ok(table.unbind())
    }

    /// List available SQL table names.
    fn tables(&self) -> Vec<String> {
        self.engine.list_tables()
    }
}

#[pymodule]
fn _cohort_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(open_cohort, m)?)?;
    m.add_function(wrap_pyfunction!(read_dataset_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(read_dataset_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(write_cohort_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(apply_revision, m)?)?;
    m.add_function(wrap_pyfunction!(inspect_cohort, m)?)?;
    m.add_function(wrap_pyfunction!(union_cohorts, m)?)?;
    m.add_function(wrap_pyfunction!(intersect_cohorts, m)?)?;
    m.add_function(wrap_pyfunction!(annotate_cohort, m)?)?;
    m.add_function(wrap_pyfunction!(diff_cohorts, m)?)?;
    m.add_function(wrap_pyfunction!(diff_revisions_cohort, m)?)?;
    m.add_function(wrap_pyfunction!(append_cohort, m)?)?;
    m.add_function(wrap_pyfunction!(delete_rows_cohort, m)?)?;
    m.add_function(wrap_pyfunction!(delete_dataset_cohort, m)?)?;
    m.add_function(wrap_pyfunction!(delete_selectors_cohort, m)?)?;
    m.add_function(wrap_pyfunction!(upsert_cohort, m)?)?;
    m.add_function(wrap_pyfunction!(increment_cohort, m)?)?;
    m.add_function(wrap_pyfunction!(partition_cohort, m)?)?;
    m.add_class::<PyCohortCatalog>()?;
    Ok(())
}

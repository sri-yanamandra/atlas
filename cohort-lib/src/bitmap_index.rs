use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use arrow::array::{Array, Int64Array, StringArray};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use roaring::RoaringTreemap;

/// Hash a string key to a u64 using DefaultHasher (deterministic within a build).
pub fn hash_key(value: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// Build a RoaringTreemap from the key column across multiple RecordBatches.
///
/// - Integer columns (Int8..Int64): cast to i64, use the value directly as u64.
/// - String columns: hash each value to u64.
pub fn build_bitmap(
    batches: &[RecordBatch],
    key_col: &str,
) -> Result<RoaringTreemap, Box<dyn std::error::Error>> {
    let mut bitmap = RoaringTreemap::new();

    for batch in batches {
        let col_idx = batch.schema().index_of(key_col)?;
        let array = batch.column(col_idx);

        match array.data_type() {
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
                let int_array = cast(array, &DataType::Int64)?;
                let int_array = int_array
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or("Failed to cast to Int64Array")?;
                for i in 0..int_array.len() {
                    if !int_array.is_null(i) {
                        bitmap.insert(int_array.value(i) as u64);
                    }
                }
            }
            _ => {
                let str_array = cast(array, &DataType::Utf8)?;
                let str_array = str_array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        format!("Cannot cast column '{key_col}' to string for bitmap index")
                    })?;
                for i in 0..str_array.len() {
                    if !str_array.is_null(i) {
                        bitmap.insert(hash_key(str_array.value(i)));
                    }
                }
            }
        }
    }

    Ok(bitmap)
}

/// Serialize a RoaringTreemap to bytes.
pub fn serialize_bitmap(bitmap: &RoaringTreemap) -> Vec<u8> {
    let mut buf = Vec::with_capacity(bitmap.serialized_size());
    bitmap
        .serialize_into(&mut buf)
        .expect("bitmap serialization should not fail");
    buf
}

/// Deserialize a RoaringTreemap from bytes.
pub fn deserialize_bitmap(bytes: &[u8]) -> Result<RoaringTreemap, Box<dyn std::error::Error>> {
    Ok(RoaringTreemap::deserialize_from(bytes)?)
}

/// Returns `true` if every queried value is absent from the bitmap,
/// meaning the file can be skipped for this query.
pub fn bitmap_can_skip_file(bitmap: &RoaringTreemap, queried_values: &[u64]) -> bool {
    queried_values.iter().all(|v| !bitmap.contains(*v))
}

/// Compute the intersection of multiple bitmaps.
pub fn intersect_bitmaps(bitmaps: &[RoaringTreemap]) -> RoaringTreemap {
    if bitmaps.is_empty() {
        return RoaringTreemap::new();
    }
    let mut result = bitmaps[0].clone();
    for bm in &bitmaps[1..] {
        result &= bm;
    }
    result
}

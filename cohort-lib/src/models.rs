use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize};

// ── Tag Directory ──────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TagDirectory {
    pub tags: Vec<Tag>,
    pub patient_index: ByteRange,
    pub cohort_definition: CohortDefinition,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<IcebergBackend>,
}

// ── Iceberg Backend ───────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IcebergSnapshot {
    pub snapshot_id: i64,
    pub timestamp_ms: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_records: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_file_size: Option<i64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IcebergTableRef {
    pub dataset_label: String,
    pub table_name: String,
    pub table_location: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snapshot_id: Option<i64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub snapshots: Vec<IcebergSnapshot>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IcebergBackend {
    pub catalog_name: String,
    pub region: String,
    pub database: String,
    pub table_name: String,
    pub table_location: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snapshot_id: Option<i64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub snapshots: Vec<IcebergSnapshot>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tables: Vec<IcebergTableRef>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Tag {
    pub tag_id: u32,
    pub label: String,
    pub codec: String,
    pub offset: u64,
    pub length: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema_type: Option<String>,
    /// Row count stamped at write time so inspect() never touches Parquet.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_rows: Option<i64>,
    /// Column count stamped at write time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_columns: Option<usize>,
    /// Column schema stamped at write time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub columns: Option<Vec<TagColumnInfo>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TagColumnInfo {
    pub name: String,
    pub data_type: String,
    pub nullable: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ByteRange {
    pub offset: u64,
    pub length: u64,
}

// ── Cohort Definition ──────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CohortDefinition {
    pub revision: u32,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,
    pub created_by: String,
    pub logic_operator: String,
    #[serde(default)]
    pub selectors: Vec<Selector>,
    pub lineage: Lineage,
    #[serde(default)]
    pub revision_history: Vec<RevisionEntry>,
    #[serde(default)]
    pub primary_keys: Vec<String>,
}

impl CohortDefinition {
    /// Create a minimal cohort definition with sensible defaults.
    pub fn new(name: &str) -> Self {
        Self {
            revision: 1,
            name: name.to_string(),
            description: None,
            created_at: Utc::now(),
            created_by: String::new(),
            logic_operator: "AND".to_string(),
            selectors: vec![],
            lineage: Lineage::empty(),
            revision_history: vec![],
            primary_keys: vec![],
        }
    }

    /// Apply a new revision to the cohort definition.
    /// Updates selectors in place and appends to revision_history.
    pub fn apply_revision(&mut self, entry: RevisionEntry) -> Result<(), String> {
        if entry.revision != self.revision + 1 {
            return Err(format!(
                "Expected revision {}, got {}",
                self.revision + 1,
                entry.revision
            ));
        }

        for delta in &entry.deltas {
            match delta.action {
                DeltaAction::Add => {
                    let selector = delta
                        .selector
                        .clone()
                        .ok_or("Add delta missing selector")?;
                    self.selectors.push(selector);
                }
                DeltaAction::Remove => {
                    self.selectors
                        .retain(|s| s.selector_id != delta.selector_id);
                }
                DeltaAction::Modify => {
                    let sel = self
                        .selectors
                        .iter_mut()
                        .find(|s| s.selector_id == delta.selector_id)
                        .ok_or(format!(
                            "Selector '{}' not found",
                            delta.selector_id
                        ))?;
                    let field = delta
                        .field
                        .as_ref()
                        .ok_or("Modify delta missing field")?;
                    let after = delta
                        .after
                        .clone()
                        .ok_or("Modify delta missing 'after' value")?;

                    match field.as_str() {
                        "values" => sel.values = after,
                        "operator" => {
                            sel.operator = serde_json::from_value(after)
                                .map_err(|e| e.to_string())?;
                        }
                        "is_exclusion" => {
                            sel.is_exclusion = serde_json::from_value(after)
                                .map_err(|e| e.to_string())?;
                        }
                        "label" => {
                            sel.label = serde_json::from_value(after)
                                .map_err(|e| e.to_string())?;
                        }
                        other => return Err(format!("Unknown selector field: {other}")),
                    }
                }
            }
        }

        self.revision = entry.revision;
        self.revision_history.push(entry);
        Ok(())
    }
}

// ── Selector ───────────────────────────────────────────────────

/// A single filter criterion in a cohort definition.
///
/// Supported operators:
///   - `in`        — field value is in the list (values: ["E11.0", "E11.1"])
///   - `not_in`    — field value is NOT in the list
///   - `eq`        — equals a single value (values: ["CA"])
///   - `neq`       — not equal to a single value
///   - `gt`        — greater than (values: [65])
///   - `gte`       — greater than or equal
///   - `lt`        — less than
///   - `lte`       — less than or equal
///   - `between`   — inclusive range (values: [min, max])
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Selector {
    pub selector_id: String,
    pub field: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_system: Option<String>,
    pub operator: String,
    pub values: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default)]
    pub is_exclusion: bool,
}

// ── Lineage ────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Lineage {
    #[serde(deserialize_with = "string_or_vec")]
    pub source_system: Vec<String>,
    pub source_tables: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_hash: Option<String>,
    pub extraction_date: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_job_id: Option<String>,
}

/// Deserialize a field that may be either a single string or a list of strings.
/// Provides backward compatibility: old files have `"source_system": "Snowflake"`,
/// new files have `"source_system": ["Snowflake", "Redshift"]`.
fn string_or_vec<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrVec {
        String(String),
        Vec(Vec<String>),
    }
    match StringOrVec::deserialize(deserializer)? {
        StringOrVec::String(s) => Ok(vec![s]),
        StringOrVec::Vec(v) => Ok(v),
    }
}

impl Lineage {
    /// Create an empty placeholder lineage.
    pub fn empty() -> Self {
        Self {
            source_system: vec![],
            source_tables: vec![],
            query_hash: None,
            extraction_date: String::new(),
            generation_job_id: None,
        }
    }
}

// ── Revision History ───────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RevisionEntry {
    pub revision: u32,
    pub timestamp: DateTime<Utc>,
    pub author: String,
    pub message: String,
    pub deltas: Vec<Delta>,
    /// Total row count across all datasets at this revision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Delta {
    pub action: DeltaAction,
    pub selector_id: String,
    /// Present for "modify" deltas.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
    /// Previous value (modify only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before: Option<serde_json::Value>,
    /// New value (modify only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after: Option<serde_json::Value>,
    /// Full selector (add only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selector: Option<Selector>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum DeltaAction {
    Add,
    Remove,
    Modify,
}

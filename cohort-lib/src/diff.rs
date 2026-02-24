use std::collections::{BTreeMap, HashSet};

use serde::Serialize;

use crate::bitmap_index::deserialize_bitmap;
use crate::models::*;
use crate::reader::CohortReader;

// ── Diff types ────────────────────────────────────────────────

#[derive(Serialize, Debug, Clone)]
pub struct PatientOverlapStats {
    pub left_patient_count: u64,
    pub right_patient_count: u64,
    pub shared_patient_count: u64,
    pub left_only_count: u64,
    pub right_only_count: u64,
    pub jaccard_similarity: f64,
    pub overlap_coefficient: f64,
}

#[derive(Serialize, Debug, Clone)]
pub struct CohortDiff {
    pub left_name: String,
    pub right_name: String,
    pub left_revision: u32,
    pub right_revision: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name_changed: Option<StringChange>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description_changed: Option<OptionStringChange>,
    pub selectors: SelectorDiff,
    pub datasets: Vec<DatasetDiff>,
    pub lineage_changed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub patient_overlap: Option<PatientOverlapStats>,
}

#[derive(Serialize, Debug, Clone)]
pub struct StringChange {
    pub left: String,
    pub right: String,
}

#[derive(Serialize, Debug, Clone)]
pub struct OptionStringChange {
    pub left: Option<String>,
    pub right: Option<String>,
}

#[derive(Serialize, Debug, Clone)]
pub struct SelectorDiff {
    pub added: Vec<Selector>,
    pub removed: Vec<Selector>,
    pub modified: Vec<SelectorChange>,
}

#[derive(Serialize, Debug, Clone)]
pub struct SelectorChange {
    pub selector_id: String,
    pub left: Selector,
    pub right: Selector,
}

#[derive(Serialize, Debug, Clone)]
pub struct DatasetDiff {
    pub label: String,
    pub status: DatasetStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub left_rows: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub right_rows: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub left_cols: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub right_cols: Option<usize>,
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum DatasetStatus {
    Same,
    Changed,
    Added,
    Removed,
}

// ── Diff implementation ───────────────────────────────────────

/// Compare two .cohort files and produce a diff.
pub fn diff(
    left_path: &str,
    right_path: &str,
) -> Result<CohortDiff, Box<dyn std::error::Error>> {
    let left_reader = CohortReader::open(left_path)?;
    let right_reader = CohortReader::open(right_path)?;

    diff_definitions(
        &left_reader.directory,
        &right_reader.directory,
        &left_reader,
        &right_reader,
    )
}

/// Compare two revisions within the same .cohort file.
/// Reconstructs state at each revision by replaying history.
pub fn diff_revisions(
    path: &str,
    rev_a: u32,
    rev_b: u32,
) -> Result<CohortDiff, Box<dyn std::error::Error>> {
    let reader = CohortReader::open(path)?;
    let def = reader.cohort_definition();

    if rev_a == 0 || rev_b == 0 {
        return Err("Revision numbers must be >= 1".into());
    }
    if rev_a > def.revision || rev_b > def.revision {
        return Err(format!(
            "Revision out of range: max is {} (got {} and {})",
            def.revision, rev_a, rev_b
        )
        .into());
    }

    // Reconstruct definitions at each target revision by replaying history
    let def_a = reconstruct_at_revision(def, rev_a)?;
    let def_b = reconstruct_at_revision(def, rev_b)?;

    let dir_a = TagDirectory {
        tags: reader.directory.tags.clone(),
        patient_index: reader.directory.patient_index.clone(),
        cohort_definition: def_a,
        backend: reader.directory.backend.clone(),
    };
    let dir_b = TagDirectory {
        tags: reader.directory.tags.clone(),
        patient_index: reader.directory.patient_index.clone(),
        cohort_definition: def_b,
        backend: reader.directory.backend.clone(),
    };

    // For revision diff, datasets are the same (same file) so we just compare definitions
    let dataset_diffs = reader
        .directory
        .tags
        .iter()
        .map(|t| DatasetDiff {
            label: t.label.clone(),
            status: DatasetStatus::Same,
            left_rows: t.num_rows.map(|n| n as usize),
            right_rows: t.num_rows.map(|n| n as usize),
            left_cols: t.num_columns,
            right_cols: t.num_columns,
        })
        .collect();

    let name_changed = if dir_a.cohort_definition.name != dir_b.cohort_definition.name {
        Some(StringChange {
            left: dir_a.cohort_definition.name.clone(),
            right: dir_b.cohort_definition.name.clone(),
        })
    } else {
        None
    };

    let description_changed = if dir_a.cohort_definition.description != dir_b.cohort_definition.description {
        Some(OptionStringChange {
            left: dir_a.cohort_definition.description.clone(),
            right: dir_b.cohort_definition.description.clone(),
        })
    } else {
        None
    };

    let selectors = diff_selectors(
        &dir_a.cohort_definition.selectors,
        &dir_b.cohort_definition.selectors,
    );

    let lineage_changed = serde_json::to_string(&dir_a.cohort_definition.lineage)?
        != serde_json::to_string(&dir_b.cohort_definition.lineage)?;

    Ok(CohortDiff {
        left_name: format!("{}@r{}", dir_a.cohort_definition.name, rev_a),
        right_name: format!("{}@r{}", dir_b.cohort_definition.name, rev_b),
        left_revision: rev_a,
        right_revision: rev_b,
        name_changed,
        description_changed,
        selectors,
        datasets: dataset_diffs,
        lineage_changed,
        patient_overlap: None,
    })
}

/// Reconstruct the CohortDefinition as it was at a given revision.
fn reconstruct_at_revision(
    current: &CohortDefinition,
    target_rev: u32,
) -> Result<CohortDefinition, Box<dyn std::error::Error>> {
    if target_rev == current.revision {
        return Ok(current.clone());
    }

    let history = &current.revision_history;
    let min_tracked = history.iter().map(|e| e.revision).min().unwrap_or(current.revision);

    // Compute the base state BEFORE any tracked history by reverse-applying
    // all deltas from the current state. This recovers the original selectors
    // that existed at (min_tracked - 1), even when the first revision entry
    // has no deltas (e.g. "Initial extraction").
    let mut base_selectors = current.selectors.clone();
    for entry in history.iter().rev() {
        for delta in entry.deltas.iter().rev() {
            reverse_apply_delta(&mut base_selectors, delta);
        }
    }

    if target_rev < min_tracked {
        // Target is before any tracked history — return the base state
        let mut def = current.clone();
        def.revision = target_rev;
        def.selectors = base_selectors;
        def.revision_history = vec![];
        // Reverse-apply only deltas after target_rev (in case target is between base and min)
        for entry in history.iter().rev() {
            if entry.revision > target_rev {
                for delta in entry.deltas.iter().rev() {
                    reverse_apply_delta(&mut def.selectors, delta);
                }
            }
        }
        return Ok(def);
    }

    // Build a definition at the base revision, then forward-apply up to target
    let base_rev = min_tracked.saturating_sub(1);
    let mut def = CohortDefinition {
        revision: base_rev,
        name: current.name.clone(),
        description: current.description.clone(),
        created_at: current.created_at,
        created_by: current.created_by.clone(),
        logic_operator: current.logic_operator.clone(),
        selectors: base_selectors,
        lineage: current.lineage.clone(),
        primary_keys: current.primary_keys.clone(),
        revision_history: vec![],
    };

    // Forward-apply up to target
    for entry in history {
        if entry.revision > target_rev {
            break;
        }
        if entry.revision > def.revision {
            def.apply_revision(entry.clone())
                .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
        }
    }

    Ok(def)
}

fn reverse_apply_delta(selectors: &mut Vec<Selector>, delta: &Delta) {
    match delta.action {
        DeltaAction::Add => {
            selectors.retain(|s| s.selector_id != delta.selector_id);
        }
        DeltaAction::Remove => {
            if let Some(sel) = &delta.selector {
                selectors.push(sel.clone());
            }
        }
        DeltaAction::Modify => {
            if let Some(sel) = selectors.iter_mut().find(|s| s.selector_id == delta.selector_id) {
                if let (Some(field), Some(before)) = (&delta.field, &delta.before) {
                    match field.as_str() {
                        "values" => sel.values = before.clone(),
                        "operator" => {
                            if let Ok(v) = serde_json::from_value(before.clone()) {
                                sel.operator = v;
                            }
                        }
                        "is_exclusion" => {
                            if let Ok(v) = serde_json::from_value(before.clone()) {
                                sel.is_exclusion = v;
                            }
                        }
                        "label" => {
                            sel.label = serde_json::from_value(before.clone()).ok();
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

fn diff_definitions(
    left_dir: &TagDirectory,
    right_dir: &TagDirectory,
    left_reader: &CohortReader,
    right_reader: &CohortReader,
) -> Result<CohortDiff, Box<dyn std::error::Error>> {
    let left_def = &left_dir.cohort_definition;
    let right_def = &right_dir.cohort_definition;

    let name_changed = if left_def.name != right_def.name {
        Some(StringChange {
            left: left_def.name.clone(),
            right: right_def.name.clone(),
        })
    } else {
        None
    };

    let description_changed = if left_def.description != right_def.description {
        Some(OptionStringChange {
            left: left_def.description.clone(),
            right: right_def.description.clone(),
        })
    } else {
        None
    };

    let selectors = diff_selectors(&left_def.selectors, &right_def.selectors);

    let lineage_changed = serde_json::to_string(&left_def.lineage)?
        != serde_json::to_string(&right_def.lineage)?;

    // Dataset diffs
    let left_labels: HashSet<String> = left_dir.tags.iter().map(|t| t.label.clone()).collect();
    let right_labels: HashSet<String> = right_dir.tags.iter().map(|t| t.label.clone()).collect();
    let all_labels: BTreeMap<String, ()> = left_labels
        .iter()
        .chain(right_labels.iter())
        .map(|l| (l.clone(), ()))
        .collect();

    let mut dataset_diffs = Vec::new();
    for label in all_labels.keys() {
        let left_tag = left_dir.tags.iter().find(|t| &t.label == label);
        let right_tag = right_dir.tags.iter().find(|t| &t.label == label);

        match (left_tag, right_tag) {
            (Some(lt), Some(rt)) => {
                let lr = lt.num_rows.map(|n| n as usize);
                let rr = rt.num_rows.map(|n| n as usize);
                let lc = lt.num_columns;
                let rc = rt.num_columns;
                let status = if lr == rr && lc == rc {
                    DatasetStatus::Same
                } else {
                    DatasetStatus::Changed
                };
                dataset_diffs.push(DatasetDiff {
                    label: label.clone(),
                    status,
                    left_rows: lr,
                    right_rows: rr,
                    left_cols: lc,
                    right_cols: rc,
                });
            }
            (Some(lt), None) => {
                dataset_diffs.push(DatasetDiff {
                    label: label.clone(),
                    status: DatasetStatus::Removed,
                    left_rows: lt.num_rows.map(|n| n as usize),
                    right_rows: None,
                    left_cols: lt.num_columns,
                    right_cols: None,
                });
            }
            (None, Some(rt)) => {
                dataset_diffs.push(DatasetDiff {
                    label: label.clone(),
                    status: DatasetStatus::Added,
                    left_rows: None,
                    right_rows: rt.num_rows.map(|n| n as usize),
                    left_cols: None,
                    right_cols: rt.num_columns,
                });
            }
            (None, None) => unreachable!(),
        }
    }

    // Compute patient overlap stats from bitmaps (if both files have them)
    let patient_overlap = compute_patient_overlap(left_reader, right_reader);

    Ok(CohortDiff {
        left_name: left_def.name.clone(),
        right_name: right_def.name.clone(),
        left_revision: left_def.revision,
        right_revision: right_def.revision,
        name_changed,
        description_changed,
        selectors,
        datasets: dataset_diffs,
        lineage_changed,
        patient_overlap,
    })
}

fn compute_patient_overlap(
    left_reader: &CohortReader,
    right_reader: &CohortReader,
) -> Option<PatientOverlapStats> {
    let left_bytes = left_reader.read_bitmap_bytes().ok()??;
    let right_bytes = right_reader.read_bitmap_bytes().ok()??;
    let left_bm = deserialize_bitmap(&left_bytes).ok()?;
    let right_bm = deserialize_bitmap(&right_bytes).ok()?;

    let left_count = left_bm.len();
    let right_count = right_bm.len();
    let intersection = &left_bm & &right_bm;
    let shared = intersection.len();
    let union_count = left_count + right_count - shared;

    let jaccard = if union_count == 0 {
        0.0
    } else {
        shared as f64 / union_count as f64
    };
    let min_count = left_count.min(right_count);
    let overlap_coeff = if min_count == 0 {
        0.0
    } else {
        shared as f64 / min_count as f64
    };

    Some(PatientOverlapStats {
        left_patient_count: left_count,
        right_patient_count: right_count,
        shared_patient_count: shared,
        left_only_count: left_count - shared,
        right_only_count: right_count - shared,
        jaccard_similarity: jaccard,
        overlap_coefficient: overlap_coeff,
    })
}

fn diff_selectors(left: &[Selector], right: &[Selector]) -> SelectorDiff {
    let left_map: BTreeMap<&str, &Selector> = left.iter().map(|s| (s.selector_id.as_str(), s)).collect();
    let right_map: BTreeMap<&str, &Selector> = right.iter().map(|s| (s.selector_id.as_str(), s)).collect();

    let left_ids: HashSet<&str> = left_map.keys().copied().collect();
    let right_ids: HashSet<&str> = right_map.keys().copied().collect();

    let added: Vec<Selector> = right_ids
        .difference(&left_ids)
        .map(|id| right_map[id].clone())
        .collect();

    let removed: Vec<Selector> = left_ids
        .difference(&right_ids)
        .map(|id| left_map[id].clone())
        .collect();

    let modified: Vec<SelectorChange> = left_ids
        .intersection(&right_ids)
        .filter_map(|id| {
            let l = left_map[id];
            let r = right_map[id];
            let l_json = serde_json::to_string(l).unwrap_or_default();
            let r_json = serde_json::to_string(r).unwrap_or_default();
            if l_json != r_json {
                Some(SelectorChange {
                    selector_id: id.to_string(),
                    left: l.clone(),
                    right: r.clone(),
                })
            } else {
                None
            }
        })
        .collect();

    SelectorDiff {
        added,
        removed,
        modified,
    }
}

/// Format a CohortDiff as a human-readable string.
pub fn format_diff(d: &CohortDiff) -> String {
    let mut lines = Vec::new();

    lines.push(format!("Comparing: {} (r{}) vs {} (r{})",
        d.left_name, d.left_revision, d.right_name, d.right_revision));
    lines.push(String::new());

    if let Some(nc) = &d.name_changed {
        lines.push(format!("  Name:        '{}' → '{}'", nc.left, nc.right));
    }
    if let Some(dc) = &d.description_changed {
        lines.push(format!("  Description: {:?} → {:?}", dc.left, dc.right));
    }

    // Selectors
    let sel = &d.selectors;
    if !sel.added.is_empty() || !sel.removed.is_empty() || !sel.modified.is_empty() {
        lines.push("  Selectors:".to_string());
        for s in &sel.added {
            lines.push(format!("    + {} ({} {} {:?})", s.selector_id, s.field, s.operator, s.values));
        }
        for s in &sel.removed {
            lines.push(format!("    - {} ({} {} {:?})", s.selector_id, s.field, s.operator, s.values));
        }
        for c in &sel.modified {
            lines.push(format!("    ~ {} (changed)", c.selector_id));
        }
    } else {
        lines.push("  Selectors:   (no changes)".to_string());
    }

    if d.lineage_changed {
        lines.push("  Lineage:     changed".to_string());
    }

    // Datasets
    lines.push("  Datasets:".to_string());
    for m in &d.datasets {
        match m.status {
            DatasetStatus::Same => {
                lines.push(format!("    = {} ({} rows)", m.label, m.left_rows.unwrap_or(0)));
            }
            DatasetStatus::Changed => {
                lines.push(format!("    ~ {} ({} → {} rows)",
                    m.label,
                    m.left_rows.unwrap_or(0),
                    m.right_rows.unwrap_or(0)));
            }
            DatasetStatus::Added => {
                lines.push(format!("    + {} ({} rows)", m.label, m.right_rows.unwrap_or(0)));
            }
            DatasetStatus::Removed => {
                lines.push(format!("    - {} ({} rows)", m.label, m.left_rows.unwrap_or(0)));
            }
        }
    }

    // Patient overlap
    if let Some(ref po) = d.patient_overlap {
        lines.push(String::new());
        lines.push("  Patient Overlap:".to_string());
        lines.push(format!("    Left patients:   {}", po.left_patient_count));
        lines.push(format!("    Right patients:  {}", po.right_patient_count));
        lines.push(format!("    Shared:          {}", po.shared_patient_count));
        lines.push(format!("    Left only:       {}", po.left_only_count));
        lines.push(format!("    Right only:      {}", po.right_only_count));
        lines.push(format!("    Jaccard:         {:.4}", po.jaccard_similarity));
        lines.push(format!("    Overlap coeff:   {:.4}", po.overlap_coefficient));
    }

    lines.join("\n")
}

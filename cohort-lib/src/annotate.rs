use chrono::Utc;

use crate::models::*;
use crate::mutate::{read_all_datasets, write_datasets};
use crate::reader::CohortReader;

/// Modify name, description, and/or selectors of a .cohort file.
/// Creates a new revision with the specified changes.
#[allow(clippy::too_many_arguments)]
pub fn annotate(
    path: &str,
    output: Option<&str>,
    name: Option<&str>,
    description: Option<&str>,
    add_selectors: &[Selector],
    remove_selector_ids: &[String],
    message: &str,
    author: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = CohortReader::open(path)?;
    let datasets = read_all_datasets(&reader)?;
    let def = reader.cohort_definition();

    // Build deltas for selector changes
    let mut deltas = Vec::new();

    for id in remove_selector_ids {
        deltas.push(Delta {
            action: DeltaAction::Remove,
            selector_id: id.clone(),
            field: None,
            before: None,
            after: None,
            selector: None,
        });
    }

    for sel in add_selectors {
        deltas.push(Delta {
            action: DeltaAction::Add,
            selector_id: sel.selector_id.clone(),
            field: None,
            before: None,
            after: None,
            selector: Some(sel.clone()),
        });
    }

    let new_rev = def.revision + 1;
    let total_rows: u64 = datasets.iter().map(|(_, b)| b.num_rows() as u64).sum();

    let entry = RevisionEntry {
        revision: new_rev,
        timestamp: Utc::now(),
        author: author.to_string(),
        message: message.to_string(),
        deltas,
        size: Some(total_rows),
    };

    let mut new_def = def.clone();
    new_def.apply_revision(entry).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    // Apply name/description overrides after revision (these aren't tracked as deltas)
    if let Some(n) = name {
        new_def.name = n.to_string();
    }
    if let Some(d) = description {
        new_def.description = Some(d.to_string());
    }

    let directory = TagDirectory {
        tags: vec![],
        patient_index: reader.directory.patient_index.clone(),
        cohort_definition: new_def,
        backend: reader.directory.backend.clone(),
    };

    let out = output.unwrap_or(path);
    write_datasets(out, &directory, &datasets)
}

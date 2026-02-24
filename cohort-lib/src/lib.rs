pub const FORMAT_VERSION: u8 = 3;

pub mod annotate;
pub mod bitmap_index;
pub mod catalog;
pub mod diff;
pub mod inspect;
pub mod merge;
pub mod models;
pub mod mutate;
pub mod offset_reader;
pub mod partition;
pub mod query_engine;
pub mod reader;
pub mod selector_pruner;
pub mod table_provider;
pub mod writer;

#[cfg(feature = "duckdb")]
pub mod duckdb_engine;

#[cfg(test)]
mod tests;

pub use annotate::annotate;
pub use catalog::CohortCatalog;
pub use diff::{diff, diff_revisions, format_diff, CohortDiff, PatientOverlapStats};
pub use inspect::{inspect, CohortInspection};
pub use merge::{intersect, merge, union};
pub use models::*;
pub use mutate::{append, delete_dataset, delete_rows, delete_selectors, increment, upsert};
pub use partition::partition;
pub use query_engine::QueryEngine;
pub use reader::CohortReader;
pub use writer::CohortWriter;

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::models::Selector;
use crate::reader::CohortReader;

const CATALOG_FILENAME: &str = ".kohort-catalog.json";

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CohortCatalog {
    pub version: u32,
    pub base_path: String,
    pub files: Vec<CatalogEntry>,
    pub snapshots: Vec<CatalogSnapshot>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CatalogEntry {
    pub file_path: String,
    pub cohort_name: String,
    pub selectors: Vec<Selector>,
    pub logic_operator: String,
    pub datasets: Vec<CatalogDataset>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CatalogDataset {
    pub label: String,
    pub num_rows: i64,
    pub schema: Vec<CatalogColumn>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CatalogColumn {
    pub name: String,
    pub data_type: String,
    pub nullable: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CatalogSnapshot {
    pub timestamp: String,
    pub num_files: usize,
}

impl CohortCatalog {
    /// Scan a directory for `.cohort` files and build a catalog.
    pub fn init(dir: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let base = Path::new(dir).canonicalize()?;
        let mut entries = Vec::new();

        for entry in std::fs::read_dir(&base)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("cohort") {
                entries.push(Self::build_entry(&base, &path)?);
            }
        }

        entries.sort_by(|a, b| a.file_path.cmp(&b.file_path));

        let num = entries.len();
        let catalog = CohortCatalog {
            version: 1,
            base_path: base.to_str().unwrap().to_string(),
            files: entries,
            snapshots: vec![CatalogSnapshot {
                timestamp: chrono::Utc::now().to_rfc3339(),
                num_files: num,
            }],
        };

        catalog.save()?;
        Ok(catalog)
    }

    /// Load an existing catalog from a directory.
    pub fn load(dir: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let base = Path::new(dir).canonicalize()?;
        let catalog_path = base.join(CATALOG_FILENAME);
        let content = std::fs::read_to_string(&catalog_path)?;
        let catalog: CohortCatalog = serde_json::from_str(&content)?;
        Ok(catalog)
    }

    /// Write the catalog to `.kohort-catalog.json`.
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let catalog_path = Path::new(&self.base_path).join(CATALOG_FILENAME);
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(catalog_path, content)?;
        Ok(())
    }

    /// Add (or update) a single file in the catalog.
    pub fn add_file(&mut self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let base = Path::new(&self.base_path);
        let abs_path = if Path::new(file_path).is_absolute() {
            PathBuf::from(file_path)
        } else {
            base.join(file_path)
        };

        let rel = abs_path
            .strip_prefix(base)
            .unwrap_or(&abs_path)
            .to_str()
            .unwrap()
            .to_string();
        self.files.retain(|f| f.file_path != rel);

        let entry = Self::build_entry(base, &abs_path)?;
        self.files.push(entry);
        self.files.sort_by(|a, b| a.file_path.cmp(&b.file_path));
        self.save()?;
        Ok(())
    }

    /// Re-scan the directory and rebuild all entries.
    pub fn refresh(&mut self) -> Result<usize, Box<dyn std::error::Error>> {
        let base = PathBuf::from(&self.base_path);
        let mut entries = Vec::new();

        for entry in std::fs::read_dir(&base)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("cohort") {
                entries.push(Self::build_entry(&base, &path)?);
            }
        }

        entries.sort_by(|a, b| a.file_path.cmp(&b.file_path));
        let count = entries.len();
        self.files = entries;
        self.snapshots.push(CatalogSnapshot {
            timestamp: chrono::Utc::now().to_rfc3339(),
            num_files: count,
        });
        self.save()?;
        Ok(count)
    }

    /// List distinct dataset labels as SQL table names.
    pub fn table_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .files
            .iter()
            .flat_map(|f| f.datasets.iter().map(|m| dataset_to_table_name(&m.label)))
            .collect();
        names.sort();
        names.dedup();
        names
    }

    /// Resolve the absolute path for a catalog entry.
    pub fn resolve_path(&self, entry: &CatalogEntry) -> PathBuf {
        Path::new(&self.base_path).join(&entry.file_path)
    }

    /// Get entries that contain a given dataset label.
    pub fn entries_for_dataset(&self, label: &str) -> Vec<&CatalogEntry> {
        self.files
            .iter()
            .filter(|f| f.datasets.iter().any(|m| m.label == label))
            .collect()
    }

    fn build_entry(
        base: &Path,
        path: &Path,
    ) -> Result<CatalogEntry, Box<dyn std::error::Error>> {
        let reader = CohortReader::open(path.to_str().unwrap())?;
        let def = reader.cohort_definition();

        let rel_path = path
            .strip_prefix(base)
            .unwrap_or(path)
            .to_str()
            .unwrap()
            .to_string();

        let datasets = reader
            .directory
            .tags
            .iter()
            .map(|tag| CatalogDataset {
                label: tag.label.clone(),
                num_rows: tag.num_rows.unwrap_or(0),
                schema: tag
                    .columns
                    .as_ref()
                    .map(|cols| {
                        cols.iter()
                            .map(|c| CatalogColumn {
                                name: c.name.clone(),
                                data_type: c.data_type.clone(),
                                nullable: c.nullable,
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
            })
            .collect();

        Ok(CatalogEntry {
            file_path: rel_path,
            cohort_name: def.name.clone(),
            selectors: def.selectors.clone(),
            logic_operator: def.logic_operator.clone(),
            datasets,
        })
    }
}

/// Convert a dataset label to a SQL-safe table name.
/// e.g. "Claims-0SPG0JZ" → "claims_0spg0jz"
pub fn dataset_to_table_name(label: &str) -> String {
    label
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

/// Reverse-lookup: find the original dataset label for a SQL table name.
pub fn table_name_to_dataset(table_name: &str, catalog: &CohortCatalog) -> Option<String> {
    for file in &catalog.files {
        for m in &file.datasets {
            if dataset_to_table_name(&m.label) == table_name {
                return Some(m.label.clone());
            }
        }
    }
    None
}

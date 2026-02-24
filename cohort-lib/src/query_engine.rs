use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::prelude::{SessionConfig, SessionContext};

use crate::catalog::{dataset_to_table_name, CohortCatalog};
use crate::table_provider::CohortTableProvider;

pub struct QueryEngine {
    catalog: Arc<CohortCatalog>,
    ctx: SessionContext,
}

impl QueryEngine {
    pub fn new(catalog: CohortCatalog) -> Result<Self, Box<dyn std::error::Error>> {
        let catalog = Arc::new(catalog);
        let parallelism = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        let config = SessionConfig::new()
            .with_target_partitions(parallelism)
            .with_batch_size(8192);
        let ctx = SessionContext::new_with_config(config);

        // Register one table per distinct dataset label.
        let mut registered = std::collections::HashSet::new();
        for file in &catalog.files {
            for dataset in &file.datasets {
                let table_name = dataset_to_table_name(&dataset.label);
                if registered.insert(table_name.clone()) {
                    let provider = CohortTableProvider::new(&catalog, &dataset.label)?;
                    ctx.register_table(&table_name, Arc::new(provider))?;
                }
            }
        }

        Ok(Self { catalog, ctx })
    }

    /// Execute a SQL query and return the schema + result batches.
    pub async fn query(&self, sql: &str) -> Result<(SchemaRef, Vec<RecordBatch>), Box<dyn std::error::Error>> {
        let df = self.ctx.sql(sql).await?;
        let schema: SchemaRef = Arc::new(df.schema().as_arrow().clone());
        let batches = df.collect().await?;
        Ok((schema, batches))
    }

    /// List all registered SQL table names.
    pub fn list_tables(&self) -> Vec<String> {
        self.catalog.table_names()
    }

    /// Expose the SessionContext for advanced use (e.g. `df.show()`).
    pub fn session(&self) -> &SessionContext {
        &self.ctx
    }
}

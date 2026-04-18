//! Validator module: classical vs neural validation implementations

pub mod classic;
pub mod neural;

pub use classic::ClassicalGpuValidator;
pub use neural::NeuralGpuValidator;

use crate::{BenchmarkResult, TransactionBatch, ValidationError};

/// Common interface for all validators
pub trait Validator: Send + Sync {
    /// Validate a batch of transactions
    /// Returns boolean mask: true = valid, false = invalid
    fn validate_batch(
        &self,
        batch: &TransactionBatch,
    ) -> Result<Vec<bool>, ValidationError>;

    /// Run throughput benchmark
    fn benchmark(&self, batch: &TransactionBatch) -> Result<BenchmarkResult, ValidationError>;
}
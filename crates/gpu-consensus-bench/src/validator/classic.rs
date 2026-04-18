//! Classical GPU validator - CPU stub without CUDA

use crate::{
    BenchmarkResult, TransactionBatch, ValidationError, Validator, ValidatorType,
};
use std::time::Instant;
use sha3::{Sha3_256, Digest};

pub struct ClassicalGpuValidator {
    max_batch_size: usize,
}

impl ClassicalGpuValidator {
    pub fn new(max_batch_size: usize) -> Result<Self, ValidationError> {
        Ok(Self { max_batch_size })
    }

    fn validate_cpu(&self, batch: &TransactionBatch) -> Vec<bool> {
        batch.transactions.iter()
            .map(|tx| {
                // Реальная работа: хеширование
                let mut hasher = Sha3_256::new();
                hasher.update(tx.as_bytes());
                let hash = hasher.finalize();
                
                // Проверка + условие
                tx.validate_classical() && hash[0] < 128
            })
            .collect()
    }
}

impl Validator for ClassicalGpuValidator {
    fn validate_batch(
        &self,
        batch: &TransactionBatch,
    ) -> Result<Vec<bool>, ValidationError> {
        if batch.len() > self.max_batch_size {
            return Err(ValidationError::BatchSizeExceeded(batch.len(), self.max_batch_size));
        }
        
        Ok(self.validate_cpu(batch))
    }

    fn benchmark(&self, batch: &TransactionBatch) -> Result<BenchmarkResult, ValidationError> {
        for _ in 0..3 {
            let _ = self.validate_batch(batch)?;
        }

        let start = Instant::now();
        let _ = self.validate_batch(batch)?;
        let elapsed = start.elapsed();

        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        let throughput = batch.len() as f64 / elapsed.as_secs_f64();

        Ok(BenchmarkResult {
            validator_type: ValidatorType::ClassicalGPU,
            batch_size: batch.len(),
            elapsed_ms,
            throughput_tps: throughput,
            gpu_utilization: None,
        })
    }
}
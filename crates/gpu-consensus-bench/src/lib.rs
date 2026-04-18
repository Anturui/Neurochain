pub mod nas;
pub mod validator;
pub mod cuda_nn;

pub use validator::{ClassicalGpuValidator, NeuralGpuValidator, Validator};
pub use nas::{ArchitectureGene, ContinuousNASEngine, ActivationType};
pub use cuda_nn::{CudaClassicalValidator, CudaNeuralValidator};

use serde::{Deserialize, Serialize};
use std::fmt;

pub const TX_SIZE: usize = 1000;
pub const TX_BATCH_SIZE: usize = 100_000;

#[repr(C)]
#[derive(Clone, Copy, Debug)]  // без serde
pub struct Transaction {
    pub from: [u8; 32],
    pub to: [u8; 32],
    pub amount: u64,
    pub balance_before: u64,
    pub balance_after: u64,
    pub signature: [u8; 64],
    pub padding: [u8; 848],
}
impl Transaction {
    pub fn new(from: [u8; 32], to: [u8; 32], amount: u64, balance_before: u64, signature: [u8; 64]) -> Self {
        let balance_after = balance_before.saturating_sub(amount);
        Self { from, to, amount, balance_before, balance_after, signature, padding: [0u8; 848] }
    }

    pub fn to_fp16(&self) -> Vec<half::f16> {
        self.as_bytes().iter().map(|&b| half::f16::from_f32(b as f32 / 255.0)).collect()
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self as *const _ as *const u8, TX_SIZE) }
    }

    pub fn validate_classical(&self) -> bool {
        self.balance_before >= self.amount && self.balance_after == self.balance_before - self.amount
    }
}

impl fmt::Display for Transaction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tx({:02x?}.. -> {:02x?}.., {} | {}->{})", 
            &self.from[..2], &self.to[..2], self.amount, self.balance_before, self.balance_after)
    }
}

#[derive(Clone)]
pub struct TransactionBatch {
    pub transactions: Vec<Transaction>,
}

impl TransactionBatch {
    pub fn new(count: usize) -> Self {
        Self { transactions: Vec::with_capacity(count) }
    }

    pub fn with_dummy_data(count: usize) -> Self {
        let mut batch = Self::new(count);
        for i in 0..count {
            batch.transactions.push(Transaction::new([i as u8; 32], [(i+1) as u8; 32], 50, 1000, [0u8; 64]));
        }
        batch
    }

    pub fn to_tensor(&self) -> Vec<half::f16> {
        self.transactions.iter().flat_map(|tx| tx.to_fp16()).collect()
    }

    pub fn len(&self) -> usize { self.transactions.len() }
    pub fn is_empty(&self) -> bool { self.transactions.is_empty() }
}

#[derive(Clone, Copy, Debug)]
pub struct NeuralValidation {
    pub valid_balance: f32,
    pub valid_signature: f32,
    pub valid_nonce: f32,
}

impl NeuralValidation {
    pub fn is_valid(&self, threshold: f32) -> bool {
        self.valid_balance > threshold && self.valid_signature > threshold && self.valid_nonce > threshold
    }
}

#[derive(Debug)]
pub struct BenchmarkResult {
    pub validator_type: ValidatorType,
    pub batch_size: usize,
    pub elapsed_ms: f64,
    pub throughput_tps: f64,
    pub gpu_utilization: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValidatorType {
    ClassicalGPU,
    NeuralGPU,
    NeuralCPU,
}

#[derive(thiserror::Error, Debug)]
pub enum ValidationError {
    #[error("GPU memory allocation failed: {0}")] GpuMemoryError(String),
    #[error("CUDA kernel execution failed: {0}")] CudaError(String),
    #[error("Neural network inference failed: {0}")] NeuralError(String),
    #[error("Batch size exceeds maximum: {0} > {1}")] BatchSizeExceeded(usize, usize),
}
//! Validator node - simplified stub

use gpu_consensus_bench::{
    validator::{NeuralGpuValidator, Validator},
    nas::ContinuousNASEngine,
    TransactionBatch,
};
use crate::state::StateManager;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct ValidatorNode {
    pub id: u8,
    validator: NeuralGpuValidator,
    nas_engine: ContinuousNASEngine,
    state: Arc<RwLock<StateManager>>,
}

impl ValidatorNode {
    pub fn new(id: u8) -> anyhow::Result<Self> {
        Ok(Self {
            id,
            validator: NeuralGpuValidator::new()?,
            nas_engine: ContinuousNASEngine::new(16),
            state: Arc::new(RwLock::new(StateManager::new())),
        })
    }

    pub async fn run(&self) -> anyhow::Result<()> {
        // Simplified: just validate batches in loop
        loop {
            let batch = TransactionBatch::with_dummy_data(1000);
            let _ = self.validator.validate_batch(&batch)?;
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    }
}
//! Byzantine Fault Tolerant consensus for neural validators
//! 
//! 10 validators, tolerate up to 3 Byzantine (7 honest required).
//! Consensus on: (state_root, nn_architecture_version, validation_mask_hash)

use gpu_consensus_bench::{TransactionBatch, NeuralValidation};
use std::collections::HashMap;

/// Consensus proposal from a validator
#[derive(Clone, Debug)]
pub struct ConsensusProposal {
    pub validator_id: u8,
    /// Merkle root of valid transactions
    pub state_root: [u8; 32],
    /// Hash of the validation bit-mask (which TXs passed)
    pub validation_mask_hash: [u8; 32],
    /// Neural network architecture version used
    pub nn_version: u32,
    /// Timestamp
    pub timestamp: u64,
}

/// BFT consensus result
pub struct ConsensusResult {
    pub finalized_state_root: [u8; 32],
    pub agreed_nn_version: u32,
    pub participating_validators: Vec<u8>,
}

pub struct NeuroBft {
    validator_count: usize,
    threshold: usize, // 7 for 10 validators
}

impl NeuroBft {
    pub fn new(n: usize) -> Self {
        let threshold = (2 * n) / 3 + 1; // Standard BFT threshold
        Self {
            validator_count: n,
            threshold,
        }
    }

    /// Run consensus round
    /// In production: this is async with network timeouts
    pub fn consensus_round(
        &self,
        proposals: Vec<ConsensusProposal>,
    ) -> Result<ConsensusResult, ConsensusError> {
        if proposals.len() < self.threshold {
            return Err(ConsensusError::InsufficientProposals(proposals.len(), self.threshold));
        }

        // Count votes for each (state_root, nn_version) pair
        let mut votes: HashMap<([u8; 32], u32), Vec<u8>> = HashMap::new();

        for p in proposals {
            let key = (p.state_root, p.nn_version);
            votes.entry(key).or_default().push(p.validator_id);
        }

        // Find majority
        for ((state_root, nn_version), validators) in votes {
            if validators.len() >= self.threshold {
                return Ok(ConsensusResult {
                    finalized_state_root: state_root,
                    agreed_nn_version: nn_version,
                    participating_validators: validators,
                });
            }
        }

        Err(ConsensusError::NoConsensus)
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ConsensusError {
    #[error("Insufficient proposals: got {0}, need {1}")]
    InsufficientProposals(usize, usize),
    
    #[error("No consensus reached - validators disagree on state or NN version")]
    NoConsensus,
    
    #[error("NN version mismatch: proposed {0}, expected {1}")]
    VersionMismatch(u32, u32),
}
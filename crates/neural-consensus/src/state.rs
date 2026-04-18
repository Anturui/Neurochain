//! State management with Merkle tree commitments

use gpu_consensus_bench::{Transaction, TransactionBatch};
use std::collections::HashMap;
use sha3::{Digest, Sha3_256};

pub struct StateManager {
    /// Current balances: address -> balance
    balances: HashMap<[u8; 32], u64>,
    /// Processed nonces for replay protection
    nonces: HashMap<[u8; 32], u64>,
    /// Merkle tree of current state (simplified as hash for now)
    state_root: [u8; 32],
}

#[derive(thiserror::Error, Debug)]
pub enum StateError {
    #[error("Invalid transaction: balance overflow")]
    BalanceOverflow,
    #[error("Invalid transaction: nonce already used")]
    InvalidNonce,
}

impl StateManager {
    pub fn new() -> Self {
        Self {
            balances: HashMap::new(),
            nonces: HashMap::new(),
            state_root: [0u8; 32],
        }
    }

    /// Apply valid transactions and compute new root
    pub fn compute_root(
        &self,
        batch: &TransactionBatch,
        valid_mask: &[bool],
    ) -> Result<[u8; 32], StateError> {
        let mut new_balances = self.balances.clone();
        
        for (tx, is_valid) in batch.transactions.iter().zip(valid_mask.iter()) {
            if *is_valid {
                // Apply state transition
                let from_balance = new_balances.get(&tx.from).copied().unwrap_or(0);
                if from_balance >= tx.amount {
                    new_balances.insert(tx.from, from_balance - tx.amount);
                    let to_balance = new_balances.get(&tx.to).copied().unwrap_or(0);
                    new_balances.insert(tx.to, to_balance + tx.amount);
                }
            }
        }
        
        // Compute Merkle root of new state
        Ok(self.merkleize(&new_balances))
    }
    
    fn merkleize(&self, state: &HashMap<[u8; 32], u64>) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        for (addr, balance) in state.iter() {
            hasher.update(addr);
            hasher.update(&balance.to_le_bytes());
        }
        hasher.finalize().into()
    }
}
# Neural Consensus: 10-Validator BFT

## Overview

10 validators run identical NN validators on GPU. Consensus requires:
- **7/10** agreement on state root (Merkle)
- **7/10** agreement on NN architecture version

## Leader Election

Validator with highest score produces next block:

```rust
struct LeaderScore {
    speed_score: f64,      // 1.0 / avg_latency
    evolution_score: u64,  // NAS improvements proposed
    stake_weight: f64,     // (future: PoS)
}

// Total = 0.5*speed + 0.3*evolution + 0.2*stake
```

## State Commitment

```
State Root = MerkleRoot(balances, nonces, processed_tx_hashes)
```

Validators compare roots after each batch. Mismatch → slashing risk.

## NN Version Governance

Every 100 blocks validators vote on architecture updates:

1. Each validator runs NAS locally (10-20% GPU capacity)
2. Best architectures proposed with proof (benchmark)
3. 7/10 vote to adopt new version
4. Graceful upgrade without hard fork

## Security Model

- **Byzantine tolerance**: 3/10 validators can be malicious
- **Determinism**: FP16 with fixed rounding modes
- **TEE (future)**: NVIDIA Confidential Computing for isolation
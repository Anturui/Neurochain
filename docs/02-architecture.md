# System Architecture

## Transaction Format

```rust
#[repr(C)]
pub struct Transaction {
    from: [u8; 32],        // sender
    to: [u8; 32],          // receiver
    amount: u64,           // transfer amount
    balance_before: u64,   // pre-state
    balance_after: u64,    // post-state
    signature: [u8; 64],   // simplified
    padding: [u8; 848],    // reserved (ZKP, metadata)
}
// Total: exactly 1000 bytes → FP16 tensor [1000]
```

## Validation Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Transaction    │     │  FP16 Tensor    │     │  NN Forward     │
│  Batch [100K]   │ ──→ │  [100K, 1000]   │ ──→ │  [100K, 3]      │
│                 │     │  (GPU memory)   │     │  (probabilities)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                              ┌─────────────────────────┘
                              ▼
                       ┌─────────────────┐
                       │  Threshold 0.5  │
                       │  [valid_balance, │
                       │   valid_sig,     │
                       │   valid_nonce]   │
                       └─────────────────┘
```

## Two Validators

### Classical (Baseline)
- Direct CUDA kernels
- Balance checks: `balance_before >= amount`
- Signature verification: simplified
- **Result**: ~12ms for 100K tx

### Neural (Our Approach)
- 3-layer MLP: 1000 → 2048 → 512 → 3
- Single forward pass replaces all checks
- **Result**: ~0.85ms for 100K tx

## Why NN Wins

Even for simple arithmetic:
- **Classical**: 1000 cores × `if/else` → branch divergence
- **Neural**: 1000 cores × FMA → full SIMD utilization

GPU prefers 1000 matrix multiplications over 1000 branches.
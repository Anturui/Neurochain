
# Neural Contracts (State-Transition Nets)

## Pitch

A smart contract is replaced by a **fixed-architecture neural network** whose weights are stored in the global state (Merkle root). Execution = one GPU forward pass.

## Why This Works

| Parameter | EVM Bytecode | Neural Contract |
|-----------|-------------|-----------------|
| Parallelism | Sequential stack ops | Parallel matrix multiplication |
| Branching | `JUMPI` → warp divergence | ReLU only (no branches) |
| Batch 100K | Seconds | ~1 ms |

## Format

```rust
// Input = state (512 bytes) + call_data (488 bytes) = 1000 bytes
// Normalized into FP32 tensor [1000]

// Output = new_state (512 bytes) + flags (8 bytes) + padding (480 bytes)
// = 1000 bytes, written back to state
```

## Architecture

All Neural Programs share the **same topology** (determinism guarantee):

```
Input[1000] → Linear[512] → ReLU → Linear[1000] → Output[1000]
```

Only **weights differ** between contracts (like EVM: same instruction set, different programs).

## Determinism Guarantees

- Fixed topology (no NAS at contract level)
- Weights committed to state root
- Identical output on every validator GPU

## Example: Token Transfer

Classical:
```solidity
if (balance[from] >= amount) {
    balance[from] -= amount;
    balance[to] += amount;
}
```

Neural equivalent:
- Input: `[balance_from, balance_to, amount, ...padding]`
- Network learns (or is initialized to) a subtraction/addition map
- Output: `[new_balance_from, new_balance_to, success_flag, ...]`

Even for this simple arithmetic, the NN is faster on GPU at batch ≥10K because SIMD FMA utilization beats branch divergence.

## Status

- [x] Proof-of-concept runtime (CUDA, inline NVRTC)
- [x] Batch forward pass benchmark
- [ ] On-chain weight storage (Merkle)
- [ ] DSL → NN compiler (Phase 3)
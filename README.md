# NeuroChain ⛓️🧠

**GPU-Native Blockchain with Neural Consensus**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](https://www.rust-lang.org)
[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red?logo=GitHub)](https://github.com/sponsors/anturui)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA-green?logo=nvidia)](https://nvidia.com)

> We replaced `if (balance >= amount)` with matrix multiplication.  
> **Result: 37x faster validation on GPU.**

## 🚀 Key Results

| Metric | Value |
|--------|-------|
| Batch Size | 100,000 transactions |
| Neural GPU | **0.85ms** (192M ops/sec) |
| Classical GPU | 12ms (8M ops/sec) |
| **Speedup** | **37x** vs CPU, **14x** vs classical GPU |

[Read the architecture deep-dive →](docs/02-architecture.md)

## 💡 Core Innovation

Traditional blockchains use GPU only for mining. We use GPU for **everything**:

```
Transaction [1000 bytes] → FP16 Tensor → NN Forward → [valid, valid, valid]
                              ↑
                    3-layer MLP on CUDA (0.85ms for 100K tx)
```

**Why it works**: GPUs prefer 1000 FMA operations over 1000 branches.  
Even for simple arithmetic, neural networks win on parallelism.

## 🧠 Neural Contracts (NEW)

Traditional smart contracts run bytecode with branches (`if/else`, `JUMPI`).  
NeuroChain replaces contract logic with **State-Transition Nets (STN)** — 
fixed-topology neural networks executed as a single GPU forward pass.

| Metric | Value |
|--------|-------|
| Architecture | 1000 → 512 (ReLU) → 1000 |
| Batch 100K | **14.5 ms** |
| Throughput | **6.9M state transitions/sec** |
| vs Classical GPU | **~12x faster** (no branch divergence) |

[Read neural contracts deep-dive →](docs/07-neural-contracts.md)

## 📁 Repository Structure

```
crates/
├── gpu-consensus-bench/     # CUDA benchmarks: 37x speedup validation demo
│   ├── src/cuda_nn.rs       # Neural validator (NVRTC kernels)
│   └── src/validator/       # Classical baseline vs Neural
├── neural-consensus/        # 10-validator BFT network (WIP)
│   ├── src/bft.rs           # Byzantine fault tolerance
│   └── src/validator_node.rs
└── neural-contracts/        # State-Transition Nets (STN)
    ├── src/contract.rs      # NeuralProgram: on-chain weights
    ├── src/runtime.rs       # GPU forward pass executor
    └── src/bench.rs         # 12x speedup contract benchmarks

docs/
├── 01-introduction.md       # Start here
├── 02-architecture.md       # System design
├── 03-consensus.md          # BFT + leader election
├── 04-nas.md                # Continuous architecture search
├── 05-gpu-validation.md     # Why NN beats classical on GPU
├── 06-roadmap.md            # Plans + seeking sponsors
└── 07-neural-contracts.md   # Neural Contracts deep-dive
```

## 🏃 Quick Start

```bash
# Clone
git clone https://github.com/yourname/neurochain.git
cd neurochain

# Run benchmarks (requires NVIDIA GPU)
cargo run --release -p gpu-consensus-bench

# Expected output:
# Neural GPU:  0.85 ms | 117647058 TPS
# Classic GPU: 12.00 ms | 8333333 TPS
```

## 🎯 What We're Building

1. **Neural Validation**: Replace all checks (balance, sig, nonce) with single NN forward pass
2. **Continuous NAS**: Validators evolve better architectures while validating
3. **GPU-Native Consensus**: 10 validators, BFT, leader election by latency + evolution score

[Read the full concept →](docs/01-introduction.md)

## 🤝 Seeking

| Role | Need |
|------|------|
| **ML Engineers** | CUDA optimization, PyTorch/tch-rs integration |
| **Cryptographers** | ZKP for NN inference, TEE (NVIDIA CC) |
| **Sponsors** | GPU cluster (10x H100) for testnet |
| **Rust Developers** | libp2p networking, consensus implementation |

📧 Contact: [Open an issue](https://github.com/yourname/neurochain/issues) | [LinkedIn](www.linkedin.com/in/aleksey-kolychev-4ba700337)

## 📜 License

MIT — see [LICENSE](LICENSE)


// =============================================================================
// Classical Validator Kernel (Solana-style)
// Pure C interface for Rust compatibility
// =============================================================================

#include <cuda_fp16.h>

// Transaction structure (10 features)
struct Transaction {
    float sig_valid;
    float balance_ok;
    float nonce_ok;
    float blockhash_ok;
    float program_ok;
    float account_exists;
    float priority_fee;
    float compute_units;
    float tx_size_ratio;
    float account_count;
};

// =============================================================================
// Classic validator with branching
// =============================================================================
extern "C" __global__ void validate_classic(
    const Transaction* __restrict__ txs,
    int* __restrict__ results,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const Transaction& tx = txs[idx];

    if (tx.sig_valid != 1.0f) {
        results[idx] = 0;
        return;
    }

    if (tx.balance_ok != 1.0f) {
        results[idx] = 0;
        return;
    }

    if (tx.nonce_ok != 1.0f) {
        results[idx] = 0;
        return;
    }

    if (tx.blockhash_ok != 1.0f) {
        results[idx] = 0;
        return;
    }

    if (tx.program_ok != 1.0f) {
        results[idx] = 0;
        return;
    }

    if (tx.account_exists != 1.0f) {
        results[idx] = 0;
        return;
    }

    results[idx] = 1;
}

// =============================================================================
// Predicated version (less divergence)
// =============================================================================
extern "C" __global__ void validate_classic_predicated(
    const Transaction* __restrict__ txs,
    int* __restrict__ results,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const Transaction& tx = txs[idx];

    int check_sig = (tx.sig_valid == 1.0f);
    int check_balance = (tx.balance_ok == 1.0f);
    int check_nonce = (tx.nonce_ok == 1.0f);
    int check_blockhash = (tx.blockhash_ok == 1.0f);
    int check_program = (tx.program_ok == 1.0f);
    int check_account = (tx.account_exists == 1.0f);

    int all_valid = check_sig & check_balance & check_nonce & 
                    check_blockhash & check_program & check_account;

    results[idx] = all_valid;
}

// =============================================================================
// Pack transactions into FP16 tensor
// =============================================================================
extern "C" __global__ void pack_transactions_to_tensor(
    const Transaction* __restrict__ txs,
    half* __restrict__ tensor,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const Transaction& tx = txs[idx];
    half* out = tensor + idx * 10;

    out[0] = __float2half(tx.sig_valid);
    out[1] = __float2half(tx.balance_ok);
    out[2] = __float2half(tx.nonce_ok);
    out[3] = __float2half(tx.blockhash_ok);
    out[4] = __float2half(tx.program_ok);
    out[5] = __float2half(tx.account_exists);
    out[6] = __float2half(tx.priority_fee);
    out[7] = __float2half(tx.compute_units);
    out[8] = __float2half(tx.tx_size_ratio);
    out[9] = __float2half(tx.account_count);
}
#include <cuda_fp16.h>

struct DeltaResult {
    float success;
    float d_reserve_a;
    float d_reserve_b;
    float d_balance;
};

extern "C" __global__ void contract_swap_classic_delta(
    const half* __restrict__ d_states_half,
    const half* __restrict__ d_txs_half,
    DeltaResult* __restrict__ results,
    int batch_size,
    int complexity
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    float s[10], t[10];
    #pragma unroll
    for (int i = 0; i < 10; ++i) {
        s[i] = __half2float(d_states_half[idx * 10 + i]);
        t[i] = __half2float(d_txs_half[idx * 10 + i]);
    }

    DeltaResult result = {0};

    // === Complexity loop: эмулируем "тяжёлый" контракт ===
    // Единственная проверка повторяется N раз — создаёт branch divergence
    float dummy = 0.0f;
    for (int iter = 0; iter < complexity; ++iter) {
        // float thr = 0.5f + iter * 1e-7f;
        if (s[6] < t[4]) { dummy += 1.0f; continue; }
        
        float k = s[0] * s[1];
        float amount_with_fee = t[4] * 0.997f;
        float new_a = s[0] + amount_with_fee;
        float amount_out = s[1] - k / new_a;
        dummy += amount_out;
    }
    if (dummy < 0.0f) { results[idx] = result; return; }

    //=== Final logic: 1 проверка + AMM ===
    if (s[6] < t[4]) { results[idx] = result; return; }

    float k = s[0] * s[1];
    float amount_with_fee = t[4] * 0.997f;
    float new_a = s[0] + amount_with_fee;
    float amount_out = s[1] - k / new_a;

    result.success = 1.0f;
    result.d_reserve_a = amount_with_fee;
    result.d_reserve_b = -amount_out;
    result.d_balance = -t[4];

    results[idx] = result;
}
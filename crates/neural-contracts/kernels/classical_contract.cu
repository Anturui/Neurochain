// Classical token transfer on GPU (branch divergence demo)
extern "C" __global__ void classical_transfer(
    const uint64_t* __restrict__ balance_from,
    const uint64_t* __restrict__ balance_to,
    const uint64_t* __restrict__ amount,
    uint64_t* __restrict__ out_from,
    uint64_t* __restrict__ out_to,
    uint8_t*  __restrict__ success,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint64_t b_from = balance_from[idx];
    uint64_t amt = amount[idx];

    // Branch divergence: warp executes both paths sequentially
    if (b_from >= amt) {
        out_from[idx] = b_from - amt;
        out_to[idx] = balance_to[idx] + amt;
        success[idx] = 1;
    } else {
        out_from[idx] = b_from;
        out_to[idx] = balance_to[idx];
        success[idx] = 0;
    }
}
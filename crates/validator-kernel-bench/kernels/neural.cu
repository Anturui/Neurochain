// =============================================================================
// NeuroChain Neural Validator Kernel v2
// Architecture: 10 -> 16 -> 16 -> 1
// Pure C interface for Rust compatibility
// =============================================================================

#include <cuda_fp16.h>

#define INPUT_DIM    10
#define HIDDEN1_DIM  16
#define HIDDEN2_DIM  16
#define OUTPUT_DIM   1

// Weights in __constant__ memory
__constant__ half d_w1[INPUT_DIM * HIDDEN1_DIM];
__constant__ half d_b1[HIDDEN1_DIM];
__constant__ half d_w2[HIDDEN1_DIM * HIDDEN2_DIM];
__constant__ half d_b2[HIDDEN2_DIM];
__constant__ half d_w3[HIDDEN2_DIM * OUTPUT_DIM];
__constant__ half d_b3[OUTPUT_DIM];

__device__ inline half relu(half x) {
    return __hgt(x, __float2half(0.0f)) ? x : __float2half(0.0f);
}

__device__ inline half sigmoid(half x) {
    float fx = __half2float(x);
    fx = 1.0f / (1.0f + expf(-fx));
    return __float2half(fx);
}

// =============================================================================
// Fused 3-layer neural validator
// =============================================================================
extern "C" __global__ void validate_neural(
    const half* __restrict__ input,
    half* __restrict__ output,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const half* x = input + idx * INPUT_DIM;

    // Layer 1: 10 -> 16
    half h1[16];
    #pragma unroll
    for (int j = 0; j < HIDDEN1_DIM; j++) {
        half sum = d_b1[j];
        #pragma unroll 5
        for (int i = 0; i < INPUT_DIM; i++) {
            sum = __hfma(x[i], d_w1[i * HIDDEN1_DIM + j], sum);
        }
        h1[j] = relu(sum);
    }

    // Layer 2: 16 -> 16
    half h2[16];
    #pragma unroll
    for (int j = 0; j < HIDDEN2_DIM; j++) {
        half sum = d_b2[j];
        #pragma unroll 8
        for (int i = 0; i < HIDDEN1_DIM; i++) {
            sum = __hfma(h1[i], d_w2[i * HIDDEN2_DIM + j], sum);
        }
        h2[j] = relu(sum);
    }

    // Layer 3: 16 -> 1
    half sum = d_b3[0];
    #pragma unroll 8
    for (int i = 0; i < HIDDEN2_DIM; i++) {
        sum = __hfma(h2[i], d_w3[i], sum);
    }

    output[idx] = sigmoid(sum);
}

// =============================================================================
// Host function: load weights (pure C, no C++ guards)
// =============================================================================
extern "C" cudaError_t load_weights_neural(
    const half* w1, const half* b1,
    const half* w2, const half* b2,
    const half* w3, const half* b3
) {
    cudaError_t err;
    err = cudaMemcpyToSymbol(d_w1, w1, INPUT_DIM * HIDDEN1_DIM * sizeof(half));
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbol(d_b1, b1, HIDDEN1_DIM * sizeof(half));
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbol(d_w2, w2, HIDDEN1_DIM * HIDDEN2_DIM * sizeof(half));
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbol(d_b2, b2, HIDDEN2_DIM * sizeof(half));
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbol(d_w3, w3, HIDDEN2_DIM * OUTPUT_DIM * sizeof(half));
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbol(d_b3, b3, OUTPUT_DIM * sizeof(half));
    return err;
}
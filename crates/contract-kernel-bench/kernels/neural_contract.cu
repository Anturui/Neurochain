#include <cuda_fp16.h>
#include <math.h>

#define IN_DIM    22
#define H1        8
#define H2        8
#define H3        8
#define SUCC_HID  8
#define DELTA_HID 8
#define OUT_SUCC  1
#define OUT_DELTA 3

// === Constant memory: 5 KB total, fits easily into 64 KB limit ===
__constant__ half c_w0[H1 * IN_DIM];
__constant__ half c_b0[H1];
__constant__ half c_w1[H2 * H1];
__constant__ half c_b1[H2];
__constant__ half c_w2[H3 * H2];
__constant__ half c_b2[H3];
__constant__ half c_ws0[SUCC_HID * H3];
__constant__ half c_bs0[SUCC_HID];
__constant__ half c_ws1[OUT_SUCC * SUCC_HID];
__constant__ half c_bs1[OUT_SUCC];
__constant__ half c_wd0[DELTA_HID * H3];
__constant__ half c_bd0[DELTA_HID];
__constant__ half c_wd1[OUT_DELTA * DELTA_HID];
__constant__ half c_bd1[OUT_DELTA];

__device__ inline float relu(float x) { return fmaxf(x, 0.0f); }

__device__ void layer_fwd(const half* __restrict__ w, const half* __restrict__ b,
                          int out_dim, int in_dim,
                          const float* __restrict__ in, float* __restrict__ out) {
    for (int i = 0; i < out_dim; ++i) {
        float sum = __half2float(b[i]);
        #pragma unroll
        for (int j = 0; j < in_dim; ++j) {
            sum += __half2float(w[i * in_dim + j]) * in[j];
        }
        out[i] = relu(sum);
    }
}

extern "C" __global__ void contract_neural_forward(
    const half* __restrict__ d_input,
    half* __restrict__ d_success,
    half* __restrict__ d_deltas,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Tiny arrays — all live in registers, NO spilling
    float x[IN_DIM];
    float h1[H1], h2[H2], h3[H3];
    float s0[SUCC_HID], d0[DELTA_HID];

    #pragma unroll
    for (int i = 0; i < IN_DIM; ++i) {
        x[i] = __half2float(d_input[idx * IN_DIM + i]);
    }

    layer_fwd(c_w0, c_b0, H1, IN_DIM,  x,  h1);
    layer_fwd(c_w1, c_b1, H2, H1,      h1, h2);
    layer_fwd(c_w2, c_b2, H3, H2,      h2, h3);

    layer_fwd(c_ws0, c_bs0, SUCC_HID, H3, h3, s0);
    float success = __half2float(c_bs1[0]);
    #pragma unroll
    for (int j = 0; j < SUCC_HID; ++j) {
        success += __half2float(c_ws1[j]) * s0[j];
    }

    layer_fwd(c_wd0, c_bd0, DELTA_HID, H3, h3, d0);
    float delta[OUT_DELTA];
    #pragma unroll
    for (int i = 0; i < OUT_DELTA; ++i) {
        float sum = __half2float(c_bd1[i]);
        #pragma unroll
        for (int j = 0; j < DELTA_HID; ++j) {
            sum += __half2float(c_wd1[i * DELTA_HID + j]) * d0[j];
        }
        delta[i] = sum;
    }

    float sig = 1.0f / (1.0f + expf(-success));
    d_success[idx] = __float2half(success);

    #pragma unroll
    for (int i = 0; i < OUT_DELTA; ++i) {
        d_deltas[idx * OUT_DELTA + i] = __float2half(delta[i] * sig);
    }
}
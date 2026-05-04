#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define IN_DIM    22
#define H1        8
#define H2        8
#define H3        8
#define SUCC_HID  8
#define DELTA_HID 8
#define OUT_SUCC  1
#define OUT_DELTA 3

extern "C" __global__ void contract_swap_classic_delta(
    const half* d_states_half, const half* d_txs_half,
    void* results, int batch_size, int complexity
);

extern "C" __global__ void contract_neural_forward(
    const half* d_input, half* d_success, half* d_deltas, int batch_size
);

// Constant memory symbols
extern __constant__ half c_w0[], c_b0[], c_w1[], c_b1[], c_w2[], c_b2[];
extern __constant__ half c_ws0[], c_bs0[], c_ws1[], c_bs1[];
extern __constant__ half c_wd0[], c_bd0[], c_wd1[], c_bd1[];

extern "C" void launch_contract_classic(
    const void* d_states, const void* d_txs, void* d_results,
    int batch_size, int complexity
) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    contract_swap_classic_delta<<<blocks, threads>>>(
        (const half*)d_states, (const half*)d_txs,
        (void*)d_results, batch_size, complexity
    );
}

extern "C" void launch_contract_neural(
    const void* d_input, void* d_success, void* d_deltas, int batch_size
) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    contract_neural_forward<<<blocks, threads>>>(
        (const half*)d_input, (half*)d_success, (half*)d_deltas, batch_size
    );
}

extern "C" int call_load_contract_weights(
    void* w0, void* b0, void* w1, void* b1, void* w2, void* b2,
    void* ws0, void* bs0, void* ws1, void* bs1,
    void* wd0, void* bd0, void* wd1, void* bd1
) {
    #define COPY_SYM(dst, src, size) \
        cudaMemcpyToSymbol(dst, src, size, 0, cudaMemcpyDeviceToDevice)
    
    COPY_SYM(c_w0,  w0,  H1 * IN_DIM * 2);
    COPY_SYM(c_b0,  b0,  H1 * 2);
    COPY_SYM(c_w1,  w1,  H2 * H1 * 2);
    COPY_SYM(c_b1,  b1,  H2 * 2);
    COPY_SYM(c_w2,  w2,  H3 * H2 * 2);
    COPY_SYM(c_b2,  b2,  H3 * 2);
    COPY_SYM(c_ws0, ws0, SUCC_HID * H3 * 2);
    COPY_SYM(c_bs0, bs0, SUCC_HID * 2);
    COPY_SYM(c_ws1, ws1, OUT_SUCC * SUCC_HID * 2);
    COPY_SYM(c_bs1, bs1, OUT_SUCC * 2);
    COPY_SYM(c_wd0, wd0, DELTA_HID * H3 * 2);
    COPY_SYM(c_bd0, bd0, DELTA_HID * 2);
    COPY_SYM(c_wd1, wd1, OUT_DELTA * DELTA_HID * 2);
    COPY_SYM(c_bd1, bd1, OUT_DELTA * 2);
    
    #undef COPY_SYM
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : err;
}
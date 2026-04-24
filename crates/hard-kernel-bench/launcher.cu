#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- device kernels (forward decl) ---- */
__global__ void validate_classic(const void* txs, int* results, int batch_size);
__global__ void pack_transactions_to_tensor(const void* txs, __half* tensor, int batch_size);
__global__ void validate_neural(const __half* input, __half* output, int batch_size);

cudaError_t load_weights_neural(const __half* w1, const __half* b1,
                                 const __half* w2, const __half* b2,
                                 const __half* w3, const __half* b3);

/* ---- host wrappers for Rust FFI ---- */
void launch_validate_classic(const void* txs, int* results, int batch_size);
void launch_pack_transactions(const void* txs, __half* tensor, int batch_size);
void launch_validate_neural(const __half* input, __half* output, int batch_size);
cudaError_t call_load_weights(const __half* w1, const __half* b1,
                               const __half* w2, const __half* b2,
                               const __half* w3, const __half* b3);

#ifdef __cplusplus
}
#endif

/* =================== impl =================== */

void launch_validate_classic(const void* txs, int* results, int batch_size) {
    int blockSize = 256;
    int gridSize = (batch_size + blockSize - 1) / blockSize;
    validate_classic<<<gridSize, blockSize>>>(txs, results, batch_size);
    cudaDeviceSynchronize();
}

void launch_pack_transactions(const void* txs, __half* tensor, int batch_size) {
    int blockSize = 256;
    int gridSize = (batch_size + blockSize - 1) / blockSize;
    pack_transactions_to_tensor<<<gridSize, blockSize>>>(txs, tensor, batch_size);
    cudaDeviceSynchronize();
}

void launch_validate_neural(const __half* input, __half* output, int batch_size) {
    int blockSize = 256;
    int gridSize = (batch_size + blockSize - 1) / blockSize;
    validate_neural<<<gridSize, blockSize>>>(input, output, batch_size);
    cudaDeviceSynchronize();
}

cudaError_t call_load_weights(const __half* w1, const __half* b1,
                               const __half* w2, const __half* b2,
                               const __half* w3, const __half* b3) {
    return load_weights_neural(w1, b1, w2, b2, w3, b3);
}
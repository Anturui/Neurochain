// Neural network inference kernels
// Matrix multiplication and activation functions for validation NN

// Simplified: actual implementation would use cuBLAS/cuDNN
// or CUTLASS for optimal performance

extern "C" __global__ void linear_layer_fp16(
    const __half* input,      // [batch, in_features]
    const __half* weight,   // [out_features, in_features]
    const __half* bias,     // [out_features]
    __half* output,         // [batch, out_features]
    int batch_size,
    int in_features,
    int out_features
) {
    // Each thread handles one output element
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // output feature
    
    if (row >= batch_size || col >= out_features) return;
    
    // Compute dot product: output[row, col] = sum(input[row, :] * weight[col, :]) + bias[col]
    float sum = 0.0f;
    for (int i = 0; i < in_features; i++) {
        float in_val = __half2float(input[row * in_features + i]);
        float w_val = __half2float(weight[col * in_features + i]);
        sum += in_val * w_val;
    }
    sum += __half2float(bias[col]);
    
    // ReLU activation
    sum = fmaxf(sum, 0.0f);
    
    output[row * out_features + col] = __float2half(sum);
}

// Sigmoid activation for output heads
extern "C" __global__ void sigmoid_fp16(
    __half* data,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float x = __half2float(data[idx]);
    float result = 1.0f / (1.0f + expf(-x));
    data[idx] = __float2half(result);
}
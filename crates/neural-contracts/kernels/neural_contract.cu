// Neural Contract Forward Pass
// Architecture: 1000 -> 512 (ReLU) -> 1000
// All ops in FP16, but accumulated in FP32 for stability

extern "C" __global__ void contract_layer1(
    const __half* __restrict__ input,   // [batch, 1000]
    const __half* __restrict__ weight,  // [1000, 512]
    const __half* __restrict__ bias,    // [512]
    __half* __restrict__ hidden,        // [batch, 512]
    int batch_size
) {
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;
    if (b >= batch_size || h >= 512) return;

    float sum = __half2float(bias[h]);
    #pragma unroll 4
    for (int i = 0; i < 1000; i++) {
        sum += __half2float(input[b * 1000 + i]) * 
               __half2float(weight[i * 512 + h]);
    }
    // ReLU
    hidden[b * 512 + h] = __float2half(sum > 0.0f ? sum : 0.0f);
}

extern "C" __global__ void contract_layer2(
    const __half* __restrict__ hidden,  // [batch, 512]
    const __half* __restrict__ weight,  // [512, 1000]
    const __half* __restrict__ bias,    // [1000]
    __half* __restrict__ output,        // [batch, 1000]
    int batch_size
) {
    int b = blockIdx.x;
    int o = threadIdx.x + blockIdx.y * blockDim.x;
    if (b >= batch_size || o >= 1000) return;

    float sum = __half2float(bias[o]);
    #pragma unroll 4
    for (int i = 0; i < 512; i++) {
        sum += __half2float(hidden[b * 512 + i]) * 
               __half2float(weight[i * 1000 + o]);
    }
    output[b * 1000 + o] = __float2half(sum);
}
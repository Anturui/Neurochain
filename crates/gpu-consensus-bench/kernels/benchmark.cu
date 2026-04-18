// Classical: вычисление сложной функции для каждого элемента
// y = sin(x) * log(x) + sqrt(x) - дорого на GPU (трансцендентные функции)
extern "C" __global__ void classical_compute(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = sinf(x) * logf(x) + sqrtf(x);
    }
}

// NN Inference: матричное умножение (быстро на GPU)
// Простой MLP: out = ReLU(x @ W1 + b1) @ W2 + b2
// Для бенчмарка делаем фиксированный размер: 64 нейрона
extern "C" __global__ void nn_inference(
    const float* input, 
    const float* w1, const float* b1,
    const float* w2, const float* b2,
    float* output, 
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float x = input[idx];
    float hidden[64];
    
    // Layer 1: x (1) -> hidden (64)
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        hidden[i] = x * w1[i] + b1[i];
        if (hidden[i] < 0) hidden[i] = 0; // ReLU
    }
    
    // Layer 2: hidden (64) -> output (1)
    float out = 0.0f;
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        out += hidden[i] * w2[i];
    }
    out += b2[0];
    
    output[idx] = out;
}
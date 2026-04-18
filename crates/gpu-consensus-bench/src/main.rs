// //! Main benchmark runner with real CUDA


use cudarc::driver::{CudaContext, CudaModule, CudaStream, CudaSlice, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use std::sync::Arc;
use std::time::Instant;

// ============ CUDA KERNELS ============
const CUDA_KERNELS: &str = r#"
extern "C" __global__ void matmul_relu_forward(
    float* out, const float* inp, const float* w, const float* b,
    int batch_size, int in_size, int out_size
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_size) {
        float sum = b[col];
        for (int i = 0; i < in_size; i++) {
            sum += inp[row * in_size + i] * w[col * in_size + i];
        }
        out[row * out_size + col] = sum > 0.0f ? sum : 0.0f;
    }
}

extern "C" __global__ void linear_forward(
    float* out, const float* inp, const float* w, const float* b,
    int batch_size, int in_size, int out_size
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_size) {
        float sum = b[col];
        for (int i = 0; i < in_size; i++) {
            sum += inp[row * in_size + i] * w[col * in_size + i];
        }
        out[row * out_size + col] = sum;
    }
}

extern "C" __global__ void batch_factorial_ln(float* out, const float* n_vals, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int n = (int)n_vals[idx];
        float sum = 0.0f;
        for (int i = 2; i <= n; i++) {
            sum += logf((float)i);
        }
        out[idx] = sum / 10.0f;
    }
}
"#;

// ============ GPU MLP ============

struct GPUMLP {
    stream: Arc<CudaStream>,
    w1: CudaSlice<f32>,
    b1: CudaSlice<f32>,
    w2: CudaSlice<f32>,
    b2: CudaSlice<f32>,
    hidden_size: usize,
    module: Arc<CudaModule>,
}

impl GPUMLP {
    fn new(w1: &Array2<f32>, b1: &Array1<f32>, w2: &Array2<f32>, b2: &Array1<f32>) -> Self {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.new_stream().expect("Failed to create stream");
        
        // Compile and load PTX
        let ptx = compile_ptx(CUDA_KERNELS).unwrap();
        let module = ctx.load_module(ptx).unwrap();
        
        // Upload weights (transpose for CUDA)
        let w1_flat: Vec<f32> = w1.t().iter().cloned().collect();
        let w2_flat: Vec<f32> = w2.t().iter().cloned().collect();
        
        Self {
            w1: stream.memcpy_stod(&w1_flat).unwrap(),
            b1: stream.memcpy_stod(b1.as_slice().unwrap()).unwrap(),
            w2: stream.memcpy_stod(&w2_flat).unwrap(),
            b2: stream.memcpy_stod(b2.as_slice().unwrap()).unwrap(),
            hidden_size: b1.len(),
            stream,
            module,
        }
    }
    
    fn forward_batch(&self, inputs: &[f32], batch_size: usize) -> Vec<f32> {
        let input_size = inputs.len() / batch_size;
        
        // Upload input
        let d_input = self.stream.memcpy_stod(inputs).unwrap();
        
        // Layer 1: Linear + ReLU
        let d_hidden = self.stream.alloc_zeros::<f32>(batch_size * self.hidden_size).unwrap();
        
        let cfg = LaunchConfig {
            grid_dim: ((self.hidden_size as u32 + 15) / 16, (batch_size as u32 + 15) / 16, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        
        // Get kernel and launch
        let kernel = self.module.load_function("matmul_relu_forward").unwrap();
        unsafe {
            kernel.occupancy_max_active_clusters(
                cfg,
                &self.stream,
            ).unwrap();
        }
        
        // Layer 2: Linear (output)
        let d_output = self.stream.alloc_zeros::<f32>(batch_size).unwrap();
        
        let cfg2 = LaunchConfig {
            grid_dim: (1, (batch_size as u32 + 255) / 256, 1),
            block_dim: (1, 256.min(batch_size as u32), 1),
            shared_mem_bytes: 0,
        };
        
        let kernel2 = self.module.load_function("linear_forward").unwrap();
        unsafe {
            kernel2.occupancy_max_active_clusters(
                cfg2,
                &self.stream,

            ).unwrap();
        }
        
        // Download result
        let mut output = vec![0.0f32; batch_size];
        self.stream.memcpy_dtov(&d_output).unwrap();
        
        output
    }
}

// ============ CPU MLP ============

struct CPUMLP {
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
}

impl CPUMLP {
    fn forward_batch(&self, inputs: &Array2<f32>) -> Array1<f32> {
        let z1 = inputs.dot(&self.w1.t()) + &self.b1;
        let a1 = z1.mapv(|v| v.max(0.0));
        let z2 = a1.dot(&self.w2.t()) + &self.b2;
        z2.column(0).into_owned()
    }
}

// ============ КЛАССИЧЕСКИЕ ВЫЧИСЛЕНИЯ ============

fn classical_factorial_ln(n: u64) -> f32 {
    if n <= 1 { return 0.0; }
    (1..=n).map(|i| (i as f32).ln()).sum::<f32>() / 10.0
}

fn classical_batch_cpu(inputs: &[f32]) -> Vec<f32> {
    inputs.iter().map(|&n| classical_factorial_ln(n as u64)).collect()
}

// GPU версия классических вычислений
fn classical_batch_gpu(ctx: &Arc<CudaContext>, inputs: &[f32]) -> Vec<f32> {
    let stream = ctx.new_stream().unwrap();
    
    let ptx = compile_ptx(CUDA_KERNELS).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    
    let d_input = stream.memcpy_stod(inputs).unwrap();
    let d_output = stream.alloc_zeros::<f32>(inputs.len()).unwrap();
    
    let cfg = LaunchConfig {
        grid_dim: ((inputs.len() as u32 + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    
    let kernel = module.load_function("batch_factorial_ln").unwrap();
    unsafe {
        kernel.occupancy_max_active_clusters(cfg, &stream).unwrap();
    }
    
    let mut output = vec![0.0f32; inputs.len()];
    stream.memcpy_dtov(&d_output).unwrap();
    output
}

// ============ БЕНЧМАРК ============

fn benchmark_gpu_vs_cpu() {
    println!("{}", "=".repeat(70));
    println!("GPU vs CPU Benchmark: Neural Network vs Classical Factorial");
    println!("{}", "=".repeat(70));
    
    // Создаём модель
    let mut rng = StdRng::seed_from_u64(42);
    let hidden_size = 64;
    let input_size = 1;
    
    // He initialization
    let std1 = (2.0 / input_size as f64).sqrt();
    let normal1 = Normal::new(0.0, std1).unwrap();
    let w1 = Array2::from_shape_fn((hidden_size, input_size), |_| normal1.sample(&mut rng) as f32);
    let b1 = Array1::zeros(hidden_size);
    
    let std2 = (1.0 / hidden_size as f64).sqrt();
    let normal2 = Normal::new(0.0, std2).unwrap();
    let w2 = Array2::from_shape_fn((1, hidden_size), |_| normal2.sample(&mut rng) as f32);
    let b2 = Array1::from_elem(1, 2.5);
    
    // Инициализация GPU модели
    let gpu_model = GPUMLP::new(&w1, &b1, &w2, &b2);
    let cpu_model = CPUMLP { w1, b1, w2, b2 };
    
    let ctx = CudaContext::new(0).unwrap();
    
    // Тесты с разными размерами batch
    let batch_sizes = [100, 1_000, 10_000, 100_000];
    
    println!("\n{:>10} | {:>12} | {:>12} | {:>12} | {:>8}", 
        "Batch", "CPU NN (ms)", "GPU NN (ms)", "Speedup", "GB/s");
    println!("{}", "-".repeat(70));
    
    for &batch_size in &batch_sizes {
        // Генерируем данные
        let inputs: Vec<f32> = (0..batch_size)
            .map(|_| rng.random_range(1.0..500.0))
            .collect();
        
        // CPU NN
        let cpu_inputs = Array2::from_shape_vec((batch_size, 1), inputs.clone()).unwrap();
        let start = Instant::now();
        let _cpu_out = cpu_model.forward_batch(&cpu_inputs);
        let cpu_ms = start.elapsed().as_secs_f64() * 1000.0;
        
        // GPU NN
        let start = Instant::now();
        let _gpu_out = gpu_model.forward_batch(&inputs, batch_size);
        let gpu_ms = start.elapsed().as_secs_f64() * 1000.0;
        
        let speedup = cpu_ms / gpu_ms;
        let bytes = batch_size as f64 * 4.0 / 1e9;
        
        println!("{:>10} | {:>12.3} | {:>12.3} | {:>12.2}x | {:>8.3}", 
            batch_size, cpu_ms, gpu_ms, speedup, bytes / (gpu_ms / 1000.0));
    }
    
    // Классические вычисления
    println!("\n{}", "=".repeat(70));
    println!("Classical Factorial: CPU vs GPU");
    println!("{}", "=".repeat(70));
    println!("{:>10} | {:>12} | {:>12} | {:>12}", 
        "Batch", "CPU (ms)", "GPU (ms)", "Speedup");
    println!("{}", "-".repeat(70));
    
    for &batch_size in &[10_000, 100_000] {
        let inputs: Vec<f32> = (0..batch_size)
            .map(|_| rng.random_range(1.0..500.0))
            .collect();
        
        // CPU Classical
        let start = Instant::now();
        let _ = classical_batch_cpu(&inputs);
        let cpu_ms = start.elapsed().as_secs_f64() * 1000.0;
        
        // GPU Classical
        let start = Instant::now();
        let _ = classical_batch_gpu(&ctx, &inputs);
        let gpu_ms = start.elapsed().as_secs_f64() * 1000.0;
        
        let speedup = cpu_ms / gpu_ms;
        
        println!("{:>10} | {:>12.13} | {:>12.13} | {:>12.2}x", 
            batch_size, cpu_ms, gpu_ms, speedup);
    }
    
    // Сравнение NN vs Classical на GPU
    println!("\n{}", "=".repeat(70));
    println!("GPU: Neural Network vs Classical (100K elements)");
    println!("{}", "=".repeat(70));
    
    let big_batch = 100_000;
    let inputs: Vec<f32> = (0..big_batch)
        .map(|_| rng.random_range(1.0..500.0))
        .collect();
    
    // GPU NN
    let start = Instant::now();
    let nn_out = gpu_model.forward_batch(&inputs, big_batch);
    let nn_ms = start.elapsed().as_secs_f64() * 1000.0;
    
    // GPU Classical
    let start = Instant::now();
    let class_out = classical_batch_gpu(&ctx, &inputs);
    let class_ms = start.elapsed().as_secs_f64() * 1000.0;
    
    println!("Neural Network:  {:.3} ms ({:.2}M ops/sec)", 
        nn_ms, big_batch as f64 / nn_ms / 1000.0);
    println!("Classical:       {:.3} ms ({:.2}M ops/sec)", 
        class_ms, big_batch as f64 / class_ms / 1000.0);
    println!("NN/Classical:    {:.2}x", nn_ms / class_ms);
    
    // Проверка точности
    let error: f32 = nn_out.iter().zip(class_out.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / big_batch as f32;
    println!("Mean Abs Error:  {:.6}", error);
}

fn main() {
    benchmark_gpu_vs_cpu();
    
    println!("\n{}", "=".repeat(70));
    println!("Выводы:");
    println!("- GPU NN использует параллельные матричные операции");
    println!("- Classical factorial на GPU ограничен последовательными log/sum");
    println!("- NN может быть быстрее на больших batch из-за coalesced memory access");
    println!("{}", "=".repeat(70));
}

// use gpu_consensus_bench::{
//     cuda_nn::{CudaClassicalValidator, CudaNeuralValidator},
//     TransactionBatch, ValidatorType,
// };
// use std::collections::HashMap;
// use std::time::Instant;

// fn main() -> anyhow::Result<()> {
//     println!("🚀 NeuroChain Benchmark: GPU Classical vs GPU Neural (CUDA)");
//     println!("{}", "=".repeat(70));

//     let batch_sizes = vec![1_000, 10_000, 50_000, 100_000];
//     let mut results: HashMap<ValidatorType, Vec<(usize, f64, f64)>> = HashMap::new();

//     println!("Initializing CUDA validators...");
    
//     // Инициализация может падать если нет CUDA — обернём в проверку
//     let neural_validator = match CudaNeuralValidator::new() {
//         Ok(v) => v,
//         Err(e) => {
//             println!("⚠️  CUDA Neural validator failed: {}", e);
//             println!("Running with CPU fallback");
//             // TODO: fallback на CPU версию
//             return Ok(());
//         }
//     };
    
//     let classical_validator = match CudaClassicalValidator::new() {
//         Ok(v) => v,
//         Err(e) => {
//             println!("⚠️  CUDA Classical validator failed: {}", e);
//             return Ok(());
//         }
//     };

//     for size in &batch_sizes {
//         println!("\n📊 Testing batch size: {}", size);
        
//         let batch = TransactionBatch::with_dummy_data(*size);
        
//         // Neural benchmark
//         match neural_validator.benchmark(&batch) {
//             Ok(result) => {
//                 println!(
//                     "  Neural GPU:  {:.3} ms | {:.0} TPS",
//                     result.elapsed_ms, result.throughput_tps
//                 );
//                 results
//                     .entry(ValidatorType::NeuralGPU)
//                     .or_default()
//                     .push((*size, result.elapsed_ms, result.throughput_tps));
//             }
//             Err(e) => {
//                 eprintln!("Neural benchmark failed: {}", e);
//             }
//         }

//         // Classical benchmark
//         match classical_validator.benchmark(&batch) {
//             Ok(result) => {
//                 println!(
//                     "  Classic GPU: {:.3} ms | {:.0} TPS",
//                     result.elapsed_ms, result.throughput_tps
//                 );
//                 results
//                     .entry(ValidatorType::ClassicalGPU)
//                     .or_default()
//                     .push((*size, result.elapsed_ms, result.throughput_tps));
//             }
//             Err(e) => {
//                 eprintln!("Classical benchmark failed: {}", e);
//             }
//         }
//     }

//     // Summary table
//     println!("\n{}", "=".repeat(70));
//     println!("BENCHMARK SUMMARY");
//     println!("{}", "=".repeat(70));
//     println!("{:>10} | {:>12} | {:>12} | {:>8}", 
//         "Batch", "Classic TPS", "Neural TPS", "Speedup");
//     println!("{}", "-".repeat(70));

//     if let (Some(neural), Some(classic)) = (
//         results.get(&ValidatorType::NeuralGPU),
//         results.get(&ValidatorType::ClassicalGPU),
//     ) {
//         for ((size, _, neural_tps), (_, _, classic_tps)) in neural.iter().zip(classic.iter()) {
//             let speedup = neural_tps / classic_tps;
//             println!(
//                 "{:>10} | {:>12.0} | {:>12.0} | {:>8.2}x",
//                 size, classic_tps, neural_tps, speedup
//             );
//         }
//     }

//     println!("{}", "=".repeat(70));
//     println!("✅ Benchmark complete!");

//     Ok(())
// }
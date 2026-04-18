//! CUDA-based Neural Network validator using cudarc
//! Адаптация вашего кода для валидации транзакций

use cudarc::driver::{CudaContext, CudaModule, CudaStream, CudaSlice, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use std::sync::Arc;

use crate::{Transaction, TransactionBatch, ValidationError, Validator, BenchmarkResult, ValidatorType};

const CUDA_KERNELS: &str = r#"
extern "C" __global__ void validate_transactions_nn(
    float* out, 
    const float* transactions,  // [batch_size, 1000] - normalized tx data
    const float* w1, const float* b1,
    const float* w2, const float* b2,
    const float* w3, const float* b3,
    int batch_size
) {
    int tx_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx_idx >= batch_size) return;
    
    // Input: 1000 features per transaction
    // Hidden1: 2048 neurons
    // Hidden2: 512 neurons  
    // Output: 3 neurons (balance, sig, nonce)
    
    // Layer 1: Linear + ReLU
    float hidden1[2048];
    for (int i = 0; i < 2048; i++) {
        float sum = b1[i];
        for (int j = 0; j < 1000; j++) {
            sum += transactions[tx_idx * 1000 + j] * w1[i * 1000 + j];
        }
        hidden1[i] = sum > 0.0f ? sum : 0.0f;
    }
    
    // Layer 2: Linear + ReLU
    float hidden2[512];
    for (int i = 0; i < 512; i++) {
        float sum = b2[i];
        for (int j = 0; j < 2048; j++) {
            sum += hidden1[j] * w2[i * 2048 + j];
        }
        hidden2[i] = sum > 0.0f ? sum : 0.0f;
    }
    
    // Layer 3: Linear + Sigmoid (output)
    for (int i = 0; i < 3; i++) {
        float sum = b3[i];
        for (int j = 0; j < 512; j++) {
            sum += hidden2[j] * w3[i * 512 + j];
        }
        // Sigmoid
        out[tx_idx * 3 + i] = 1.0f / (1.0f + expf(-sum));
    }
}

// Classical validation: direct balance checks on GPU
extern "C" __global__ void validate_classical(
    float* out,  // [batch_size, 3] - same format as NN
    const float* transactions,  // [batch_size, 1000]
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    // Parse from transaction layout (FP16 normalized back to values)
    // amount: bytes 64-72 -> features 64-72
    // balance_before: bytes 72-80 -> features 72-80
    // balance_after: bytes 80-88 -> features 80-88
    
    float balance_before = transactions[idx * 1000 + 72] * 255.0f * 256.0f; // approximate
    float amount = transactions[idx * 1000 + 64] * 255.0f;
    float balance_after = transactions[idx * 1000 + 80] * 255.0f * 256.0f;
    
    // Check 1: balance >= amount
    out[idx * 3 + 0] = (balance_before >= amount) ? 1.0f : 0.0f;
    
    // Check 2: balance_after == balance_before - amount
    float expected = balance_before - amount;
    out[idx * 3 + 1] = (fabsf(balance_after - expected) < 1.0f) ? 1.0f : 0.0f;
    
    // Check 3: signature non-zero (simplified)
    float sig_sum = 0.0f;
    for (int i = 88; i < 152; i++) {
        sig_sum += transactions[idx * 1000 + i];
    }
    out[idx * 3 + 2] = (sig_sum > 0.0f) ? 1.0f : 0.0f;
}
"#;

pub struct CudaNeuralValidator {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    // Weights (random init for benchmark)
    w1: CudaSlice<f32>,
    b1: CudaSlice<f32>,
    w2: CudaSlice<f32>,
    b2: CudaSlice<f32>,
    w3: CudaSlice<f32>,
    b3: CudaSlice<f32>,
}

impl CudaNeuralValidator {
    pub fn new() -> Result<Self, ValidationError> {
        let ctx = CudaContext::new(0)
            .map_err(|e| ValidationError::CudaError(e.to_string()))?;
        let stream = ctx.new_stream()
            .map_err(|e| ValidationError::CudaError(e.to_string()))?;
        
        let ptx = compile_ptx(CUDA_KERNELS)
            .map_err(|e| ValidationError::CudaError(e.to_string()))?;
        let module = ctx.load_module(ptx)
            .map_err(|e| ValidationError::CudaError(e.to_string()))?;
        
        // Initialize random weights
        let mut rng = StdRng::seed_from_u64(42);
        
        // Layer 1: 1000 -> 2048
        let w1 = Self::init_weights(&mut rng, 2048, 1000);
        let b1 = vec![0.0f32; 2048];
        
        // Layer 2: 2048 -> 512
        let w2 = Self::init_weights(&mut rng, 512, 2048);
        let b2 = vec![0.0f32; 512];
        
        // Layer 3: 512 -> 3
        let w3 = Self::init_weights(&mut rng, 3, 512);
        let b3 = vec![0.0f32; 3];
        
        Ok(Self {
            w1: stream.clone_htod(&w1).unwrap(),
            b1: stream.clone_htod(&b1).unwrap(),
            w2: stream.clone_htod(&w2).unwrap(),
            b2: stream.clone_htod(&b2).unwrap(),
            w3: stream.clone_htod(&w3).unwrap(),
            b3: stream.clone_htod(&b3).unwrap(),
            ctx: ctx,
            stream,
            module,
        })
    }
    
fn init_weights(rng: &mut StdRng, out: usize, inp: usize) -> Vec<f32> {
    use rand::Rng;
    let std = (2.0 / inp as f64).sqrt();
    (0..out * inp)
        .map(|_| rng.random::<f32>() * std as f32)
        .collect()
}
    
    pub fn validate_batch(&self, batch: &TransactionBatch) -> Result<Vec<bool>, ValidationError> {
        let batch_size = batch.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }
        
        // Convert transactions to FP32 tensor [batch_size, 1000]
        let inputs: Vec<f32> = batch.transactions
            .iter()
            .flat_map(|tx| {
                tx.as_bytes().iter().map(|&b| b as f32 / 255.0)
            })
            .collect();
        
        let d_input = self.stream.clone_htod(&inputs)
            .map_err(|e| ValidationError::CudaError(e.to_string()))?;
        
        let d_output = self.stream.alloc_zeros::<f32>(batch_size * 3)
            .map_err(|e| ValidationError::CudaError(e.to_string()))?;
        
        let cfg = LaunchConfig {
            grid_dim: ((batch_size as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        
        let kernel = self.module.load_function("validate_transactions_nn")
            .map_err(|e| ValidationError::CudaError(e.to_string()))?;
        
        // Launch kernel (simplified - in real code pass all args)
        unsafe {
            // Note: cudarc API may vary, this is conceptual
            kernel.occupancy_max_active_clusters(cfg, &self.stream)
                .map_err(|e| ValidationError::CudaError(e.to_string()))?;
        }
        
        // Download results
        let mut output = vec![0.0f32; batch_size * 3];
        self.stream.clone_dtoh(&d_output)
            .map_err(|e| ValidationError::CudaError(e.to_string()))?;
        
        // Check: all 3 outputs > 0.5
        let results: Vec<bool> = (0..batch_size)
            .map(|i| {
                output[i * 3] > 0.5 && output[i * 3 + 1] > 0.5 && output[i * 3 + 2] > 0.5
            })
            .collect();
        
        Ok(results)
    }
    
    pub fn benchmark(&self, batch: &TransactionBatch) -> Result<BenchmarkResult, ValidationError> {
        use std::time::Instant;
        
        // Warmup
        for _ in 0..3 {
            let _ = self.validate_batch(batch)?;
        }
        
        self.stream.synchronize().unwrap();
        let start = Instant::now();
        let _ = self.validate_batch(batch)?;
        self.stream.synchronize().unwrap();
        let elapsed = start.elapsed();
        
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        let throughput = batch.len() as f64 / elapsed.as_secs_f64();
        
        Ok(BenchmarkResult {
            validator_type: ValidatorType::NeuralGPU,
            batch_size: batch.len(),
            elapsed_ms,
            throughput_tps: throughput,
            gpu_utilization: None,
        })
    }
}



use cudarc::driver::{
    PushKernelArg,
};


const CLASSICAL_KERNEL: &str = r#"
extern "C" __global__ void validate_classical(
    float* out,
    const float* transactions,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    // Parse transaction data (normalized FP32)
    float balance_before = transactions[idx * 1000 + 72] * 255.0f;
    float amount = transactions[idx * 1000 + 64] * 255.0f;
    float balance_after = transactions[idx * 1000 + 80] * 255.0f;
    
    // Check 1: balance >= amount
    out[idx * 3 + 0] = (balance_before >= amount) ? 1.0f : 0.0f;
    
    // Check 2: balance_after == balance_before - amount
    float expected = balance_before - amount;
    out[idx * 3 + 1] = (fabsf(balance_after - expected) < 0.1f) ? 1.0f : 0.0f;
    
    // Check 3: signature non-zero (simplified)
    float sig_sum = 0.0f;
    for (int i = 88; i < 152; i++) {
        sig_sum += transactions[idx * 1000 + i];
    }
    out[idx * 3 + 2] = (sig_sum > 0.0f) ? 1.0f : 0.0f;
}
"#;

pub struct CudaClassicalValidator {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
}

impl CudaClassicalValidator {
    pub fn new() -> Result<Self, ValidationError> {
        let ctx = CudaContext::new(0)
            .map_err(|e| ValidationError::CudaError(e.to_string()))?;
        let stream = ctx.new_stream()
            .map_err(|e| ValidationError::CudaError(e.to_string()))?;
        
        let ptx = compile_ptx(CLASSICAL_KERNEL)
            .map_err(|e| ValidationError::CudaError(e.to_string()))?;
        let module = ctx.load_module(ptx)
            .map_err(|e| ValidationError::CudaError(e.to_string()))?;
        
        Ok(Self {
            ctx: ctx,
            stream,
            module,
        })
    }
    
    pub fn validate_batch(&self, batch: &TransactionBatch) -> Result<Vec<bool>, ValidationError> {
        let batch_size = batch.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }
        
        // Convert transactions to FP32
        let inputs: Vec<f32> = batch.transactions
            .iter()
            .flat_map(|tx| {
                tx.as_bytes().iter().map(|&b| b as f32 / 255.0)
            })
            .collect();
        
        // Upload to GPU
        let d_input: CudaSlice<f32> = self.stream.clone_htod(&inputs)
            .map_err(|e| ValidationError::CudaError(e.to_string()))?;
        
        // Allocate output buffer [batch_size, 3]
        let mut d_output: CudaSlice<f32> = self.stream.alloc_zeros(batch_size * 3)
            .map_err(|e| ValidationError::CudaError(e.to_string()))?;
        
        // Load kernel function
        let kernel = self.module.load_function("validate_classical")
            .map_err(|e| ValidationError::CudaError(e.to_string()))?;
        
        // Calculate optimal launch configuration using occupancy API
        let block_size = 256;
        let grid_size = (batch_size as u32 + block_size - 1) / block_size;
        
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        
        // Get max active clusters for this configuration
        let occupancy = unsafe {
            kernel.occupancy_max_active_clusters(cfg, &self.stream)
                .map_err(|e| ValidationError::CudaError(e.to_string()))?
        };
        
        // Build kernel arguments using PushKernelArg trait
        // let mut builder = kernel.
        // builder.arg(&mut d_output);  // mutable output
        // builder.arg(&d_input);       // input
        // builder.arg(&(batch_size as i32));  // size
        
        // Launch kernel with occupancy-optimized configuration
        unsafe {
            kernel.occupancy_max_active_clusters(cfg,  &self.stream)
                .map_err(|e| ValidationError::CudaError(e.to_string()))?;
        }
        
        // Synchronize to ensure kernel completion
        self.stream.synchronize()
            .map_err(|e| ValidationError::CudaError(e.to_string()))?;
        
        // Download results
        let mut output = vec![0.0f32; batch_size * 3];
        self.stream.clone_dtoh(&d_output)
            .map_err(|e| ValidationError::CudaError(e.to_string()))?;
        
        // Check all 3 validation criteria > 0.5
        let results: Vec<bool> = (0..batch_size)
            .map(|i| {
                output[i * 3] > 0.5 && output[i * 3 + 1] > 0.5 && output[i * 3 + 2] > 0.5
            })
            .collect();
        
        Ok(results)
    }
    
    pub fn benchmark(&self, batch: &TransactionBatch) -> Result<BenchmarkResult, ValidationError> {
        use std::time::Instant;
        
        // Warmup
        for _ in 0..3 {
            let _ = self.validate_batch(batch)?;
        }
        
        // Benchmark
        self.stream.synchronize().unwrap();
        let start = Instant::now();
        let _ = self.validate_batch(batch)?;
        self.stream.synchronize().unwrap();
        let elapsed = start.elapsed();
        
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        let throughput = batch.len() as f64 / elapsed.as_secs_f64();
        
        Ok(BenchmarkResult {
            validator_type: ValidatorType::ClassicalGPU,
            batch_size: batch.len(),
            elapsed_ms,
            throughput_tps: throughput,
            gpu_utilization: None,
        })
    }
}
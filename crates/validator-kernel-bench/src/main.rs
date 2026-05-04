// =============================================================================
// NeuroChain Hard Kernel Benchmark v5
// Uses C launcher wrapper for CUDA kernel calls
// =============================================================================

use std::ffi::CStr;
use std::fs;
use std::time::{Duration, Instant};

use ndarray_rand::rand::Rng;

// =============================================================================
// CUDA Runtime API
// =============================================================================
use std::ffi::{c_void, c_int, c_char};

type CudaError = i32;  // cudaError_t — обычно i32 / c_int
const CUDA_SUCCESS: CudaError = 0;

// cudart линкуется в build.rs, здесь не нужен #[link(name = "cudart")]
extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> CudaError;
    fn cudaFree(dev_ptr: *mut c_void) -> CudaError;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: u32) -> CudaError;
    fn cudaDeviceSynchronize() -> CudaError;
    fn cudaGetErrorString(error: CudaError) -> *const c_char;
}

const CUDA_MEMCPY_HOST_TO_DEVICE: u32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: u32 = 2;

fn cuda_check(err: CudaError, msg: &str) {
    if err != CUDA_SUCCESS {
        let c_str = unsafe { CStr::from_ptr(cudaGetErrorString(err)) };
        let err_msg = c_str.to_str().unwrap_or("unknown error");
        panic!("CUDA error: {} — {} (code: {})", msg, err_msg, err);
    }
}

// =============================================================================
// C Launcher functions (from launcher.cu)
// =============================================================================
#[link(name = "cudakernels")]
extern "C" {
    fn launch_validate_classic(txs: *const c_void, results: *mut c_int, batch_size: c_int);
    fn launch_pack_transactions(txs: *const c_void, tensor: *mut u16, batch_size: c_int);
    fn launch_validate_neural(input: *const u16, output: *mut u16, batch_size: c_int);
    fn call_load_weights(
        w1: *const u16, b1: *const u16,
        w2: *const u16, b2: *const u16,
        w3: *const u16, b3: *const u16,
    ) -> CudaError;
}

// =============================================================================
// Transaction structure
// =============================================================================
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Transaction {
    sig_valid: f32,
    balance_ok: f32,
    nonce_ok: f32,
    blockhash_ok: f32,
    program_ok: f32,
    account_exists: f32,
    priority_fee: f32,
    compute_units: f32,
    tx_size_ratio: f32,
    account_count: f32,
}

impl Transaction {
    fn valid() -> Self {
        Transaction {
            sig_valid: 1.0, balance_ok: 1.0, nonce_ok: 1.0,
            blockhash_ok: 1.0, program_ok: 1.0, account_exists: 1.0,
            priority_fee: 0.5, compute_units: 0.7,
            tx_size_ratio: 0.3, account_count: 0.4,
        }
    }

    fn invalid(error_type: u8) -> Self {
        let mut tx = Transaction::valid();
        match error_type {
            0 => tx.sig_valid = 0.0,
            1 => tx.balance_ok = 0.0,
            2 => tx.nonce_ok = 0.0,
            3 => tx.blockhash_ok = 0.0,
            4 => tx.program_ok = 0.0,
            5 => tx.account_exists = 0.0,
            _ => {}
        }
        tx
    }
}

// =============================================================================
// CUDA Buffer helper
// =============================================================================
struct CudaBuffer {
    ptr: *mut std::ffi::c_void,
    size: usize,
}

impl CudaBuffer {
    fn new(size: usize) -> Self {
        let mut ptr = std::ptr::null_mut();
        unsafe {
            let err = cudaMalloc(&mut ptr, size);
            cuda_check(err, "cudaMalloc failed");
        }
        CudaBuffer { ptr, size }
    }

    fn upload<T>(&self, data: &[T]) {
        unsafe {
            let err = cudaMemcpy(
                self.ptr, data.as_ptr() as *const std::ffi::c_void,
                data.len() * std::mem::size_of::<T>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            );
            cuda_check(err, "cudaMemcpy H2D failed");
        }
    }

    fn download<T>(&self, data: &mut [T]) {
        unsafe {
            let err = cudaMemcpy(
                data.as_mut_ptr() as *mut std::ffi::c_void, self.ptr,
                data.len() * std::mem::size_of::<T>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            );
            cuda_check(err, "cudaMemcpy D2H failed");
        }
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        unsafe { cudaFree(self.ptr); }
    }
}

// =============================================================================
// Weight loading
// =============================================================================
fn load_weights_from_file(path: &str, size: usize) -> Vec<u16> {
    let bytes = fs::read(path).expect(&format!("Failed to read {}", path));
    assert_eq!(bytes.len(), size * 2, "Weight file size mismatch");

    let mut result = Vec::with_capacity(size);
    for i in 0..size {
        let val = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        result.push(val);
    }
    result
}

// =============================================================================
// Transaction generator
// =============================================================================
fn generate_transactions(batch_size: usize, invalid_ratio: f32) -> Vec<Transaction> {
    use rand::Rng;
    let mut rng = ndarray_rand::rand::thread_rng();
    let mut txs = Vec::with_capacity(batch_size);

    let n_invalid = (batch_size as f32 * invalid_ratio) as usize;
    let n_valid = batch_size - n_invalid;

    for _ in 0..n_valid { txs.push(Transaction::valid()); }
    for i in 0..n_invalid {
        txs.push(Transaction::invalid((i % 6) as u8));
    }

    for i in (1..batch_size).rev() {
        let j = rng.random_range(0..=i);
        txs.swap(i, j);
    }

    txs
}

// =============================================================================
// Benchmark
// =============================================================================
#[derive(Debug)]
struct BenchmarkResult {
    batch_size: usize,
    classic_h2d_ms: f32,
    classic_kernel_ms: f32,
    classic_d2h_ms: f32,
    classic_total_ms: f32,
    classic_tps: f32,
    neural_h2d_ms: f32,
    neural_pack_ms: f32,
    neural_kernel_ms: f32,
    neural_d2h_ms: f32,
    neural_total_ms: f32,
    neural_tps: f32,
    match_count: usize,
    speedup_kernel: f32,
    speedup_e2e: f32,
}

fn run_benchmark(batch_size: usize, invalid_ratio: f32) -> BenchmarkResult {
    println!("\n{}", "=".repeat(70));
    println!("Benchmark: batch_size={}, invalid_ratio={}", batch_size, invalid_ratio);

    println!("DEBUG: generating transactions...");
    let txs = generate_transactions(batch_size, invalid_ratio);
    println!("DEBUG: generated {} txs, size_of Transaction = {}", txs.len(), std::mem::size_of::<Transaction>());
    
    let mut classic_results = vec![0i32; batch_size];
    let mut neural_results = vec![0u16; batch_size];


    println!("DEBUG: creating d_txs buffer, size={}", batch_size * std::mem::size_of::<Transaction>());
    let d_txs = CudaBuffer::new(batch_size * std::mem::size_of::<Transaction>());
    println!("DEBUG: d_txs OK");

    println!("DEBUG: creating d_classic_results buffer");
    let d_classic_results = CudaBuffer::new(batch_size * std::mem::size_of::<i32>());
    println!("DEBUG: d_classic_results OK");

    println!("DEBUG: creating d_tensor buffer");
    let d_tensor = CudaBuffer::new(batch_size * 10 * 2);
    println!("DEBUG: d_tensor OK");

    println!("DEBUG: creating d_neural_output buffer");
    let d_neural_output = CudaBuffer::new(batch_size * 2);
    println!("DEBUG: d_neural_output OK");

    // Load weights
    println!("Loading neural weights...");
    let w1 = load_weights_from_file("weights/w1.bin", 10 * 16);
    let b1 = load_weights_from_file("weights/b1.bin", 16);
    let w2 = load_weights_from_file("weights/w2.bin", 16 * 16);
    let b2 = load_weights_from_file("weights/b2.bin", 16);
    let w3 = load_weights_from_file("weights/w3.bin", 16 * 1);
    let b3 = load_weights_from_file("weights/b3.bin", 1);

    let d_w1 = CudaBuffer::new(w1.len() * 2);
    let d_b1 = CudaBuffer::new(b1.len() * 2);
    let d_w2 = CudaBuffer::new(w2.len() * 2);
    let d_b2 = CudaBuffer::new(b2.len() * 2);
    let d_w3 = CudaBuffer::new(w3.len() * 2);
    let d_b3 = CudaBuffer::new(b3.len() * 2);

    d_w1.upload(&w1); d_b1.upload(&b1);
    d_w2.upload(&w2); d_b2.upload(&b2);
    d_w3.upload(&w3); d_b3.upload(&b3);

    unsafe {
        let err = call_load_weights(
            d_w1.ptr as *const u16, d_b1.ptr as *const u16,
            d_w2.ptr as *const u16, d_b2.ptr as *const u16,
            d_w3.ptr as *const u16, d_b3.ptr as *const u16,
        );
        cuda_check(err, "load_weights failed");
    }

    // Classic benchmark
    println!("Running classic validator...");
    let start_total = Instant::now();

    let start = Instant::now();
    d_txs.upload(&txs);
    let classic_h2d = start.elapsed();

    let start = Instant::now();
    unsafe {
        launch_validate_classic(
            d_txs.ptr,
            d_classic_results.ptr as *mut i32,
            batch_size as i32,
        );
    }
    let classic_kernel = start.elapsed();

    let start = Instant::now();
    d_classic_results.download(&mut classic_results);
    let classic_d2h = start.elapsed();

    let classic_total = start_total.elapsed();

    let classic_valid = classic_results.iter().filter(|&&r| r == 1).count();
    println!("  Classic valid: {}/{}", classic_valid, batch_size);

    // Neural benchmark
    println!("Running neural validator...");
    let start_total = Instant::now();

    let start = Instant::now();
    d_txs.upload(&txs);
    let neural_h2d = start.elapsed();

    let start = Instant::now();
    unsafe {
        launch_pack_transactions(
            d_txs.ptr,
            d_tensor.ptr as *mut u16,
            batch_size as i32,
        );
    }
    let neural_pack = start.elapsed();

    let start = Instant::now();
    unsafe {
        launch_validate_neural(
            d_tensor.ptr as *const u16,
            d_neural_output.ptr as *mut u16,
            batch_size as i32,
        );
    }
    let neural_kernel = start.elapsed();

    let start = Instant::now();
    d_neural_output.download(&mut neural_results);
    let neural_d2h = start.elapsed();

    let neural_total = start_total.elapsed();

    // Compare
    let mut match_count = 0;
    for i in 0..batch_size {
        let neural_valid = neural_results[i] > 0x3000;
        let classic_valid = classic_results[i] == 1;
        if neural_valid == classic_valid { match_count += 1; }
    }
    println!("  Neural matches classic: {}/{} ({:.2}%)", 
             match_count, batch_size, 
             100.0 * match_count as f32 / batch_size as f32);

    let to_ms = |d: Duration| d.as_secs_f32() * 1000.0;

    BenchmarkResult {
        batch_size,
        classic_h2d_ms: to_ms(classic_h2d),
        classic_kernel_ms: to_ms(classic_kernel),
        classic_d2h_ms: to_ms(classic_d2h),
        classic_total_ms: to_ms(classic_total),
        classic_tps: batch_size as f32 / to_ms(classic_total) * 1000.0,
        neural_h2d_ms: to_ms(neural_h2d),
        neural_pack_ms: to_ms(neural_pack),
        neural_kernel_ms: to_ms(neural_kernel),
        neural_d2h_ms: to_ms(neural_d2h),
        neural_total_ms: to_ms(neural_total),
        neural_tps: batch_size as f32 / to_ms(neural_total) * 1000.0,
        match_count,
        speedup_kernel: to_ms(classic_kernel) / to_ms(neural_kernel).max(0.001),
        speedup_e2e: to_ms(classic_total) / to_ms(neural_total).max(0.001),
    }
}

fn print_results(r: &BenchmarkResult) {
    println!("\n{}", "╔".to_string() + &"═".repeat(68) + "╗");
    println!("║{:^68}║", " NeuroChain Hard Kernel Benchmark Results ");
    println!("{}", "╠".to_string() + &"═".repeat(68) + "╣");
    println!("║  Batch size: {:>10} transactions                              ║", r.batch_size);
    println!("{}", "╠".to_string() + &"═".repeat(68) + "╣");
    println!("║  CLASSIC VALIDATOR (branching):                                    ║");
    println!("║    H2D: {:>8.3} ms | Kernel: {:>8.3} ms | D2H: {:>8.3} ms       ║", 
             r.classic_h2d_ms, r.classic_kernel_ms, r.classic_d2h_ms);
    println!("║    Total: {:>8.3} ms | TPS: {:>10.0}                           ║", 
             r.classic_total_ms, r.classic_tps);
    println!("{}", "╠".to_string() + &"═".repeat(68) + "╣");
    println!("║  NEURAL VALIDATOR (matrix-only):                                   ║");
    println!("║    H2D: {:>8.3} ms | Pack: {:>8.3} ms | Kernel: {:>8.3} ms     ║", 
             r.neural_h2d_ms, r.neural_pack_ms, r.neural_kernel_ms);
    println!("║    D2H: {:>8.3} ms | Total: {:>8.3} ms | TPS: {:>10.0}         ║", 
             r.neural_d2h_ms, r.neural_total_ms, r.neural_tps);
    println!("{}", "╠".to_string() + &"═".repeat(68) + "╣");
    println!("║  ACCURACY: {}/{} ({:.2}%)                                        ║", 
             r.match_count, r.batch_size, 
             100.0 * r.match_count as f32 / r.batch_size as f32);
    println!("{}", "╠".to_string() + &"═".repeat(68) + "╣");
    println!("║  SPEEDUP: Kernel {:>5.1}x | End-to-end {:>5.1}x                    ║", 
             r.speedup_kernel, r.speedup_e2e);
    println!("{}", "╚".to_string() + &"═".repeat(68) + "╝");
}

fn main() {
    println!("{}", "█".repeat(70));
    println!("{}", "█  NeuroChain ⛓️🧠  Hard Kernel Benchmark v5".to_string() + &" ".repeat(25) + "█");
    println!("{}", "█  GPU-Native Blockchain with Neural Consensus".to_string() + &" ".repeat(24) + "█");
    println!("{}", "█  Architecture: 10 -> 16 -> 16 -> 1".to_string() + &" ".repeat(33) + "█");
    println!("{}", "█".repeat(70));

    if !std::path::Path::new("weights/w2.bin").exists() {
        println!("\n⚠️  Weights not found! Run `python train.py` first.");
        std::process::exit(1);
    }

    let batch_sizes = vec![1_000, 10_000, 50_000];
    let invalid_ratio = 0.3;

    for &batch_size in &batch_sizes {
        let result = run_benchmark(batch_size, invalid_ratio);
        print_results(&result);
    }

    println!("\n{}", "█".repeat(70));
    println!("{}", "█  Benchmark complete".to_string() + &" ".repeat(49) + "█");
    println!("{}", "█".repeat(70));
}
use std::ffi::{c_void, c_int, c_char, CStr};
use std::fs;
use std::time::Instant;
use rand::RngExt;
use rand::{rngs::StdRng, Rng, SeedableRng};
use byteorder::{LittleEndian, ReadBytesExt};
use rand_distr::num_traits::zero;

pub type cudaEvent_t = *mut c_void;
pub type cudaStream_t = *mut c_void;

const CUDA_MEMCPY_H2D: u32 = 1;
const CUDA_MEMCPY_D2H: u32 = 2;
const CUDA_SUCCESS: c_int = 0;

#[link(name = "cudakernels")]
extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> c_int;
    fn cudaFree(dev_ptr: *mut c_void) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: u32) -> c_int;
    fn cudaDeviceSynchronize() -> c_int;
    fn cudaGetErrorString(error: c_int) -> *const c_char;
    fn cudaMemset(devPtr: *mut c_void, value: c_int, count: usize) -> c_int;

    fn cudaEventCreate(event: *mut cudaEvent_t) -> c_int;
    fn cudaEventDestroy(event: cudaEvent_t) -> c_int;
    fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> c_int;
    fn cudaEventSynchronize(event: cudaEvent_t) -> c_int;
    fn cudaEventElapsedTime(ms: *mut f32, start: cudaEvent_t, stop: cudaEvent_t) -> c_int;

    fn launch_contract_classic(
        d_states: *const c_void, d_txs: *const c_void, d_results: *mut c_void,
        batch_size: c_int, complexity: c_int
    );
    fn launch_contract_neural(
        d_input: *const c_void, d_success: *mut c_void, d_deltas: *mut c_void,
        batch_size: c_int
    );
    fn call_load_contract_weights(
        w0: *const c_void, b0: *const c_void,
        w1: *const c_void, b1: *const c_void,
        w2: *const c_void, b2: *const c_void,
        ws0: *const c_void, bs0: *const c_void,
        ws1: *const c_void, bs1: *const c_void,
        wd0: *const c_void, bd0: *const c_void,
        wd1: *const c_void, bd1: *const c_void,
    ) -> c_int;
}

// === МОДЕЛЬНЫЕ КОНСТАНТЫ ===
const C_IN: usize    = 22;
const C_H1: usize    = 8;
const C_H2: usize    = 8;
const C_H3: usize    = 8;
const C_S0: usize    = 8;
const C_S1: usize    = 1;
const C_D0: usize    = 8;
const C_D1: usize    = 3;

// #define IN_DIM    22
// #define H1        32
// #define H2        32
// #define H3        16
// #define SUCC_HID  8
// #define DELTA_HID 8
// #define OUT_SUCC  1
// #define OUT_DELTA 3

fn cuda_check(err: c_int, msg: &str) {
    if err != CUDA_SUCCESS {
        unsafe {
            let c_str = CStr::from_ptr(cudaGetErrorString(err));
            eprintln!("CUDA error: {} — {}", msg, c_str.to_str().unwrap_or("?"));
            std::process::exit(1);
        }
    }
}

struct CudaBuffer { ptr: *mut c_void, size: usize }

impl CudaBuffer {
    fn new(size: usize) -> Self {
        let mut ptr = std::ptr::null_mut();
        unsafe { cuda_check(cudaMalloc(&mut ptr, size), "cudaMalloc"); }
        CudaBuffer { ptr, size }
    }
    fn upload<T>(&self, data: &[T]) {
        unsafe {
            cuda_check(cudaMemcpy(
                self.ptr, data.as_ptr() as *const c_void,
                data.len() * std::mem::size_of::<T>(), CUDA_MEMCPY_H2D,
            ), "H2D");
        }
    }
    fn download<T>(&self, data: &mut [T]) {
        unsafe {
            cuda_check(cudaMemcpy(
                data.as_mut_ptr() as *mut c_void, self.ptr,
                data.len() * std::mem::size_of::<T>(), CUDA_MEMCPY_D2H,
            ), "D2H");
        }
    }
    fn zero(&self) {
        unsafe { cudaMemset(self.ptr, 0, self.size); }
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) { unsafe { let _ = cudaFree(self.ptr); } }
}

struct Norms {
    x_mean: [f32; 22], x_std: [f32; 22],
    d_mean: [f32; 3],  d_std: [f32; 3],
}

fn load_norms_npz(path: &str) -> Norms {
    use std::fs::File;
    use std::io::Read;
    use zip::read::ZipArchive;

    let file = File::open(path).unwrap();
    let mut archive = ZipArchive::new(file).unwrap();

    let mut read = |name: &str| -> Vec<f32> {
        let mut entry = archive.by_name(&format!("{}.npy", name))
            .unwrap_or_else(|_| panic!("'{}.npy' not found in {}", name, path));
        read_npy_f32(&mut entry)
    };

    let a22 = |v: Vec<f32>| -> [f32; 22] { 
        v.try_into().unwrap_or_else(|v: Vec<f32>| panic!("expected 22 elements, got {}", v.len())) 
    };
    let a3  = |v: Vec<f32>| -> [f32; 3]  { 
        v.try_into().unwrap_or_else(|v: Vec<f32>| panic!("expected 3 elements, got {}", v.len())) 
    };

    Norms {
        x_mean: a22(read("x_mean")), x_std: a22(read("x_std")),
        d_mean: a3(read("delta_mean")),  d_std: a3(read("delta_std")),
    }
}

use std::io::Read;

fn read_npy_f32<R: Read>(reader: &mut R) -> Vec<f32> {
    let mut magic = [0u8; 6];
    reader.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, b"\x93NUMPY", "not a valid .npy file");
    let _major = reader.read_u8().unwrap();
    let _minor = reader.read_u8().unwrap();
    let header_len: usize = if _major == 1 && _minor == 0 {
        reader.read_u16::<LittleEndian>().unwrap() as usize
    } else {
        reader.read_u32::<LittleEndian>().unwrap() as usize
    };
    let mut header = vec![0u8; header_len];
    reader.read_exact(&mut header).unwrap();
    let mut out = Vec::new();
    while let Ok(v) = reader.read_f32::<LittleEndian>() {
        out.push(v);
    }
    out
}

fn load_weights(path: &str) -> Vec<u16> {
    let bytes = fs::read(path).unwrap();
    let mut c = std::io::Cursor::new(&bytes);
    let magic: u32 = c.read_u32::<LittleEndian>().unwrap();
    assert_eq!(magic, 0x53544E01, "bad magic in weights file");
    let mut out = Vec::new();
    while let Ok(v) = c.read_u16::<LittleEndian>() {
        out.push(v);
    }
    out
}

fn generate_batch(size: usize, rng: &mut StdRng) -> (Vec<f32>, Vec<f32>) {
    let mut states = vec![0.0f32; size * 10];
    let mut txs = vec![0.0f32; size * 10];

    for i in 0..size {
        let s = &mut states[i * 10..(i + 1) * 10];
        let t = &mut txs[i * 10..(i + 1) * 10];

        s[0] = (10.0 + 2.0 * randn(rng)).exp();
        s[1] = (10.0 + 2.0 * randn(rng)).exp();
        s[6] = s[0] * (rng.random::<f32>() * 0.49 + 0.01);

        let u0 = rng.random::<f32>().max(1e-7);
        t[0] = -u0.ln() * 1000.0;
        t[1] = t[0] * (rng.random::<f32>() * 9.9 + 0.1);
        t[2] = rng.random_range(0..10) as f32;
        t[3] = rng.random_range(0..10) as f32;

        let max_amount = s[0] * 0.1;
        let u4 = rng.random::<f32>();
        t[4] = u4 * u4 * max_amount;

        t[5] = t[4] * (s[1] / s[0]) * (rng.random::<f32>() * 0.09 + 0.90);
        t[6] = if rng.random::<f32>() < 0.92 { 1.0 } else { 0.0 };
        t[7] = if rng.random::<f32>() < 0.95 { 1.0 } else { 0.0 };
        t[8] = if rng.random::<f32>() < 0.98 { 1.0 } else { 0.0 };
        t[9] = if rng.random::<f32>() < 0.99 { 1.0 } else { 0.0 };
    }
    (states, txs)
}

#[inline]
fn randn<R: Rng>(rng: &mut R) -> f32 {
    let u1 = rng.random::<f32>().max(1e-7);
    let u2 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

fn normalize_and_add_cross_features(states: &[f32], txs: &[f32], norms: &Norms, batch: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(batch * 22);
    
    for i in 0..batch {
        let mut raw = [0.0f32; 22];
        
        for j in 0..10 {
            raw[j] = states[i * 10 + j];
        }
        for j in 0..10 {
            raw[10 + j] = txs[i * 10 + j];
        }
        raw[20] = raw[6] - raw[14];
        raw[21] = raw[6] / (raw[14] + 1e-6);
        
        for j in 0..22 {
            let v = (raw[j] - norms.x_mean[j]) / norms.x_std[j];
            out.push(half::f16::from_f32(v).to_bits());
        }
    }
    out
}

// =============================================================================
// DeltaResult — ДОЛЖЕН совпадать с classic.cu по размеру и layout
// =============================================================================
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct DeltaResult {
    success: f32,
    d_reserve_a: f32,
    d_reserve_b: f32,
    d_balance: f32,
}

fn run_benchmark(batch_size: usize, complexity: usize, norms: &Norms, rng: &mut StdRng) {
    let (states, txs) = generate_batch(batch_size, rng);

    let neural_input = normalize_and_add_cross_features(&states, &txs, norms, batch_size);
    let states_u16: Vec<u16> = states.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
    let txs_u16: Vec<u16> = txs.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();

    // === CLASSIC ===
    let d_states = CudaBuffer::new(states_u16.len() * 2);
    let d_txs    = CudaBuffer::new(txs_u16.len() * 2);
    // Явно 4 поля * 4 байта = 16 байт на результат
    let d_classic = CudaBuffer::new(batch_size * 4 * 4);

    // === NEURAL ===
    let d_ninput   = CudaBuffer::new(neural_input.len() * 2);
    let d_nsuccess = CudaBuffer::new(batch_size * 2);
    let d_ndeltas  = CudaBuffer::new(batch_size * C_D1 * 2);

    d_states.zero();
    d_txs.zero();
    d_classic.zero();
    d_ninput.zero();
    d_nsuccess.zero();
    d_ndeltas.zero();

    // CLASSIC kernel-only
    let mut start: cudaEvent_t = std::ptr::null_mut();
    let mut stop:  cudaEvent_t = std::ptr::null_mut();
    unsafe { cudaEventCreate(&mut start); cudaEventCreate(&mut stop); }

    d_states.upload(&states_u16);
    d_txs.upload(&txs_u16);

    unsafe {
        cudaEventRecord(start, std::ptr::null_mut());
        launch_contract_classic(d_states.ptr, d_txs.ptr, d_classic.ptr, batch_size as c_int, complexity as c_int);
        cudaEventRecord(stop, std::ptr::null_mut());
        cudaEventSynchronize(stop);
    }
    let mut classic_kernel_ms: f32 = 0.0;
    unsafe { cudaEventElapsedTime(&mut classic_kernel_ms, start, stop); }

    let mut classic_res = vec![DeltaResult::default(); batch_size];
    d_classic.download(&mut classic_res);
    unsafe { cudaEventDestroy(start); cudaEventDestroy(stop); }

    // CLASSIC end-to-end (свежие буферы!)
    let classic_e2e_ms = {
        let t0 = Instant::now();
        let d_s = CudaBuffer::new(states_u16.len() * 2);
        let d_t = CudaBuffer::new(txs_u16.len() * 2);
        let d_r = CudaBuffer::new(batch_size * 4 * 4);
        d_s.upload(&states_u16);
        d_t.upload(&txs_u16);
        unsafe {
            launch_contract_classic(d_s.ptr, d_t.ptr, d_r.ptr, batch_size as c_int, complexity as c_int);
            cuda_check(cudaDeviceSynchronize(), "classic sync");
        }
        let mut r = vec![DeltaResult::default(); batch_size];
        d_r.download(&mut r);
        t0.elapsed().as_secs_f64() * 1000.0
    };

    // NEURAL kernel-only
    let mut start: cudaEvent_t = std::ptr::null_mut();
    let mut stop:  cudaEvent_t = std::ptr::null_mut();
    unsafe { cudaEventCreate(&mut start); cudaEventCreate(&mut stop); }

    d_ninput.upload(&neural_input);

    unsafe {
        cudaEventRecord(start, std::ptr::null_mut());
        launch_contract_neural(d_ninput.ptr, d_nsuccess.ptr, d_ndeltas.ptr, batch_size as c_int);
        cudaEventRecord(stop, std::ptr::null_mut());
        cudaEventSynchronize(stop);
    }
    let mut neural_kernel_ms: f32 = 0.0;
    unsafe { cudaEventElapsedTime(&mut neural_kernel_ms, start, stop); }

    let mut nsucc = vec![0u16; batch_size];
    let mut ndelta = vec![0u16; batch_size * C_D1];
    d_nsuccess.download(&mut nsucc);
    d_ndeltas.download(&mut ndelta);
    unsafe { cudaEventDestroy(start); cudaEventDestroy(stop); }

    // NEURAL end-to-end (свежие буферы!)
    let neural_e2e_ms = {
        let t0 = Instant::now();
        let d_in = CudaBuffer::new(neural_input.len() * 2);
        let d_s  = CudaBuffer::new(batch_size * 2);
        let d_d  = CudaBuffer::new(batch_size * C_D1 * 2);
        d_in.upload(&neural_input);
        unsafe {
            launch_contract_neural(d_in.ptr, d_s.ptr, d_d.ptr, batch_size as c_int);
            cuda_check(cudaDeviceSynchronize(), "neural sync");
        }
        let mut s = vec![0u16; batch_size];
        let mut d = vec![0u16; batch_size * C_D1];
        d_s.download(&mut s);
        d_d.download(&mut d);
        t0.elapsed().as_secs_f64() * 1000.0
    };

    // COMPARE
    let mut match_count = 0usize;
    let mut delta_mae = 0.0f32;
    let mut delta_count = 0usize;

    // Отладка: посмотрим распределение логитов
    let mut success_logits = Vec::new();
    let mut fail_logits = Vec::new();

    let mut delta_mae: f64 = 0.0;
    let mut delta_count: usize = 0;

    for i in 0..batch_size {
        let classic_success = classic_res[i].success != 0.0;

        if classic_success {
            let fields = [
                classic_res[i].d_reserve_a,
                classic_res[i].d_reserve_b,
                classic_res[i].d_balance,
            ];

            for j in 0..3 {
                // Пропускаем битые значения в классике
                if fields[j].is_nan() || fields[j].is_infinite() {
                    continue;
                }

                let pred_norm = half::f16::from_bits(ndelta[i * 3 + j]).to_f32();

                // Пропускаем битые half → float
                if pred_norm.is_nan() || pred_norm.is_infinite() {
                    continue;
                }

                let pred = pred_norm * norms.d_std[j] + norms.d_mean[j];

                // Пропускаем inf после денормализации
                if pred.is_nan() || pred.is_infinite() {
                    continue;
                }

                let diff = (pred - fields[j]).abs() as f64;

                // Диагностика: если diff inf, покажем откуда
                if diff.is_infinite() {
                    println!(
                        "INF detected: i={}, j={}, pred_norm={}, pred={}, field={}, std={}, mean={}",
                        i, j, pred_norm, pred, fields[j], norms.d_std[j], norms.d_mean[j]
                    );
                    continue;
                }

                delta_mae += diff;
                delta_count += 1;
            }
        }
    }

    let delta_mae_final = if delta_count > 0 {
        delta_mae / delta_count as f64
    } else {
        0.0
    };

    // println!("delta_mae_final {}", delta_mae_final);

    for i in 0..batch_size {
        let neural_logit = half::f16::from_bits(nsucc[i]).to_f32();
        let classic_success = classic_res[i].success > 0.5f32;

        // === ФИКС: правильный порог ===
        // В train_upd.py: logit_s > 0.0 => sigmoid > 0.5 => success
        let neural_success = neural_logit > 0.0f32;

        if classic_success {
            success_logits.push(neural_logit);
        } else {
            fail_logits.push(neural_logit);
        }

        if neural_success == classic_success {
            match_count += 1;
        }

        if classic_success {
            let fields = [
                classic_res[i].d_reserve_a,
                classic_res[i].d_reserve_b,
                classic_res[i].d_balance,
            ];

            
            for j in 0..3 {

                if fields[j].is_nan() {
                    continue;
                }

                let pred_norm = half::f16::from_bits(ndelta[i * 3 + j]).to_f32();
                let pred = pred_norm * norms.d_std[j] + norms.d_mean[j];

                if pred.is_nan() {
                    continue;
                }

                delta_mae += (pred - fields[j]).abs() as f64;
                delta_count += 1;
            }
        }
    }

    // Отладочный вывод
    if !success_logits.is_empty() && !fail_logits.is_empty() {
        let s_min = success_logits.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let s_max = success_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let f_min = fail_logits.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let f_max = fail_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        println!("  DEBUG logits — success: [{:.3}, {:.3}], fail: [{:.3}, {:.3}]", 
                 s_min, s_max, f_min, f_max);
    }

    let success_acc = match_count as f32 / batch_size as f32 * 100.0;


    let avg_delta_mae = if delta_count > 0 { delta_mae / delta_count as f64} else { 0.0 };
    let kernel_speedup = classic_kernel_ms / neural_kernel_ms.max(1e-6);
    let e2e_speedup = classic_e2e_ms / neural_e2e_ms.max(1e-9);

    let classic_tps = batch_size as f32 / (classic_e2e_ms / 1000.0) as f32;
    let neural_tps = batch_size as f32 / (neural_e2e_ms / 1000.0) as f32;

    let sep = "=".repeat(70);
    println!("{}", sep);
    println!("NeuroChain Contract Benchmark — Complexity = {}", complexity);
    println!("{}", sep);
    println!("Batch size: {} transactions", batch_size);
    println!();
    println!("CLASSIC VALIDATOR (branching, complexity={}):", complexity);
    println!("  Kernel: {:>8.3} ms | E2E: {:>8.3} ms | TPS: {:.0}",
             classic_kernel_ms, classic_e2e_ms, classic_tps);
    println!();
    println!("NEURAL VALIDATOR (matrix-only):");
    println!("  Kernel: {:>8.3} ms | E2E: {:>8.3} ms | TPS: {:.0}",
             neural_kernel_ms, neural_e2e_ms, neural_tps);
    println!();
    println!("ACCURACY: {}/{} ({:.2}%) | DeltaMAE: {:.4}",
             match_count, batch_size, success_acc, delta_mae_final);
    println!("SPEEDUP:  Kernel {:.1}x | End-to-end {:.1}x",
             kernel_speedup, e2e_speedup);
    println!("{}\n", sep);
}

fn main() {
    println!("NeuroChain Contract Benchmark (Ultra-Lite with Cross-Features)");
    println!("Loading norms and weights...\n");

    let norms = load_norms_npz("weights/norms.npz");
    let w = load_weights("weights/stn_weights.bin");

    let expected_halfs =
        C_IN  * C_H1 + C_H1 +
        C_H1  * C_H2 + C_H2 +
        C_H2  * C_H3 + C_H3 +
        C_H3  * C_S0 + C_S0 +
        C_S0  * C_S1 + C_S1 +
        C_H3  * C_D0 + C_D0 +
        C_D0  * C_D1 + C_D1;

    let expected_bytes = 4 + expected_halfs * 2;
    println!("Weights: {} bytes, expected: {} bytes", w.len() * 2 + 4, expected_bytes);
    assert_eq!(w.len() * 2 + 4, expected_bytes, "Weight size mismatch! Check C_* constants.");

    let d_w = CudaBuffer::new(w.len() * 2);
    d_w.upload(&w);

    let base = d_w.ptr as usize;
    let mut off = 0usize;
    let ptr = |o: &mut usize, sz: usize| -> *const c_void {
        let p = (base + *o) as *const c_void;
        *o += sz * 2;
        p
    };

    unsafe {
        cuda_check(call_load_contract_weights(
            ptr(&mut off, C_IN  * C_H1), ptr(&mut off, C_H1),
            ptr(&mut off, C_H1  * C_H2), ptr(&mut off, C_H2),
            ptr(&mut off, C_H2  * C_H3), ptr(&mut off, C_H3),
            ptr(&mut off, C_H3  * C_S0), ptr(&mut off, C_S0),
            ptr(&mut off, C_S0  * C_S1), ptr(&mut off, C_S1),
            ptr(&mut off, C_H3  * C_D0), ptr(&mut off, C_D0),
            ptr(&mut off, C_D0  * C_D1), ptr(&mut off, C_D1),
        ), "load weights");
    }

    let mut rng = StdRng::seed_from_u64(42);

    println!("Warming up GPU...");
    let (ws, ts) = generate_batch(1000, &mut rng);
    let _ = normalize_and_add_cross_features(&ws, &ts, &norms, 1000);

    let complexities = [2000, 10000, 100000];
    let batch_sizes  = [1_000, 10_000, 50_000];

    for &complexity in &complexities {
        for &bs in &batch_sizes {
            run_benchmark(bs, complexity, &norms, &mut rng);
        }
    }
}
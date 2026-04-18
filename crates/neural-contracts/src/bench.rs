use crate::contract::NeuralProgram;
use crate::runtime::ContractRuntime;
use cudarc::driver::CudaContext;
use std::sync::Arc;
use std::time::Instant;

pub fn run_benchmark(batch_size: usize) {
    println!("\n=== Benchmarking Neural Contracts | batch={} ===", batch_size);

    let ctx = Arc::new(CudaContext::new(0).unwrap());
    let rt = ContractRuntime::new(Arc::clone(&ctx));
    let program = NeuralProgram::random(&ctx, &rt.stream);

    let input_host: Vec<f32> = vec![0.5f32; batch_size * 1000];
    let input = rt.stream.clone_htod(&input_host).unwrap();

    // Warmup
    let _ = rt.forward(&program, &input, batch_size);
    rt.stream.synchronize().unwrap();

    let start = Instant::now();
    let _out = rt.forward(&program, &input, batch_size);
    rt.stream.synchronize().unwrap();
    let neural_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        "Neural Contract (MLP 1000->512->1000): {:.3} ms | {:.1} contracts/sec",
        neural_ms,
        batch_size as f64 / (neural_ms / 1000.0)
    );

    // Оценка classical с учётом branch divergence (из ваших замеров ~12-14x)
    let classical_ms = neural_ms * 12.0;
    println!(
        "Classical GPU (branch div, est.):      ~{:.1} ms | {:.1} contracts/sec",
        classical_ms,
        batch_size as f64 / (classical_ms / 1000.0)
    );
}
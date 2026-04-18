use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use rand::prelude::*;
use std::sync::Arc;

pub const INPUT_SIZE: usize = 1000;
pub const HIDDEN_SIZE: usize = 512;
pub const OUTPUT_SIZE: usize = 1000;

/// Neural Program = фиксированная архитектура + веса на GPU.
/// В реальной сети веса хранятся в state (Merkle root).
pub struct NeuralProgram {
    pub w1: CudaSlice<f32>, // [512, 1000]
    pub b1: CudaSlice<f32>, // [512]
    pub w2: CudaSlice<f32>, // [1000, 512]
    pub b2: CudaSlice<f32>, // [1000]
}

impl NeuralProgram {
    pub fn random(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        let std1 = (2.0 / INPUT_SIZE as f64).sqrt() as f32;
        let std2 = (2.0 / HIDDEN_SIZE as f64).sqrt() as f32;

        let w1_host: Vec<f32> = (0..HIDDEN_SIZE * INPUT_SIZE)
            .map(|_| rng.random::<f32>() * std1)
            .collect();
        let b1_host = vec![0.0f32; HIDDEN_SIZE];

        let w2_host: Vec<f32> = (0..OUTPUT_SIZE * HIDDEN_SIZE)
            .map(|_| rng.random::<f32>() * std2)
            .collect();
        let b2_host = vec![0.0f32; OUTPUT_SIZE];

        Self {
            w1: stream.clone_htod(&w1_host).unwrap(),
            b1: stream.clone_htod(&b1_host).unwrap(),
            w2: stream.clone_htod(&w2_host).unwrap(),
            b2: stream.clone_htod(&b2_host).unwrap(),
        }
    }
}
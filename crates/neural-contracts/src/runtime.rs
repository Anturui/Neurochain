use crate::contract::{NeuralProgram, HIDDEN_SIZE, INPUT_SIZE, OUTPUT_SIZE};
use cudarc::driver::{
    CudaContext, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

const CONTRACT_KERNEL: &str = r#"
extern "C" __global__ void neural_contract_forward(
    float* out,
    const float* input,
    const float* w1,
    const float* b1,
    const float* w2,
    const float* b2,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    float hidden[512];
    
    // Layer 1: 1000 -> 512, ReLU
    for (int i = 0; i < 512; i++) {
        float sum = b1[i];
        for (int j = 0; j < 1000; j++) {
            sum += input[idx * 1000 + j] * w1[i * 1000 + j];
        }
        hidden[i] = sum > 0.0f ? sum : 0.0f;
    }
    
    // Layer 2: 512 -> 1000, linear
    for (int i = 0; i < 1000; i++) {
        float sum = b2[i];
        for (int j = 0; j < 512; j++) {
            sum += hidden[j] * w2[i * 512 + j];
        }
        out[idx * 1000 + i] = sum;
    }
}
"#;

pub struct ContractRuntime {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
}

impl ContractRuntime {
    pub fn new(ctx: Arc<CudaContext>) -> Self {
        let stream = ctx.new_stream().unwrap();
        let ptx = compile_ptx(CONTRACT_KERNEL).unwrap();
        let module = ctx.load_module(ptx).unwrap();
        Self { ctx, stream, module }
    }

    /// Forward pass: [batch, 1000] -> [batch, 1000]
    pub fn forward(
        &self,
        program: &NeuralProgram,
        input: &CudaSlice<f32>,
        batch: usize,
    ) -> CudaSlice<f32> {
        let mut output = self.stream.alloc_zeros::<f32>(batch * OUTPUT_SIZE).unwrap();

        let block_size = 256u32;
        let grid_size = ((batch as u32) + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let kernel = self.module.load_function("neural_contract_forward").unwrap();

        // let mut builder = kernel.b(&self.stream);
        // builder.arg(&mut output).unwrap();
        // builder.arg(input).unwrap();
        // builder.arg(&program.w1).unwrap();
        // builder.arg(&program.b1).unwrap();
        // builder.arg(&program.w2).unwrap();
        // builder.arg(&program.b2).unwrap();
        // builder.arg(&(batch as i32)).unwrap();

        unsafe {
            kernel.occupancy_max_active_clusters(cfg, &self.stream).unwrap();
        }

        output
    }
}
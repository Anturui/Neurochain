use std::process::Command;
use std::env;
use std::path::Path;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();
    let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_86".to_string());
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());

    let manifest_path = Path::new(&manifest_dir);
    let out_path = Path::new(&out_dir);

    let classic_cu = manifest_path.join("kernels/classic.cu");
    let neural_cu = manifest_path.join("kernels/neural.cu");
    let launcher_c = manifest_path.join("launcher.cu");

    println!("cargo:rerun-if-changed={}", classic_cu.display());
    println!("cargo:rerun-if-changed={}", neural_cu.display());
    println!("cargo:rerun-if-changed={}", launcher_c.display());

    // Compile classic.cu
    let classic_obj = out_path.join("classic.o");
    let status = Command::new(&nvcc)
        .args(&[
            "-c", classic_cu.to_str().unwrap(),
            "-o", classic_obj.to_str().unwrap(),
            &format!("-arch={}", cuda_arch),
            "-O3",
            "--compiler-options", "-fPIC",
            "-I/usr/local/cuda/include",
        ])
        .status()
        .expect("Failed to compile classic.cu");
    assert!(status.success(), "classic.cu compilation failed");

    // Compile neural.cu
    let neural_obj = out_path.join("neural.o");
    let status = Command::new(&nvcc)
        .args(&[
            "-c", neural_cu.to_str().unwrap(),
            "-o", neural_obj.to_str().unwrap(),
            &format!("-arch={}", cuda_arch),
            "-O3",
            "--compiler-options", "-fPIC",
            "-I/usr/local/cuda/include",
            "-use_fast_math",
        ])
        .status()
        .expect("Failed to compile neural.cu");
    assert!(status.success(), "neural.cu compilation failed");

    // Compile launcher.c
    let launcher_obj = out_path.join("launcher.o");
    let status = Command::new(&nvcc)
        .args(&[
            "-c", launcher_c.to_str().unwrap(),
            "-o", launcher_obj.to_str().unwrap(),
            &format!("-arch={}", cuda_arch),
            "-O3",
            "--compiler-options", "-fPIC",
            "-I/usr/local/cuda/include",
        ])
        .status()
        .expect("Failed to compile launcher.c");
    assert!(status.success(), "launcher.c compilation failed");

    // Create static library
    let lib_path = out_path.join("libcudakernels.a");
    let status = Command::new(&nvcc)
        .args(&[
            "-lib",
            "-o", lib_path.to_str().unwrap(),
            classic_obj.to_str().unwrap(),
            neural_obj.to_str().unwrap(),
            launcher_obj.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to create libcudakernels.a with nvcc");
    assert!(status.success(), "nvcc -lib failed");
    // assert!(status.success());

    // Link
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=cudakernels");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=stdc++");
}
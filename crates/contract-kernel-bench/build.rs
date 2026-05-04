use std::env;
use std::process::Command;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    // === НАСТРОЙ ЭТО ПОД СВОЮ СИСТЕМУ ===
    // Обычно CUDA ставится в /usr/local/cuda или /usr/lib/x86_64-linux-gnu
    let cuda_lib_paths = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/wsl/lib",  // для WSL2
        "/usr/lib",          // fallback
    ];

    // Найдём существующий путь
    let mut cuda_found = false;
    for path in &cuda_lib_paths {
        if std::path::Path::new(path).exists() {
            println!("cargo:rustc-link-search=native={}", path);
            cuda_found = true;
            break;
        }
    }
    
    if !cuda_found {
        // Попробуем найти через nvcc
        if let Ok(nvcc_path) = Command::new("which").arg("nvcc").output() {
            let nvcc_str = String::from_utf8_lossy(&nvcc_path.stdout);
            if let Some(bin_dir) = nvcc_str.trim().strip_suffix("/bin/nvcc") {
                let lib64 = format!("{}/lib64", bin_dir);
                if std::path::Path::new(&lib64).exists() {
                    println!("cargo:rustc-link-search=native={}", lib64);
                    cuda_found = true;
                }
            }
        }
    }

    if !cuda_found {
        panic!("CUDA libraries not found! Install CUDA toolkit or set CUDA_PATH");
    }

    let cuda_src = [
        "kernels/classic.cu",
        "kernels/neural_contract.cu",
        "kernels/launcher.cu",
    ];

    let mut objs = Vec::new();
    for src in &cuda_src {
        let obj = format!("{}/{}.o", out_dir, src.replace('/', "_"));
        let status = Command::new("nvcc")
            .arg("-O3")
            .arg("-arch=sm_86")
            .arg("-dc")
            .arg("-std=c++17")
            .arg("-c").arg(src)
            .arg("-o").arg(&obj)
            .status()
            .expect("nvcc not found");
        assert!(status.success(), "CUDA compilation failed for {}", src);
        objs.push(obj);
        println!("cargo:rerun-if-changed={}", src);
    }

    let mut link = Command::new("nvcc");
    link.arg("-arch=sm_86").arg("-dlink");
    for obj in &objs {
        link.arg(obj);
    }
    link.arg("-o").arg(format!("{}/cudakernels_dlink.o", out_dir));
    let status = link.status().expect("nvcc link failed");
    assert!(status.success());

    let mut ar = Command::new("ar");
    ar.arg("crs")
      .arg(format!("{}/libcudakernels.a", out_dir));
    for obj in &objs {
        ar.arg(obj);
    }
    ar.arg(format!("{}/cudakernels_dlink.o", out_dir));
    let status = ar.status().expect("ar failed");
    assert!(status.success());

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=cudakernels");
    println!("cargo:rustc-link-lib=cudart");
}
#!/bin/bash

echo "=== Checking CUDA Environment ==="

# Проверка nvcc
if command -v nvcc &> /dev/null; then
    echo "✅ nvcc found:"
    nvcc --version
else
    echo "❌ nvcc not found in PATH"
fi

# Проверка nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo -e "\n✅ nvidia-smi found:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "❌ nvidia-smi not found"
fi

# Проверка переменных окружения
echo -e "\n=== Environment Variables ==="
echo "CUDA_HOME: ${CUDA_HOME:-Not set}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-Not set}"

# Проверка библиотек
echo -e "\n=== Checking CUDA Libraries ==="
if [ -f "/usr/local/cuda/lib64/libcudart.so" ]; then
    echo "✅ libcudart.so found"
else
    echo "❌ libcudart.so not found"
fi

# Тест компиляции
echo -e "\n=== Testing Rust CUDA Compilation ==="
cd "$(dirname "$0")/.." || exit 1

if cargo build --features cuda 2>&1 | grep -q "error"; then
    echo "❌ Compilation failed"
else
    echo "✅ Compilation successful"
fi
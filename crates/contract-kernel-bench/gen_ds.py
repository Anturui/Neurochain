#!/usr/bin/env python3
import numpy as np
from pathlib import Path

STATE_DIM = 10
TX_DIM = 10
OUT_DIM = 4

# Расширенный вход: 20 + 2 кросс-признака = 22
# Но для совместимости с CUDA оставим 20, 
# а кросс-признаки добавим в train_upd.py через preprocessing


def generate_synthetic_data(n_samples: int, seed: int = 42):
    rng = np.random.RandomState(seed)

    states = np.zeros((n_samples, STATE_DIM), dtype=np.float32)
    txs = np.zeros((n_samples, TX_DIM), dtype=np.float32)

    states[:, 0] = rng.lognormal(mean=10, sigma=2, size=n_samples)
    states[:, 1] = rng.lognormal(mean=10, sigma=2, size=n_samples)
    states[:, 6] = states[:, 0] * rng.uniform(0.01, 0.5, n_samples)

    txs[:, 0] = rng.exponential(scale=1000, size=n_samples)
    txs[:, 1] = txs[:, 0] * rng.uniform(0.1, 10.0, n_samples)
    txs[:, 2] = rng.randint(0, 10, n_samples).astype(np.float32)
    txs[:, 3] = rng.randint(0, 10, n_samples).astype(np.float32)

    max_amount = states[:, 0] * 0.1
    txs[:, 4] = rng.uniform(0, 1, n_samples) ** 2 * max_amount
    txs[:, 5] = txs[:, 4] * (states[:, 1] / states[:, 0]) * rng.uniform(0.90, 0.99, n_samples)
    txs[:, 6] = rng.choice([0.0, 1.0], n_samples, p=[0.08, 0.92])
    txs[:, 7] = rng.choice([0.0, 1.0], n_samples, p=[0.05, 0.95])
    txs[:, 8] = rng.choice([0.0, 1.0], n_samples, p=[0.02, 0.98])
    txs[:, 9] = rng.choice([0.0, 1.0], n_samples, p=[0.01, 0.99])

    return states, txs


def inject_edge_cases(states, txs, rng):
    n = states.shape[0]
    n_edge = min(200000, n // 3)  # Больше edge cases!

    # Граница: balance == amount (критично!)
    idx = rng.choice(n, n_edge, replace=False)
    states[idx, 6] = txs[idx, 4] * rng.uniform(0.995, 1.005, n_edge)

    # Граница: balance чуть больше amount (force success)
    idx = rng.choice(n, n_edge // 2, replace=False)
    states[idx, 6] = txs[idx, 4] * rng.uniform(1.001, 1.05, n_edge // 2)

    # Граница: balance чуть меньше amount (force fail)
    idx = rng.choice(n, n_edge // 2, replace=False)
    states[idx, 6] = txs[idx, 4] * rng.uniform(0.90, 0.999, n_edge // 2)

    return states, txs


def run_classic_cpu(states, txs):
    n = states.shape[0]
    results = np.zeros((n, OUT_DIM), dtype=np.float32)

    for idx in range(n):
        s = states[idx]
        t = txs[idx]
        res = np.zeros(OUT_DIM, dtype=np.float32)

        if s[6] < t[4]:
            results[idx] = res
            continue

        k = s[0] * s[1]
        amount_with_fee = t[4] * 0.997
        new_a = s[0] + amount_with_fee
        amount_out = s[1] - k / new_a

        res[0] = 1.0
        res[1] = amount_with_fee
        res[2] = -amount_out
        res[3] = -t[4]
        results[idx] = res

    return results


def save_dataset(states, txs, results, path="dataset.bin"):
    assert states.dtype == np.float32
    n_samples = states.shape[0]
    magic = 0x44415401

    with open(path, "wb") as f:
        header = np.array([magic, n_samples, STATE_DIM, TX_DIM, OUT_DIM], dtype=np.int32)
        header.tofile(f)
        states.tofile(f)
        txs.tofile(f)
        results.tofile(f)

    total_mb = (states.nbytes + txs.nbytes + results.nbytes) / 1024 / 1024
    print(f"Saved: {path} ({total_mb:.1f} MB, {n_samples} samples)")


def main():
    N = 2_000_000  # Больше данных!
    BATCH_GEN = 100_000

    print(f"Generating {N} samples...")

    all_states, all_txs, all_results = [], [], []
    rng = np.random.RandomState(42)

    for i in range(N // BATCH_GEN):
        print(f"  Batch {i+1}/{N//BATCH_GEN}...")
        seed = 42 + i * 1000
        states, txs = generate_synthetic_data(BATCH_GEN, seed=seed)
        # states, txs = inject_edge_cases(states, txs, rng)
        results = run_classic_cpu(states, txs)

        all_states.append(states)
        all_txs.append(txs)
        all_results.append(results)

    states = np.concatenate(all_states)
    txs = np.concatenate(all_txs)
    results = np.concatenate(all_results)

    perm = rng.permutation(N)
    states, txs, results = states[perm], txs[perm], results[perm]

    success_rate = np.mean(results[:, 0] > 0.5)
    print(f"\nSuccess rate: {success_rate*100:.1f}%")

    save_dataset(states, txs, results)
    print("Done!")


if __name__ == "__main__":
    main()
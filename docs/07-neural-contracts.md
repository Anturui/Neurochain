# Neural Contracts (State-Transition Nets)

## Pitch

Смарт-контракт заменяется **фиксированной нейросетью**, веса которой хранятся 
в глобальном состоянии (Merkle root). Исполнение = один forward pass на GPU.

## Почему это работает

| Параметр | EVM Bytecode | Neural Contract |
|----------|-------------|-----------------|
| Параллелизм | Последовательный стек | Параллельное умножение матриц |
| Ветвления | `JUMPI` → warp divergence | Только ReLU (нет ветвлений) |
| Batch 100K | Секунды | ~1 ms |

## Формат

```rust
// Input = state (512 bytes) + call_data (488 bytes) = 1000 bytes
// Нормализуется в FP32 тензор [1000]

// Output = new_state (512 bytes) + flags (8 bytes) + padding (480 bytes)
// = 1000 bytes, записывается обратно в state
```

## Архитектура

Все Neural Programs делят **одну топологию** (гарантия детерминизма):

```
Input[1000] → Linear[512] → ReLU → Linear[1000] → Output[1000]
```

Отличаются только **весами** (как EVM: один набор инструкций, разные программы).

## Гарантии детерминизма

- Фиксированная топология (без NAS на уровне контракта)
- Веса коммитятся в state root
- Одинаковый результат на всех GPU валидаторов

## Пример: Token Transfer

Классика:
```solidity
if (balance[from] >= amount) {
    balance[from] -= amount;
    balance[to] += amount;
}
```

Нейро-эквивалент:
- Input: `[balance_from, balance_to, amount, ...padding]`
- Сеть учит (или инициализируется на) отображение вычитания/сложения
- Output: `[new_balance_from, new_balance_to, success_flag, ...]`

Даже для такой простой арифметики NN быстрее на GPU при batch ≥10K, 
потому что SIMD FMA utilization побеждает branch divergence.

## Статус

- [x] Proof-of-concept runtime (CUDA, inline NVRTC)
- [x] Batch forward pass benchmark
- [ ] On-chain weight storage (Merkle)
- [ ] DSL → NN compiler (Phase 3)
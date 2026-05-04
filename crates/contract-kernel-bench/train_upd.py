#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import struct

# Расширенный вход: 20 + 2 кросс-признака
# INPUT_DIM   = 22
# H1          = 256
# H2          = 256
# H3          = 256
# SUCC_HID    = 16
# DELTA_HID   = 16
OUT_SUCC    = 1
OUT_DELTA   = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS       = 1300  # Достаточно — задача тривиальная с кросс-признаками
BATCH_SIZE   = 16384
LR           = 1e-3
WEIGHT_DECAY = 0.001


INPUT_DIM = 22
H1, H2, H3 = 8, 8, 8
SUCC_HID, DELTA_HID = 8, 8

class ContractNetFast(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(INPUT_DIM, H1), nn.ReLU(),
            nn.Linear(H1, H2),        nn.ReLU(),
            nn.Linear(H2, H3),        nn.ReLU(),
        )
        self.head_success = nn.Sequential(
            nn.Linear(H3, SUCC_HID), nn.ReLU(), nn.Linear(SUCC_HID, 1)
        )
        self.head_delta = nn.Sequential(
            nn.Linear(H3, DELTA_HID), nn.ReLU(), nn.Linear(DELTA_HID, 3)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.backbone(x)
        s = self.head_success(h)
        d = self.head_delta(h)
        sig = torch.sigmoid(s)
        return s, d * sig


def add_cross_features(states, txs):
    """
    Ключевое изменение: добавляем явные кросс-признаки.
    Это позволяет нейросети сразу увидеть threshold balance >= amount.
    """
    # balance - amount (нормализованная разность)
    diff = (states[:, 6:7] - txs[:, 4:5])
    # balance / amount (отношение)
    ratio = states[:, 6:7] / (txs[:, 4:5] + 1e-6)
    
    return np.concatenate([states, txs, diff, ratio], axis=1).astype(np.float32)


def load_dataset(path="dataset.bin", device='cuda'):
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=5)
        magic, n_samples, state_dim, tx_dim, out_dim = header

        states  = np.fromfile(f, dtype=np.float32, count=n_samples * state_dim).reshape(n_samples, state_dim)
        txs     = np.fromfile(f, dtype=np.float32, count=n_samples * tx_dim).reshape(n_samples, tx_dim)
        results = np.fromfile(f, dtype=np.float32, count=n_samples * out_dim).reshape(n_samples, out_dim)

    # Кросс-признаки ДО нормализации (важно!)
    X_raw = add_cross_features(states, txs)
    
    # Нормализация всего вектора
    x_mean = X_raw.mean(axis=0)
    x_std = X_raw.std(axis=0) + 1e-6
    X = (X_raw - x_mean) / x_std

    success_mask = results[:, 0] > 0.5
    deltas_ok = results[:, 1:][success_mask]
    delta_mean = deltas_ok.mean(axis=0) if len(deltas_ok) else np.zeros(OUT_DELTA, dtype=np.float32)
    delta_std  = deltas_ok.std(axis=0)  if len(deltas_ok) else np.ones(OUT_DELTA, dtype=np.float32)
    delta_std  = np.where(delta_std < 1e-4, 1.0, delta_std)

    y_deltas_norm = (results[:, 1:] - delta_mean) / delta_std

    Path("weights").mkdir(exist_ok=True)
    np.savez("weights/norms.npz",
             x_mean=x_mean, x_std=x_std,
             delta_mean=delta_mean, delta_std=delta_std)

    return (
        torch.from_numpy(X).to(device=device, dtype=torch.float32),
        torch.from_numpy(results[:, 0:1]).to(device=device, dtype=torch.float32),
        torch.from_numpy(y_deltas_norm).to(device=device, dtype=torch.float32),
    )

def export_weights(model: nn.Module):
    state = model.state_dict()

    def to_f16(w: torch.Tensor):
        return w.detach().cpu().numpy().astype(np.float16)

    # Явно задаём порядок слоёв для бинарника
    layer_names = [
        'backbone.0',   # Linear 22->32
        'backbone.2',   # Linear 32->32
        'backbone.4',   # Linear 32->16
        'head_success.0', # Linear 16->8
        'head_success.2', # Linear 8->1
        'head_delta.0',   # Linear 16->8
        'head_delta.2',   # Linear 8->3
    ]

    layers = []
    for name in layer_names:
        w_key = f"{name}.weight"
        b_key = f"{name}.bias"
        if w_key not in state:
            raise KeyError(f"Weight {w_key} not found in state_dict. Available: {list(state.keys())}")
        layers.append((to_f16(state[w_key]), to_f16(state[b_key])))

    with open("weights/stn_weights.bin", "wb") as f:
        f.write(struct.pack("I", 0x53544E01))
        for w, b in layers:
            w.tofile(f)
            b.tofile(f)

    total = Path("weights/stn_weights.bin").stat().st_size
    expected = 4 + sum(w.size + b.size for w, b in layers) * 2
    print(f"Exported: {total} bytes (expected: {expected})")
    assert total == expected, f"Size mismatch: {total} != {expected}"


def train():
    print(f"Device: {DEVICE}")
    X, y_success, y_deltas = load_dataset("dataset.bin")
    n = X.shape[0]
    n_train = int(n * 0.9)

    X_train, X_val = X[:n_train], X[n_train:]
    y_s_train, y_s_val = y_success[:n_train], y_success[n_train:]
    y_d_train, y_d_val = y_deltas[:n_train], y_deltas[n_train:]

    norms = np.load("weights/norms.npz")
    delta_mean = torch.from_numpy(norms['delta_mean']).to(DEVICE, dtype=torch.float32)
    delta_std  = torch.from_numpy(norms['delta_std']).to(DEVICE, dtype=torch.float32)

    model = ContractNetFast().to(DEVICE)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"ContractNet with cross-features: {total_p:,} params | input={INPUT_DIM}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = (n_train + BATCH_SIZE - 1) // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=EPOCHS * steps_per_epoch,
        pct_start=0.3, anneal_strategy='cos'
    )

    best_score = -1.0

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb = X_train[idx]
            sb = y_s_train[idx]
            db = y_d_train[idx]

            optimizer.zero_grad()
            logit_s, pred_d = model(xb)

            loss_s = F.binary_cross_entropy_with_logits(logit_s, sb)

            mask = (sb > 0.5).float()
            loss_d_raw = F.smooth_l1_loss(pred_d, db, reduction='none')
            loss_d = (loss_d_raw * mask).sum() / (mask.sum() + 1e-6)

            loss = loss_s + 0.5 * loss_d
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        if epoch % 2 == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                logit_s, pred_d = model(X_val)
                probs = torch.sigmoid(logit_s)
                pred_s = (probs > 0.5).float()

                tp = ((pred_s == 1) & (y_s_val == 1)).sum().item()
                fp = ((pred_s == 1) & (y_s_val == 0)).sum().item()
                fn = ((pred_s == 0) & (y_s_val == 1)).sum().item()

                acc = (pred_s == y_s_val).float().mean().item()
                prec = tp / (tp + fp + 1e-6)
                rec  = tp / (tp + fn + 1e-6)
                f1   = 2 * prec * rec / (prec + rec + 1e-6)

                pred_d_denorm = pred_d * delta_std + delta_mean
                true_d_denorm = y_d_val * delta_std + delta_mean
                mask = (y_s_val > 0.5).squeeze()
                delta_mae = 0.0
                if mask.sum() > 0:
                    delta_mae = F.l1_loss(pred_d_denorm[mask], true_d_denorm[mask]).item()

                score = acc * 0.5 + f1 * 0.5

                if score > best_score:
                    best_score = score
                    torch.save({
                        'epoch': epoch,
                        'model_state': model.state_dict(),
                        'acc': acc, 'f1': f1, 'prec': prec, 'rec': rec,
                        'delta_mae': delta_mae,
                    }, "weights/best_model.pt")
                    print(f"  >>> NEW BEST score={score:.4f}")

                lr = scheduler.get_last_lr()[0]
                print(f"Ep {epoch:3d} | Loss: {total_loss/n_batches:.4f} | "
                      f"Acc: {acc*100:.2f}% | P/R/F1: {prec:.3f}/{rec:.3f}/{f1:.3f} | "
                      f"ΔMAE: {delta_mae:.2f} | LR: {lr:.2e}")

    ckpt = torch.load("weights/best_model.pt")
    model.load_state_dict(ckpt['model_state'])
    print(f"\nBEST ep {ckpt['epoch']}: Acc={ckpt['acc']*100:.2f}%  F1={ckpt['f1']:.4f}")

    export_weights(model)
    print("Done!")


if __name__ == "__main__":
    train()
    
    
# python train_upd.py 
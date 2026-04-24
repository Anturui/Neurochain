#!/usr/bin/env python3
"""
NeuroChain Neural Validator Training Script

Trains a small MLP to emulate classical transaction validation.
The NN learns to predict: is_transaction_valid(tx_features)?

Architecture: 256 -> 512 -> 512 -> 1
Input:  transaction features (amount, balance, nonce, sig_valid, etc.)
Output: probability of validity [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import os
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================
# INPUT_DIM = 256
# HIDDEN_DIM = 512
# OUTPUT_DIM = 1
# BATCH_SIZE = 4096
# EPOCHS = 20
# LEARNING_RATE = 1e-3

# === CONFIG ===
INPUT_DIM = 10              # 10 структурированных фичей
HIDDEN_DIM = 16             # 16 нейронов (достаточно для 5-7 AND)
OUTPUT_DIM = 1
BATCH_SIZE = 8192
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# Neural Validator Model
# =============================================================================
class NeuralValidator(nn.Module):
    """
    MLP that learns classical validation logic.

    The network must learn implicit rules:
    - valid = sig_valid AND balance >= amount AND nonce > last_nonce 
              AND program_id == valid AND is_funded
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

        # Initialize with small weights for stable training
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.1)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.xavier_uniform_(self.fc3.weight, gain=0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        # x: [batch, 256]
        h1 = F.relu(self.fc1(x))   # [batch, 512]
        h2 = F.relu(self.fc2(h1))  # [batch, 512]
        out = torch.sigmoid(self.fc3(h2))  # [batch, 1]
        return out

# =============================================================================
# Synthetic Transaction Generator (FIXED: exact class balance)
# =============================================================================
# Генератор (эмулирует реальные Solana проверки)
def generate_transactions(batch_size, invalid_ratio=0.3):
    features = torch.zeros(batch_size, 10, dtype=torch.float32)
    labels = torch.zeros(batch_size, 1, dtype=torch.float32)
    
    n_invalid = int(batch_size * invalid_ratio)
    n_valid = batch_size - n_invalid
    
    for i in range(n_valid):
        features[i, 0] = 1.0   # sig_valid
        features[i, 1] = 1.0   # balance >= amount
        features[i, 2] = 1.0   # nonce > last_nonce
        features[i, 3] = 1.0   # blockhash valid
        features[i, 4] = 1.0   # program_id valid
        features[i, 5] = 1.0   # account exists
        features[i, 6] = np.random.uniform(0, 1)   # priority fee (не влияет на валидность)
        features[i, 7] = np.random.uniform(0, 1)   # compute units
        features[i, 8] = np.random.uniform(0, 1)   # transaction size ratio
        features[i, 9] = np.random.uniform(0, 1)   # account count
        labels[i] = 1.0
    
    for i in range(n_invalid):
        idx = n_valid + i
        features[idx] = features[i % n_valid].clone()  # копируем валидную
        
        error = np.random.randint(0, 6)
        if error == 0:   features[idx, 0] = 0.0   # bad sig
        elif error == 1: features[idx, 1] = 0.0   # insufficient funds
        elif error == 2: features[idx, 2] = 0.0   # nonce reused
        elif error == 3: features[idx, 3] = 0.0   # old blockhash
        elif error == 4: features[idx, 4] = 0.0   # invalid program
        elif error == 5: features[idx, 5] = 0.0   # account not found
        
        labels[idx] = 0.0
    
    perm = torch.randperm(batch_size)
    return features[perm], labels[perm]

# =============================================================================
# Export weights to binary files for CUDA constant memory
# =============================================================================
def export_weights(model):
    """Export trained weights to binary files readable by CUDA."""

    def tensor_to_bin(tensor, filepath):
        """Convert torch tensor to raw binary (FP16)."""
        arr = tensor.detach().cpu().numpy().astype(np.float16)
        arr.tofile(filepath)
        print(f"  Exported {filepath}: shape={tensor.shape}, size={arr.nbytes} bytes")

    print("\nExporting weights to binary files...")

    # Layer 1: [256, 512]
    tensor_to_bin(model.fc1.weight.data.T, WEIGHTS_DIR / "w1.bin")
    tensor_to_bin(model.fc1.bias.data, WEIGHTS_DIR / "b1.bin")

    # Layer 2: [512, 512]
    tensor_to_bin(model.fc2.weight.data.T, WEIGHTS_DIR / "w2.bin")
    tensor_to_bin(model.fc2.bias.data, WEIGHTS_DIR / "b2.bin")

    # Layer 3: [512, 1]
    tensor_to_bin(model.fc3.weight.data.T, WEIGHTS_DIR / "w3.bin")
    tensor_to_bin(model.fc3.bias.data, WEIGHTS_DIR / "b3.bin")

    print(f"\nAll weights exported to {WEIGHTS_DIR}/")
    total_params = (256*512 + 512 + 512*512 + 512 + 512*1 + 1)
    print(f"Total size: ~{total_params * 2 / 1024:.1f} KB")

# =============================================================================
# Training
# =============================================================================
def train():
    print(f"Training Neural Validator on {DEVICE}")
    print(f"Architecture: {INPUT_DIM} -> {HIDDEN_DIM} -> {HIDDEN_DIM} -> {OUTPUT_DIM}")

    model = NeuralValidator().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Weighted loss for imbalanced classes (invalid is more important)
    pos_weight = torch.tensor([2.0]).to(DEVICE)
    criterion = nn.BCELoss()

    # Early stopping
    best_loss = float('inf')  # <-- ИСПРАВЛЕНО: было f32('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        all_preds = []
        all_labels = []

        n_batches = 100
        for _ in range(n_batches):
            features, labels = generate_transactions(BATCH_SIZE)
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(features)

            # Weighted BCE: invalid samples have higher weight
            weights = torch.where(labels == 1.0, 
                                  torch.ones_like(labels), 
                                  pos_weight.expand_as(labels))
            loss = F.binary_cross_entropy(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

        # Metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        acc = np.mean(all_preds == all_labels)

        # Precision/Recall per class
        tp = np.sum((all_preds == 1) & (all_labels == 1))
        fp = np.sum((all_preds == 1) & (all_labels == 0))
        tn = np.sum((all_preds == 0) & (all_labels == 0))
        fn = np.sum((all_preds == 0) & (all_labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        avg_loss = epoch_loss / n_batches

        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] "
              f"Loss: {avg_loss:.6f} | "
              f"Acc: {acc:.4f} | "
              f"P: {precision:.4f} | "
              f"R: {recall:.4f} | "
              f"F1: {f1:.4f} | "
              f"TP:{tp} FP:{fp} TN:{tn} FN:{fn}")

        # Early stopping
        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), "weights/best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load("weights/best_model.pt", weights_only=True))
    export_weights(model)
    return model

# =============================================================================
# Validate: compare NN vs classical on test set
# =============================================================================
def validate_model(model, n_samples=10000):
    """Compare neural validator against classical ground truth."""
    model.eval()

    features, labels = generate_transactions(n_samples)
    features = features.to(DEVICE)
    labels = labels.to(DEVICE)

    with torch.no_grad():
        outputs = model(features)
        predicted = (outputs > 0.5).float()

    accuracy = (predicted == labels).float().mean().item()

    # False positives / negatives
    false_pos = ((predicted == 1) & (labels == 0)).sum().item()
    false_neg = ((predicted == 0) & (labels == 1)).sum().item()

    print(f"\nValidation on {n_samples} samples:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  False positives: {false_pos} (NN says valid, classical says invalid)")
    print(f"  False negatives: {false_neg} (NN says invalid, classical says valid)")

    return accuracy

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("NeuroChain Neural Validator Training")
    print("=" * 60)

    model = train()
    validate_model(model)

    # Also export a test dataset for the benchmark
    print("\nGenerating validation dataset for CUDA benchmark...")
    test_features, test_labels = generate_transactions(100000)
    test_features.numpy().astype(np.float16).tofile(WEIGHTS_DIR / "test_input.bin")
    test_labels.numpy().astype(np.float16).tofile(WEIGHTS_DIR / "test_labels.bin")
    print("  test_input.bin: 100K transactions x 256 features")
    print("  test_labels.bin: 100K labels")

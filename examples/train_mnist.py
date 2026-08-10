"""
🥭 Mangograd MNIST Demo
========================
Training a simple MLP to classify handwritten digits (0-9).

Requirements: only numpy (no sklearn). The dataset is downloaded once
from Yann LeCun's mirror and cached in ~/.cache/mangograd/.

Architecture: 784 -> 128 -> 64 -> 10
Results: ~97.5% test accuracy in ~27 seconds on a modern CPU.
"""
import gzip
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np
import time

from mangograd.tensor import Tensor
from mangograd.nn import Linear, Module
from mangograd.optim import Adam
from mangograd.loss import CrossEntropyLoss
from mangograd.diagnostics import gradient_health


# ── 1. MNIST data loader (no sklearn required) ─────────────────────────
CACHE = Path.home() / ".cache" / "mangograd" / "mnist"
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}


def _download(fname: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / fname
    if not path.exists():
        print(f"  downloading {fname}...")
        urllib.request.urlretrieve(BASE_URL + fname, path)
    return path


def _load_images(fname: str) -> np.ndarray:
    with gzip.open(_download(fname), "rb") as f:
        _, n, r, c = struct.unpack(">4I", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r * c)


def _load_labels(fname: str) -> np.ndarray:
    with gzip.open(_download(fname), "rb") as f:
        f.read(8)  # skip magic + count
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist():
    print("Loading MNIST (downloading to ~/.cache/mangograd if needed)...")
    X_train = _load_images(FILES["train_images"]).astype(np.float64) / 255.0
    y_train = _load_labels(FILES["train_labels"]).astype(int)
    X_test  = _load_images(FILES["test_images"]).astype(np.float64)  / 255.0
    y_test  = _load_labels(FILES["test_labels"]).astype(int)
    return X_train, y_train, X_test, y_test


# ── 2. Model ───────────────────────────────────────────────────────────
class MNISTNet(Module):
    def __init__(self):
        self.fc1 = Linear(784, 128)
        self.fc2 = Linear(128, 64)
        self.fc3 = Linear(64, 10)

    def __call__(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        return self.fc3(x)


# ── 3. Training loop ───────────────────────────────────────────────────
def main():
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    model = MNISTNet()
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()

    total_params = sum(p.data.size for p in model.parameters())
    print(f"Parameters: {total_params:,}\n")

    epochs     = 5
    batch_size = 64
    n_batches  = len(X_train) // batch_size

    print(f"{'Epoch':>5} | {'Loss':>8} | {'Train acc':>10} | {'Time':>8}")
    print("-" * 40)

    start = time.perf_counter()

    for epoch in range(epochs):
        idx = np.random.permutation(len(X_train))
        X_s, y_s = X_train[idx], y_train[idx]
        total_loss = correct = 0

        for b in range(n_batches):
            sl = slice(b * batch_size, (b + 1) * batch_size)
            X_b = Tensor(X_s[sl])
            y_b = y_s[sl]

            logits = model(X_b)
            loss = criterion(logits, y_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.data
            correct    += (np.argmax(logits.data, axis=1) == y_b).sum()

        acc = correct / (n_batches * batch_size) * 100
        print(f"{epoch+1:>5} | {total_loss/n_batches:>8.4f} | {acc:>9.1f}% | {time.perf_counter()-start:>7.1f}s")

    # ── 4. Test set evaluation ─────────────────────────────────────────
    correct = 0
    test_batches = len(X_test) // batch_size
    for b in range(test_batches):
        sl = slice(b * batch_size, (b + 1) * batch_size)
        logits = model(Tensor(X_test[sl]))
        correct += (np.argmax(logits.data, axis=1) == y_test[sl]).sum()

    test_acc = correct / (test_batches * batch_size) * 100
    elapsed  = time.perf_counter() - start

    print(f"\n{'=' * 40}")
    print(f"🥭 MNIST test accuracy : {test_acc:.2f}%")
    print(f"⏱  Total time          : {elapsed:.1f}s")
    print(f"{'=' * 40}\n")

    # ── 5. Gradient health check ───────────────────────────────────────
    logits = model(Tensor(X_test[:64]))
    loss   = criterion(logits, y_test[:64])
    optimizer.zero_grad()
    loss.backward()
    gradient_health(model)

    # ── 6. Save ────────────────────────────────────────────────────────
    saved = model.save_state_dict("mnist_weights")
    print(f"\nWeights saved to {saved}")


if __name__ == "__main__":
    main()

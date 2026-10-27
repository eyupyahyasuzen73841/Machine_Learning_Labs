import os
import sys
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neural_scratch.model_graph import EyupMultiLayerNet
from neural_scratch.grad_check import run_gradient_check

def main():
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=73841)

    model = EyupMultiLayerNet([64, 40, 20, 10], seed=73841)
    max_err = run_gradient_check(model, X_tr[:32], y_tr[:32])
    print(f"[CHECK] Finite Difference Relative Error: {max_err:.3e}")

    start = time.perf_counter()
    epochs = 45
    bs = 32
    for ep in range(epochs):
        perm = np.random.permutation(len(X_tr))
        for i in range(0, len(X_tr), bs):
            idx = perm[i:i+bs]
            model.forward(X_tr[idx], y_tr[idx])
            model.backward()
            model.step(lr=0.15)
    train_time = time.perf_counter() - start

    preds = np.argmax(model.forward(X_te), axis=1)
    acc = np.mean(preds == y_te) * 100.0
    print(f"[EVAL] Test Accuracy: {acc:.2f}% | Wall Time: {train_time:.3f}s")

if __name__ == "__main__":
    main()

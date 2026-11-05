"""
End-to-End Classification Trainer on make_moons: Custom Engine vs. PyTorch
Student Researcher: Eyüp Yahya Süzen (ID: 73841)
Architecture: 2 -> 8 -> 1 (ReLU hidden, Sigmoid output, BCE loss)
"""

import time
import random
import statistics
import json
import os
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from core_scalar import GradScalar


class SynapseNeuron:
    def __init__(self, fan_in: int, use_relu: bool = True):
        # Xavier uniform initialization
        bound = (6.0 / fan_in) ** 0.5
        self.weights = [GradScalar(random.uniform(-bound, bound)) for _ in range(fan_in)]
        self.bias = GradScalar(0.0)
        self.use_relu = use_relu

    def __call__(self, x_in):
        acc = self.bias
        for w, x in zip(self.weights, x_in):
            acc = acc + (w * x)
        return acc.relu() if self.use_relu else acc

    def get_params(self):
        return self.weights + [self.bias]


class PerceptronLayer:
    def __init__(self, fan_in: int, fan_out: int, use_relu: bool = True):
        self.units = [SynapseNeuron(fan_in, use_relu=use_relu) for _ in range(fan_out)]

    def __call__(self, inputs):
        outputs = [unit(inputs) for unit in self.units]
        return outputs[0] if len(outputs) == 1 else outputs

    def get_params(self):
        p = []
        for u in self.units:
            p.extend(u.get_params())
        return p


class MoonsNetwork:
    def __init__(self):
        self.layer1 = PerceptronLayer(2, 8, use_relu=True)
        self.layer2 = PerceptronLayer(8, 1, use_relu=False)

    def forward(self, x):
        h = self.layer1(x)
        raw = self.layer2(h)
        return raw.sigmoid()

    def get_all_params(self):
        return self.layer1.get_params() + self.layer2.get_params()

    def reset_gradients(self):
        for param in self.get_all_params():
            param.grad = 0.0


class PyTorchMoonsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def compute_bce_loss(targets, predictions):
    n = len(targets)
    losses = []
    for y_true, y_prob in zip(targets, predictions):
        if y_true == 1:
            loss_item = -y_prob.log()
        else:
            loss_item = -(GradScalar(1.0) - y_prob).log()
        losses.append(loss_item)
    
    total = losses[0]
    for li in losses[1:]:
        total = total + li
    return total * (1.0 / n)


def train_custom_instance(X, y, epochs=35, lr=0.25):
    random.seed(2026)
    model = MoonsNetwork()
    X_scalars = [[GradScalar(val) for val in row] for row in X]
    
    t0 = time.perf_counter()
    loss_history = []
    
    for epoch in range(epochs):
        preds = [model.forward(x) for x in X_scalars]
        loss = compute_bce_loss(y, preds)
        
        model.reset_gradients()
        loss.backward()
        
        for p in model.get_all_params():
            p.value -= lr * p.grad
            
        loss_history.append(loss.value)
        
    duration = time.perf_counter() - t0
    acc = sum((1 if p.value >= 0.5 else 0) == yt for p, yt in zip(preds, y)) / len(y)
    return duration, loss_history[-1], acc


def train_pytorch_instance(X, y, epochs=35, lr=0.25):
    torch.manual_seed(2026)
    model = PyTorchMoonsNet()
    criterion = nn.BCELoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    
    tx = torch.tensor(X, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    t0 = time.perf_counter()
    loss_history = []
    
    for epoch in range(epochs):
        opt.zero_grad()
        out = model(tx)
        loss = criterion(out, ty)
        loss.backward()
        opt.step()
        loss_history.append(loss.item())
        
    duration = time.perf_counter() - t0
    with torch.no_grad():
        preds = (model(tx) >= 0.5).float()
        acc = (preds == ty).float().mean().item()
    return duration, loss_history[-1], acc


def benchmark_5_runs():
    print("[EYUP SUZEN - BENCHMARK] Executing 5-run make_moons benchmark...")
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
    
    c_times = []
    p_times = []
    c_losses = []
    p_losses = []
    
    for trial in range(5):
        t_c, l_c, a_c = train_custom_instance(X, y, epochs=35, lr=0.25)
        t_p, l_p, a_p = train_pytorch_instance(X, y, epochs=35, lr=0.25)
        
        c_times.append(t_c * 1000.0)
        p_times.append(t_p * 1000.0)
        c_losses.append(l_c)
        p_losses.append(l_p)
        print(f"  Run {trial+1}/5 | Custom: {c_times[-1]:.2f} ms | PyTorch: {p_times[-1]:.2f} ms")
        
    c_median = statistics.median(c_times)
    p_median = statistics.median(p_times)
    ratio = c_median / p_median
    
    output = {
        'student': 'Eyüp Yahya Süzen (73841)',
        'runs': 5,
        'custom_median_ms': c_median,
        'pytorch_median_ms': p_median,
        'slowdown_factor': ratio,
        'custom_losses': c_losses,
        'pytorch_losses': p_losses,
        'custom_all_ms': c_times,
        'pytorch_all_ms': p_times
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/moons_benchmark_5trials.json', 'w') as f:
        json.dump(output, f, indent=2)
        
    print(f"\n[SUMMARY] Custom Median: {c_median:.2f} ms | PyTorch Median: {p_median:.2f} ms | Ratio: {ratio:.1f}x")
    return output


if __name__ == '__main__':
    benchmark_5_runs()

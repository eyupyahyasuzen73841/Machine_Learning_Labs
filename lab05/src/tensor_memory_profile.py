"""
PyTorch Computational Graph Tracing & Tensor Memory Contiguity Profiler
Student Researcher: Eyüp Yahya Süzen (ID: 73841)
Curriculum: Machine Learning Systems (Lab 05)
"""

import time
import json
import os
import torch
import torch.nn as nn


class MoonsEvaluationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.act(self.fc1(x))))


def profile_jit_and_strides():
    print("[EYUP SUZEN] Profiling TorchScript JIT tracing and stride effects...")
    model = MoonsEvaluationModel().eval()
    sample = torch.randn(64, 2, dtype=torch.float32)

    # 1. Traced graph
    traced = torch.jit.trace(model, sample)
    ir_text = str(traced.inlined_graph)

    # 2. Timing eager vs traced after warm-up
    warmup = 100
    loops = 1000

    for _ in range(warmup):
        _ = model(sample)
        _ = traced(sample)

    t0 = time.perf_counter()
    for _ in range(loops):
        _ = model(sample)
    eager_us = ((time.perf_counter() - t0) / loops) * 1e6

    t0 = time.perf_counter()
    for _ in range(loops):
        _ = traced(sample)
    traced_us = ((time.perf_counter() - t0) / loops) * 1e6

    print(f"  Eager Dispatch:  {eager_us:.2f} µs")
    print(f"  Traced Dispatch: {traced_us:.2f} µs (Speedup: {eager_us/traced_us:.2f}x)")

    # 3. Memory Contiguity and Strided Reductions
    DIM = 2048
    T_contig = torch.randn(DIM, DIM, dtype=torch.float32)
    T_non_contig = T_contig.t()

    c_info = {
        'shape': list(T_contig.shape),
        'stride': list(T_contig.stride()),
        'is_contiguous': T_contig.is_contiguous()
    }
    nc_info = {
        'shape': list(T_non_contig.shape),
        'stride': list(T_non_contig.stride()),
        'is_contiguous': T_non_contig.is_contiguous()
    }

    reps = 200
    for _ in range(20):
        _ = torch.sum(T_contig, dim=1)
        _ = torch.sum(T_non_contig, dim=1)

    t0 = time.perf_counter()
    for _ in range(reps):
        _ = torch.sum(T_contig, dim=1)
    contig_ms = ((time.perf_counter() - t0) / reps) * 1000.0

    t0 = time.perf_counter()
    for _ in range(reps):
        _ = torch.sum(T_non_contig, dim=1)
    non_contig_ms = ((time.perf_counter() - t0) / reps) * 1000.0

    t0 = time.perf_counter()
    for _ in range(reps):
        _ = T_non_contig.contiguous()
    copy_ms = ((time.perf_counter() - t0) / reps) * 1000.0

    penalty = non_contig_ms / contig_ms
    print(f"  Contiguous Row-Sum:     {contig_ms:.3f} ms")
    print(f"  Non-Contiguous Row-Sum: {non_contig_ms:.3f} ms")
    print(f"  Stride Penalty:         {penalty:.2f}x")
    print(f"  Contiguous Memory Copy: {copy_ms:.3f} ms")

    telemetry = {
        'student': 'Eyüp Yahya Süzen (73841)',
        'eager_latency_us': eager_us,
        'traced_latency_us': traced_us,
        'traced_speedup': eager_us / traced_us,
        'contiguous_meta': c_info,
        'non_contiguous_meta': nc_info,
        'contig_sum_ms': contig_ms,
        'non_contig_sum_ms': non_contig_ms,
        'stride_penalty_factor': penalty,
        'contiguous_allocation_cost_ms': copy_ms,
        'ir_snippet': ir_text[:600]
    }

    os.makedirs('results', exist_ok=True)
    with open('results/stride_latency_profile.json', 'w') as f:
        json.dump(telemetry, f, indent=2)

    return telemetry


if __name__ == '__main__':
    profile_jit_and_strides()

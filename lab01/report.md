# Workstation Profiling & Execution Runtime Telemetry (Lab 01)
**Investigator:** Eyüp Yahya Süzen  
**Student ID:** 73841  
**Department:** Computer Science and Engineering  

---

## 1. Benchmark Scope & Hardware Objectives
Modern machine learning applications are constrained by underlying hardware memory bandwidth and CPU/GPU dispatch latencies. This laboratory benchmarks runtime performance, memory consumption curves, and execution stability across dense floating-point tensor operations.

## 2. Experimental Setup & Measurement Stack
- **OS & Environment:** Windows 64-bit platform, Python 3.11 isolated runtime.
- **Microbenchmarking Suite:** Dense square floating-point matrix multiplications ($N \times N$) across dimension tiers $N \in \{256, 512, 1024, 2048\}$.
- **Timing & Memory Instruments:** High-precision monotonic timestamps (`time.perf_counter`) and process RSS tracking via `psutil`.

## 3. Workstation Performance Metrics & Analysis

### Empirical Matrix Multiplication Telemetry
| Tensor Size ($N \times N$) | FLOP Count ($2N^3$) | Execution Time (ms) | Computational Intensity (GFLOPs) | Peak Memory Allocation (MB) |
| :---: | :---: | :---: | :---: | :---: |
| $256 \times 256$ | $3.36 \times 10^7$ | 1.79 | 18.7 | 0.50 |
| $512 \times 512$ | $2.68 \times 10^8$ | 11.90 | 22.5 | 2.05 |
| $1024 \times 1024$ | $2.15 \times 10^9$ | 84.15 | 25.5 | 8.20 |
| $2048 \times 2048$ | $1.72 \times 10^{10}$ | 635.40 | 27.1 | 32.80 |

```
Execution Time Scaling Profile:
  256x256   :  1.79 ms [■]
  512x512   : 11.90 ms [■■■]
  1024x1024 : 84.15 ms [■■■■■■■■■■■]
  2048x2048 : 635.4 ms [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]
```

## 4. Key Takeaways & Profiling Insights (Three Points)
1. **Vectorization Saturates Hardware ALUs:** Matrix multiplication achieves up to 27.1 GFLOPs on multi-threaded CPU subroutines, highlighting the dramatic benefit of hardware SIMD vectorization over standard nested Python loops.
2. **Memory Stride Overhead Increases with Dimension:** Doubling dimension $N$ beyond 1024 causes memory footprints to surpass L3 cache bounds, introducing memory bus latency penalties that slightly reduce scaling efficiency.
3. **Environment Standardization Eliminates Drift:** Creating pinned virtual environments guarantees reproducible linear algebra execution across heterogenous operating systems.

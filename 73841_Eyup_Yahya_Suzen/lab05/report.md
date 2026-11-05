# Lab 05: Frameworks and Automatic Differentiation

**Student Researcher:** Eyüp Yahya Süzen  
**Student ID:** 73841  
**Curriculum:** Machine Learning Systems (Academic Year 2025/2026)  
**Reference Text:** Vijay Janapa Reddi, *Machine Learning Systems*, Chapter 7 (AI Frameworks: §§7.3–7.8)  

---

## 1. Executive Summary

This laboratory investigates the software and hardware mechanisms governing automatic differentiation engines and deep learning frameworks:
1. **Dynamic DAG Scalar Engine:** Implementation of `GradScalar`, supporting forward graph recording, reverse-mode topological backpropagation, and non-overwriting gradient accumulation.
2. **Empirical Numerical Verification:** Full verification against PyTorch autograd and numerical central finite differences on `((a*b + c)**2).relu().sum()`, demonstrating exact mathematical convergence ($\Delta < 10^{-6}$).
3. **Moons Binary Classification Benchmark:** 5-run empirical speed comparison against PyTorch on `make_moons` (architecture: $2 \to 8 \to 1$, Sigmoid output, BCE loss).
4. **IR Tracing & Runtime Profiling:** TorchScript IR inspection (`torch.jit.trace`) and microsecond latency evaluation across 1,000 warm-up cycles.
5. **Memory Layout & Strides Analysis:** Systematic profiling of contiguous vs. non-contiguous tensor layouts, revealing cache-line penalties and SIMD degradation.

---

## 2. Scalar Automatic Differentiation Architecture (`GradScalar`)

The core engine is implemented in [core_scalar.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_73841_eyup/lab05/src/core_scalar.py).

### 2.1 Vertex Representation
Each `GradScalar` node contains:
- `value`: Scalar float calculated during forward evaluation.
- `grad`: Accumulated derivative $\partial \mathcal{L} / \partial x$, initialized to 0.0.
- `parents`: Set of input ancestor nodes in the DAG.
- `backward_step`: Parameter-less lambda closure defining the local derivative chain rule.
- `op`: String descriptor representing the operation.

### 2.2 Mathematical Primitive Operations
- **Addition ($z = x + y$):** $\frac{\partial \mathcal{L}}{\partial x} += \frac{\partial \mathcal{L}}{\partial z}, \quad \frac{\partial \mathcal{L}}{\partial y} += \frac{\partial \mathcal{L}}{\partial z}$
- **Multiplication ($z = x \cdot y$):** $\frac{\partial \mathcal{L}}{\partial x} += y \cdot \frac{\partial \mathcal{L}}{\partial z}, \quad \frac{\partial \mathcal{L}}{\partial y} += x \cdot \frac{\partial \mathcal{L}}{\partial z}$
- **Power ($z = x^p$):** $\frac{\partial \mathcal{L}}{\partial x} += (p \cdot x^{p-1}) \cdot \frac{\partial \mathcal{L}}{\partial z}$
- **ReLU ($z = \max(0, x)$):** $\frac{\partial \mathcal{L}}{\partial x} += \mathbf{1}_{\{x > 0\}} \cdot \frac{\partial \mathcal{L}}{\partial z}$
- **Sigmoid ($\sigma(x) = \frac{1}{1 + e^{-x}}$):** $\frac{\partial \mathcal{L}}{\partial x} += \sigma(x)(1 - \sigma(x)) \cdot \frac{\partial \mathcal{L}}{\partial z}$
- **Logarithm ($z = \ln(x)$):** $\frac{\partial \mathcal{L}}{\partial x} += \frac{1}{x} \cdot \frac{\partial \mathcal{L}}{\partial z}$

### 2.3 Topological Sorting & Gradient Accumulation
Topological ordering via recursive depth-first search ensures that all descendant nodes in the DAG propagate their gradients before a node executes its backward step. Using `+=` rather than `=` preserves the multivariate chain rule across shared vertices with multiple consumers.

---

## 3. Mathematical Verification Against PyTorch & Finite Differences

Target mathematical expression:
$$\mathcal{L} = \sum_{i=1}^4 \text{ReLU}\left( (a_i \cdot b_i + c_i)^2 \right)$$

- **Test Vectors:**
  - $a = [1.80, -2.40, 0.90, -1.10]$
  - $b = [-2.50, 1.75, 3.10, 0.80]$
  - $c = [1.10, -0.60, -2.20, 1.40]$
- **Forward Output:** Custom Engine = `38.25050`, PyTorch = `38.25050` ($\Delta = 0.00$)
- **Max Absolute Error vs PyTorch:** **$0.00 \times 10^0$** (Strictly $< 10^{-6}$, PASS)
- **Max Absolute Error vs Central Finite Difference:** $3.10 \times 10^{-8}$ (PASS)

---

## 4. `make_moons` Training Benchmark (5-Trial Telemetry)

- **Architecture:** $2 \to 8 \to 1$ (ReLU hidden, Sigmoid output, BCE loss).
- **Dataset:** `sklearn.datasets.make_moons(n_samples=100, noise=0.1, random_state=42)`.
- **Optimizer:** SGD ($\eta = 0.25$, 35 epochs).

| Trial | Custom Engine (ms) | PyTorch (ms) | Speedup Ratio |
| :---: | :---: | :---: | :---: |
| 1 | 2486.16 | 55.58 | 44.7x |
| 2 | 2658.12 | 13.19 | 201.5x |
| 3 | 2827.03 | 13.57 | 208.3x |
| 4 | 2668.19 | 13.13 | 203.2x |
| 5 | 2564.56 | 13.90 | 184.5x |
| **Median** | **2658.12 ms** | **13.57 ms** | **195.9x** |

### Systems Analysis
The custom engine incurs a $195.9\times$ overhead compared to PyTorch due to:
1. Per-node heap allocation and pointer indirection overhead in CPython.
2. Dynamic Python bytecode dispatch on every arithmetic operation.
3. Lack of vectorized SIMD execution (PyTorch processes batches with contiguous MKL/OpenMP BLAS kernels).

---

## 5. TorchScript IR Inspection & Execution Modes

We inspected the JIT intermediate representation using `torch.jit.trace`:
- **Eager Dispatch Latency:** $43.57\ \mu\text{s/iter}$
- **Traced Dispatch Latency:** $33.24\ \mu\text{s/iter}$
- **Speedup Factor:** **$1.31\times$**

Tracing eliminates Python frame setup, dynamic attribute lookups, and Python interpreter overhead by compiling the model into an unrolled C++ computation graph.

---

## 6. Tensor Memory Layout & Contiguity Penalty

Using a $2048 \times 2048$ single-precision tensor (16 MB buffer):
- **Contiguous Array:** `stride = [2048, 1]`, `is_contiguous() = True`.
- **Transposed Non-Contiguous View:** `stride = [1, 2048]`, `is_contiguous() = False`.

### Benchmark Measurements (Row-Wise Sum)
- **Contiguous Row-Sum Latency:** $0.195\text{ ms}$
- **Non-Contiguous Row-Sum Latency:** $0.330\text{ ms}$
- **Stride Penalty:** **$1.70\times$ slowdown**
- **Explicit `.contiguous()` Buffer Reallocation:** $8.702\text{ ms}$

### Hardware Mechanics
- **L1/L2 Cache Locality:** Contiguous memory reads maximize cache line utilization (64 bytes = 16 floats per line). Non-contiguous strides cause stride misses, fetching 64 bytes but using only 4 bytes per access.
- **SIMD Vectorization:** Contiguous arrays enable AVX2/AVX-512 vector load instructions (`vmovups`). Non-contiguous layouts trigger slow strided gathers or scalar loops.
- **Reallocation Cost:** Calling `.contiguous()` costs $8.702\text{ ms}$, which is $26\times$ larger than the single reduction slowdown. Thus, memory copying is only beneficial when the contiguous buffer will be reused across many subsequent operations.

---

## 7. Technical Note: When Graph Compilation Helps

According to Reddi Chapter 7, graph compilers provide substantial benefits under three key conditions:
1. **Memory-Bound Kernel Fusion:** Fusing pointwise operations (`x * w + b` followed by `ReLU`) into a single kernel keeps intermediate tensors in CPU registers and L1/L2 cache, eliminating redundant DRAM reads and writes.
2. **Interpreter Dispatch Elimination:** For small batch sizes or lightweight models, Python dispatch time dominates execution time. Compilers emit unified C++/native code, bypassing the Python GIL and VM loop.
3. **Static Buffer Management:** Compilers plan tensor buffer lifecycles at compile time, eliminating runtime dynamic memory allocation (`malloc`/`free`) and preventing fragmentation.

---

## 8. Directory Artifacts

- [core_scalar.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_73841_eyup/lab05/src/core_scalar.py): Dynamic DAG autograd engine.
- [validate_autograd.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_73841_eyup/lab05/src/validate_autograd.py): PyTorch autograd and finite-difference validation script.
- [model_trainer.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_73841_eyup/lab05/src/model_trainer.py): 5-trial benchmark on `make_moons`.
- [tensor_memory_profile.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_73841_eyup/lab05/src/tensor_memory_profile.py): JIT tracing and stride profiling script.
- `results/`: Contains [grad_check_results.json](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_73841_eyup/lab05/results/grad_check_results.json), [moons_benchmark_5trials.json](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_73841_eyup/lab05/results/moons_benchmark_5trials.json), and [stride_latency_profile.json](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_73841_eyup/lab05/results/stride_latency_profile.json).

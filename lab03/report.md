# Empirical Profiling & Hardware Cost Evaluation for Vision Models (Lab 03)
**Investigator:** Eyüp Yahya Süzen  
**Student ID:** 73841  
**Department:** Computer Science and Engineering  
**Module:** Machine Learning Systems Engineering  

---

## 1. Experimental Goals & Scope
Deploying deep neural networks in performance-critical runtime environments requires balancing model predictive capacity against physical computational limits. The objective of this lab is to benchmark three mathematically distinct computer vision topologies under controlled hardware conditions using Fashion-MNIST:

1. **Multilayer Perceptron (`EyupMLPClassifier`)**: Unstructured affine mappings with LeakyReLU activations.
2. **Convolutional Network (`EyupLeNetVariant`)**: Translationally equivariant feature extractors with 2D spatial average pooling.
3. **Patch-Based Vision Transformer (`EyupVisionTransformer`)**: 49-patch sequence decomposition with multi-head self-attention.

Key questions addressed:
- How does inductive bias influence parameter efficiency and final classification accuracy?
- Why do theoretical FLOPs frequently decouple from empirical wall-clock latency?
- What are the concrete hardware bottlenecks governing each architectural archetype?

---

## 2. Experimental Methodology & Model Implementations

### Topology Design Summary
```
[1] EyupMLPClassifier:
    Input (28x28) ──> Flatten (784) ──> Linear(784, 300) ──> LeakyReLU(0.1) 
                                    ──> Linear(300, 100) ──> LeakyReLU(0.1) 
                                    ──> Linear(100, 10)

[2] EyupLeNetVariant:
    Input (1x28x28) ──> Conv2d(1->16, k=5, p=2) ──> ReLU ──> AvgPool2d(2x2)
                    ──> Conv2d(16->32, k=5, p=0) ──> ReLU ──> AvgPool2d(2x2)
                    ──> Flatten (800) ──> Linear(800, 120) ──> ReLU ──> Linear(120, 10)

[3] EyupVisionTransformer:
    Input (1x28x28) ──> 4x4 Patch Unfold ──> 49 Patches of 16 px
                    ──> Linear Projection (16 -> 48) + Positional Embeddings
                    ──> 2x Transformer Blocks (d_model=48, heads=4, d_ff=96)
                    ──> Global Sequence Average Pooling ──> Linear(48, 10)
```

### Profiling Instrumentation
All models were evaluated on the unified Fashion-MNIST test benchmark. Computational FLOPs were measured via torch-profile instrumentation, and latency was recorded through high-resolution monotonic timer blocks with 15 initial burn-in iterations to eliminate cold-start overheads.

---

## 3. Results, Visualizations, and Trade-Off Curves

### Latency & Compute Benchmarking Table

| Model Architecture | Parameter Volume | Theoretical Compute (MFLOPs) | Batch Latency (ms) | Peak Activation Memory (KB) | Empirical Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fully Connected MLP (`EyupMLPClassifier`)** | 266,710 | 0.53 | 0.38 | 64.2 | 88.54% |
| **LeNet-5 Variant (`EyupLeNetVariant`)** | 110,698 | 2.41 | 0.92 | 112.5 | 91.12% |
| **Patch Transformer (`EyupVisionTransformer`)** | 42,970 | 2.26 | 1.58 | 256.0 | 85.18% |

### Hardware Bottleneck Analysis: Memory Bandwidth vs Arithmetic Intensity
- **MLP Bottleneck (Weight Bandwidth):** The MLP contains 266k parameters but requires only 0.53 MFLOPs. Because arithmetic intensity is low, inference is heavily memory-bandwidth bound as large weight matrices must be fetched from RAM for relatively few arithmetic operations.
- **CNN Bottleneck (Compute Bound):** Convolutions reuse the same kernel weights across all spatial patch positions, providing extremely high arithmetic intensity that saturates modern SIMD vector units efficiently.
- **Transformer Bottleneck (Dispatch & Attention Overhead):** The Vision Transformer has the smallest parameter footprint (43k), yet exhibits the longest latency (**1.58 ms**). This latency inflation is caused by multi-head attention matrix projections, sequence permutations, and dynamic softmax attention matrices that incur memory bus stalls.

### Profiler Operation Accounting (FLOPs vs. MACs)
Theoretical floating-point operations follow the standard identity:
$$\text{FLOPs} = 2 \times \text{MACs}$$
The LeNet variant performs $\approx 1.20\times 10^6$ multiply-accumulate operations, corresponding to $2.41$ MFLOPs in the results table.

---

## 4. Key Takeaways & Technical Insights (Three Conclusions)

1. **Parameter Efficiency Favors Inductive Bias:**  
   The CNN achieved superior test accuracy (**91.12%**) with under half the parameters of the MLP. Enforcing 2D shift invariance natively in the architecture dramatically outclasses unconstrained linear projections for visual pattern recognition.

2. **FLOPs Do Not Equal Wall-Clock Performance:**  
   The Vision Transformer executed fewer FLOPs (2.26 MFLOPs) than the CNN (2.41 MFLOPs), yet took **71.7% longer** to complete inference. Memory-access patterns and multi-head attention reshapes dominate actual runtime costs far more than raw operation counts.

3. **Compact Transformers Need Spatial Hybrids:**  
   Directly applying self-attention on tiny $4\times 4$ patches without pre-training results in noticeable accuracy degradation (**85.18%** vs **91.12%**). For edge-scale vision systems, CNNs or hybrid Conv-Transformer topologies remain the optimal engineering choice.

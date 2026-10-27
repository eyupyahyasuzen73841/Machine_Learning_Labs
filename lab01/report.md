# Lab 01: Systems Profiling & Environment Setup

**Student Researcher:** Eyüp Yahya Süzen (Student ID: 73841)  
**Readings:** Reddi, Chapter 1 ML Systems: Overview; Chapter 2 Systems Topics: §§2.1–2.2  

---

## 1. Goal
The primary objective of this laboratory is to establish a fully reproducible Python machine learning runtime environment, evaluate two baseline classification models on the Breast Cancer Wisconsin diagnostic dataset (569 samples, 30 continuous features), quantitatively measure systems performance metrics including training duration, single-sample inference latency, storage footprint, and memory utilization, and assess the feasibility of deploying these architectures across distinct deployment tiers (Cloud, Mobile/Edge, and Microcontroller/TinyML).

---

## 2. Method
1. **Environment Setup & Data Standardization:** Pinned package dependencies (`requirements.txt`). Loaded the Breast Cancer Wisconsin dataset from scikit-learn, partitioned into 70% training (398 instances) and 30% testing (171 instances) using stratified splitting pinned to `random_state=42`.
2. **Model Implementations:**
   - **Logistic Regression:** Parametric linear classifier trained with L-BFGS solver (`max_iter=1000`, `random_state=42`).
   - **Random Forest:** Ensemble of 100 decision trees (`n_estimators=100`, `random_state=42`).
3. **Systems Telemetry & Measurement Protocol:**
   - **Training Time:** Measured median wall-clock time over 5 runs using `time.perf_counter()`.
   - **Inference Latency:** Measured median single-sample prediction latency over 100 repeated runs.
   - **Peak Memory:** Profiled using `memory_profiler.memory_usage` for both training and inference phases.
   - **Storage Footprint:** Serialized model artifacts to disk using `joblib` and recorded file sizes.
4. **Hardware Mapping:** Analyzed deployment viability across TinyML (ARM Cortex-M), Mobile Edge, and Cloud server targets.

---

## 3. Results Table or Plot

### Empirical Systems Profiling Matrix

| Model Architecture | Test Accuracy | Median Fit Time (ms) | Median Inference Latency (ms) | Serialized Disk Size | Peak Training RAM (MB) | Peak Inference RAM (MB) | Feasible Deployment Tier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | **94.15%** | 117.21 ms | **0.1830 ms** | **1,103 B (1.08 KB)** | 169.02 MB | 147.44 MB | **TinyML / Edge Microcontrollers** |
| **Random Forest (100 Trees)** | **93.57%** | 114.60 ms | **7.4594 ms** | **290,889 B (284.07 KB)** | 149.39 MB | 149.39 MB | **Mobile Devices / Cloud Instances** |

### Deployment Tier Feasibility Assessment

| Hardware Tier | Typical Resources | Model Compatibility | Architectural Justification |
| :--- | :--- | :--- | :--- |
| **TinyML / Microcontroller** | < 256 KB Flash, < 64 KB RAM | **Logistic Regression Only** | With a footprint of only ~1.1 KB and 31 parameters, Logistic Regression fits comfortably within microcontroller SRAM. Random Forest (284 KB) exceeds standard micro-controller flash memory. |
| **Mobile / Edge** | 2–8 GB RAM, Multi-Core CPU | **Both Models** | Both models execute in sub-millisecond to low-millisecond regimes, suitable for embedded mobile runtime. |
| **Cloud Server** | High RAM, Multi-Core / GPU | **Both Models** | Both models easily execute in cloud environments; however, deploying compact linear models to cloud servers introduces unnecessary network latency compared to on-device edge execution. |

---

## 4. Three Conclusions in Your Own Words
1. **Storage Footprint Disparity Shapes Deployment Reality:** Logistic Regression occupies just 1,103 bytes (1.08 KB) on disk compared to Random Forest's 290,889 bytes (284 KB)—a 263x size increase with no accuracy advantage (94.15% vs 93.57%), making Logistic Regression vastly superior for resource-constrained edge systems.
2. **Inference Latency Reflects Algorithmic Traversal Overhead:** Single-sample inference latency for Logistic Regression is 0.1830 ms, while Random Forest requires 7.4594 ms (40.8x slower) due to traversing 100 deep decision trees versus computing a single vector dot product.
3. **Parametric Models Minimize Systems Bottlenecks in Low-Power Medical Telemetry:** In diagnostic sensor devices where power, memory, and latency are tightly budgeted, linear parametric models offer optimal trade-offs by providing competitive clinical accuracy with negligible memory allocation and microsecond responsiveness.
